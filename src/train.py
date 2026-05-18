# src/train.py
# ============================================================
# Eğitim döngüsü, erken durdurma mekanizması ve
# öğrenme hızı planlayıcısını (scheduler) içerir.
#
# Contains training loop, early stopping mechanism,
# and learning rate scheduler.
# ============================================================

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import config
from src.utils import save_checkpoint, get_logger

logger = get_logger()


# ─────────────────────────────────────────────────────────────
# ERKEN DURDURMA / Early Stopping
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Doğrulama kaybı (validation loss) belirtilen sabır
    (patience) epoch'u boyunca iyileşmediğinde eğitimi durdurur.

    Stops training when validation loss doesn't improve
    for 'patience' consecutive epochs.

    Bu neden önemli? / Why is this important?
    ──────────────────────────────────────────
    Fazla epoch eğitmek modelin train verisini 'ezberlemesine'
    (overfitting) yol açar. Erken durdurma ile hem zaman hem de
    bu riski azaltırız.

    Training too many epochs leads to overfitting (memorizing
    training data). Early stopping saves time and prevents it.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        """
        Args:
            patience  : Kaç epoch iyileşmesiz beklensin
            min_delta : "İyileşme" sayılmak için minimum azalma
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0                # Ardışık iyileşmesiz epoch sayacı
        self.best_loss  = float("inf")    # Şimdiye kadar görülen en iyi val loss
        self.best_epoch = 0
        self.should_stop = False           # Eğitimi durdurma sinyali

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Her epoch sonunda çağrılır. True döndürürse eğitimi durdur.
        Called after each epoch. Returns True to stop training.
        """
        if val_loss < (self.best_loss - self.min_delta):
            # Yeterince iyileşme var → sayacı sıfırla, en iyiyi güncelle
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            return False  # Devam et / continue
        else:
            # İyileşme yok → sayacı artır
            self.counter += 1
            logger.info(
                f"  EarlyStopping: {self.counter}/{self.patience} epoch "
                f"iyileşme yok (best val_loss={self.best_loss:.6f} @ epoch {self.best_epoch})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"  [!] Early Stopping tetiklendi / triggered! "
                    f"En iyi epoch / Best epoch: {self.best_epoch}"
                )
                return True  # Dur / stop
            return False


# ─────────────────────────────────────────────────────────────
# TEK EPOCH EĞİTİMİ / Single Epoch Training
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
    grad_clip : float = 1.0
) -> float:
    """
    Bir epoch boyunca tüm training batch'lerini işler.
    Processes all training batches for one epoch.

    Gradient clipping nedir? / What is gradient clipping?
    ───────────────────────────────────────────────────────
    RNN'lerde bazen gradyanlar çok büyür (exploding gradient).
    Gradyanların normunu 'grad_clip' değeriyle sınırlayarak
    kararsız güncellemeleri önleriz.

    Returns:
        Epoch'un ortalama eğitim kaybı / Average training loss
    """
    model.train()  # Dropout ve BatchNorm'u eğitim moduna al
    total_loss = 0.0

    for X_batch, y_batch in loader:
        # Veriyi GPU/CPU'ya taşı
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # --- İleri geçiş / Forward pass ---
        predictions = model(X_batch)     # [Batch, Horizon]
        loss        = criterion(predictions, y_batch)

        # --- Geri yayılım / Backward pass ---
        optimizer.zero_grad()  # Önceki gradyanları sıfırla
        loss.backward()        # Gradyanları hesapla

        # --- Gradient clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # --- Parametre güncelleme / Weight update ---
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# DOĞRULAMA / Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()   # Gradyan hesaplamasını devre dışı bırak → hız + bellek tasarrufu
def evaluate_loader(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device
) -> float:
    """
    Verilen DataLoader üzerinde ortalama kaybı hesaplar.
    Calculates average loss over the given DataLoader.
    (Eğitim parametrelerini güncellemez / Does not update params.)
    """
    model.eval()  # Dropout ve BatchNorm'u değerlendirme moduna al
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        predictions = model(X_batch)
        loss        = criterion(predictions, y_batch)
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# ANA EĞİTİM FONKSİYONU / Main Training Function
# ─────────────────────────────────────────────────────────────

def train_model(
    model        : nn.Module,
    train_loader : DataLoader,
    val_loader   : DataLoader,
    device       : torch.device
) -> dict:
    """
    Erken durdurma ve öğrenme hızı planlaması ile
    tam eğitim döngüsünü yürütür.

    Runs the complete training loop with early stopping
    and learning rate scheduling.

    Returns:
        history: {
            "train_loss": [...],  ← her epoch'un train kaybı
            "val_loss"  : [...],  ← her epoch'un val kaybı
            "lr"        : [...],  ← her epoch'un öğrenme hızı
        }
    """

    # ── Kayıp Fonksiyonu / Loss Function ─────────────────────
    # MSELoss: Ortalama Karesel Hata — sürekli değer tahmini için standart seçim
    # MSELoss: Mean Squared Error — standard choice for continuous value prediction
    criterion = nn.MSELoss()

    # ── Optimizer ────────────────────────────────────────────
    # Adam: Adaptif moment tahmini, genellikle iyi bir başlangıç noktası
    # Adam: Adaptive Moment Estimation, typically a good default optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY   # L2 düzenlileştirme
    )

    # ── Öğrenme Hızı Planlayıcı / LR Scheduler ───────────────
    # ReduceLROnPlateau: Val loss iyileşmediğinde LR'yi düşürür.
    # Modelin "plateau"lara takılmasını önler.
    # Reduces LR when val loss stops improving → helps escape plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",    # Loss azaldıkça iyi
        factor   = 0.5,      # Her tetiklemede LR'yi yarıya indir
        patience = 5,        # 5 epoch iyileşmezse tetikle
        min_lr   = 1e-6,     # Minimum öğrenme hızı sınırı
    )

    # ── Erken Durdurma / Early Stopping ──────────────────────
    early_stopper = EarlyStopping(
        patience  = config.PATIENCE,
        min_delta = config.MIN_DELTA
    )

    # Eğitim geçmişini kaydet / Store training history
    history = {"train_loss": [], "val_loss": [], "lr": []}

    # ── Başlık / Header ───────────────────────────────────────
    logger.info("=" * 65)
    logger.info(f"  Eğitim başlıyor / Training started | "
                f"Model: {config.MODEL_TYPE} | "
                f"Device: {device}")
    logger.info(f"  Epochs: {config.EPOCHS} | "
                f"Batch: {config.BATCH_SIZE} | "
                f"LR: {config.LEARNING_RATE}")
    logger.info("=" * 65)
    logger.info(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | "
                f"{'LR':>10} | {'Time':>7}")
    logger.info("-" * 65)

    best_val_loss  = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # ── Eğitim adımı / Training step ─────────────────────
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # ── Doğrulama adımı / Validation step ────────────────
        val_loss = evaluate_loader(
            model, val_loader, criterion, device
        )

        # ── Scheduler güncelle / Update scheduler ─────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Geçmişe kaydet / Log to history ───────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # ── En iyi modeli kaydet / Save best model ─────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config.SAVE_BEST_MODEL:
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    config.CHECKPOINT_PATH
                )

        # ── Loglama / Logging ─────────────────────────────────
        logger.info(
            f"{epoch:>6} | {train_loss:>11.6f} | {val_loss:>10.6f} | "
            f"{current_lr:>10.2e} | {epoch_time:>6.1f}s"
        )

        # ── Erken durdurma kontrolü / Early stopping check ────
        if early_stopper(val_loss, epoch):
            logger.info(f"\n  Eğitim erken durduruldu / Training stopped early "
                        f"at epoch {epoch}.\n")
            break

    logger.info("=" * 65)
    logger.info(f"  Eğitim tamamlandı / Training complete! "
                f"En iyi Val Loss / Best Val Loss: {best_val_loss:.6f}")
    logger.info("=" * 65)

    return history