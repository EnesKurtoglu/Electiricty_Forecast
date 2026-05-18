# src/utils.py
# ============================================================
# Yardımcı fonksiyonlar: seed ayarlama, cihaz seçimi,
# logging ve checkpoint kayıt/yükleme işlemleri.
# (Helper utilities: seeding, device selection, logging,
#  checkpoint save/load operations.)
# ============================================================

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sonuçların tekrar üretilebilir (reproducible) olması için
    tüm rastgele sayı üreticilerini sabitler.
    Fixes all random number generators for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CuDNN'i deterministik moda al (hafif yavaşlama olabilir)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    CUDA (GPU) varsa GPU, yoksa CPU döndürür.
    Returns GPU if CUDA is available, else CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"[Device] Kullanılan cihaz / Active device : {device}")
    if torch.cuda.is_available():
        print(f"[Device] GPU: {gpu_name}")
        print(f"[Device] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def get_logger(name: str = "ElectricityForecast") -> logging.Logger:
    """
    Konsola ve dosyaya aynı anda yazan bir logger oluşturur.
    Creates a logger that writes to both console and file.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Konsol handler ---
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # --- Dosya handler (outputs/reports/train.log) ---
    from configs.config import REPORT_DIR
    os.makedirs(REPORT_DIR, exist_ok=True)
    fh = logging.FileHandler(os.path.join(REPORT_DIR, "train.log"), mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_loss: float,
                    path: str) -> None:
    """
    Model ağırlıklarını, optimizer durumunu ve epoch bilgisini
    tek bir .pt dosyasına kaydeder.
    Saves model weights, optimizer state, and epoch info to a .pt file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"     : epoch,
        "val_loss"  : val_loss,
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None,
                    path: str,
                    device: torch.device) -> dict:
    """
    Kaydedilmiş checkpoint dosyasını yükler.
    Loads a saved checkpoint file.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print(f"[Checkpoint] Epoch {checkpoint['epoch']} yüklendi, "
          f"Val Loss: {checkpoint['val_loss']:.6f}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Modelin eğitilebilir parametre sayısını döndürür.
    Returns the number of trainable parameters in the model.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Toplam eğitilebilir parametre / Trainable params: {total:,}")
    return total
