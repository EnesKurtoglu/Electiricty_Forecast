# src/evaluate.py
# ============================================================
# Model değerlendirme: RMSE, MAE hesaplama ve görselleştirme.
# Model evaluation: RMSE, MAE calculation and visualization.
# ============================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # GUI olmayan sunucularda çalışması için / for headless servers (kaydedilir direk sonuçlar)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import config
from src.utils import get_logger

logger = get_logger()

# Grafik stili ayarla
plt.rcParams.update({
    "figure.facecolor" : "#0f1117",
    "axes.facecolor"   : "#1a1d27",
    "axes.edgecolor"   : "#3a3d4d",
    "axes.labelcolor"  : "#c8ccd4",
    "text.color"       : "#c8ccd4",
    "xtick.color"      : "#8b8fa8",
    "ytick.color"      : "#8b8fa8",
    "grid.color"       : "#2a2d3d",
    "grid.alpha"       : 0.5,
    "font.family"      : "monospace",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

ACCENT_REAL  = "#4fc3f7"   # Gerçek değerler için renk (mavi)
ACCENT_PRED  = "#f06292"   # Tahminler için renk (pembe/kırmızı)
ACCENT_TRAIN = "#66bb6a"   # Eğitim kaybı için (yeşil)
ACCENT_VAL   = "#ffa726"   # Doğrulama kaybı için (turuncu)



# TAHMİN ALMA / Get Predictions


@torch.no_grad()
def get_predictions(
    model  : nn.Module,
    loader : DataLoader,
    device : torch.device,
    scaler
) -> tuple:
    """
    Test DataLoader üzerinde tahminler üretir ve
    scaler'ı tersine uygulayarak orijinal birime döndürür.

    Generates predictions on test DataLoader and
    inverse-transforms them back to original scale (MWh).

    Returns:
        y_true_inv : Gerçek değerler (MWh cinsinden)
        y_pred_inv : Tahmin değerleri (MWh cinsinden)
    """
    model.eval()

    all_preds  = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        preds   = model(X_batch).cpu().numpy()   # [Batch, Horizon]
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())

    # Tüm batch'leri birleştir / Concatenate all batches
    y_pred = np.concatenate(all_preds,  axis=0)   # (N, Horizon)
    y_true = np.concatenate(all_labels, axis=0)   # (N, Horizon)

    # ── Ters Ölçekleme / Inverse Scaling 
    # Modelin tahminleri [0,1] veya standardize aralığında.
    # Metrikleri anlamlı (MWh) cinsinden hesaplamak için
    # orijinal birime çeviriyoruz.
    #
    # Predictions are in normalized scale [0,1] or standardized.
    # Inverse transform to get values in original MWh unit.

    def inverse_transform_2d(arr: np.ndarray) -> np.ndarray:
        """
        2D diziyi (N, Horizon) scaler ile ters dönüştürür.
        Scaler tek boyutlu beklediğinden aşağıda reshape yapıyoruz.
        """
        n_samples, horizon = arr.shape
        # (N*Horizon, 1) → scaler → (N, Horizon)
        flat = arr.reshape(-1, 1)
        flat_inv = scaler.inverse_transform(flat)
        return flat_inv.reshape(n_samples, horizon)

    y_pred_inv = inverse_transform_2d(y_pred)
    y_true_inv = inverse_transform_2d(y_true)

    return y_true_inv, y_pred_inv



# METRİKLER / Metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    RMSE ve MAE hesaplar ve loglar.
    Computes and logs RMSE and MAE.

    RMSE (Root Mean Squared Error / Kök Ortalama Kare Hata):
    → Büyük hataları daha ağır cezalandırır.
    → Penalizes large errors more heavily.

    MAE (Mean Absolute Error / Ortalama Mutlak Hata):
    → Hataların doğrudan ortalaması, yorumlaması kolay.
    → Direct average of errors, easy to interpret.

    MAPE (Mean Absolute Percentage Error / Ortalama Yüzde Hata):
    → Yüzde cinsinden hata, farklı ölçeklerle karşılaştırmayı kolaylaştırır.
    → Percentage error, useful for scale-independent comparison.
    """

    # İlk tahmin adımını değerlendir (1 saatlik tahmin performansı)
    y_true_flat = y_true[:, 0]
    y_pred_flat = y_pred[:, 0]

    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae  = mean_absolute_error(y_true_flat, y_pred_flat)

    # MAPE (sıfır bölmesine karşı güvenli)
    mask = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100

    # Tüm horizon adımları için de hesapla
    rmse_all = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae_all  = mean_absolute_error(y_true.flatten(), y_pred.flatten())

    metrics = {
        "RMSE (h+1)"  : rmse,
        "MAE  (h+1)"  : mae,
        "MAPE (h+1) %": mape,
        f"RMSE (all {config.HORIZON}h)": rmse_all,
        f"MAE  (all {config.HORIZON}h)": mae_all,
    }

    logger.info("\n" + "─" * 50)
    logger.info("  DEĞERLENDİRME METRİKLERİ / EVALUATION METRICS")
    logger.info("─" * 50)
    for name, val in metrics.items():
        unit = "%" if "%" in name else "MWh"
        logger.info(f"  {name:<30}: {val:>10.2f} {unit}")
    logger.info("─" * 50 + "\n")

    return metrics



# GÖRSELLEŞTİRME / Visualization


def plot_training_history(history: dict, save_path: str) -> None:
    """
    Eğitim ve doğrulama kayıp grafiğini çizer.
    Plots training and validation loss curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"Eğitim Geçmişi / Training History — {config.MODEL_TYPE}",
        fontsize=14, color="#e8eaf0", fontweight="bold", y=1.01
    )

    # ── Sol: Loss Grafiği 
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=ACCENT_TRAIN,
            linewidth=1.8, label="Train Loss", alpha=0.9)
    ax.plot(epochs, history["val_loss"], color=ACCENT_VAL,
            linewidth=1.8, label="Val Loss", alpha=0.9)
    ax.set_title("MSE Loss", color="#e8eaf0", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.2, edgecolor="#3a3d4d")
    ax.grid(True, alpha=0.3)

    # En düşük val loss noktasını işaretle
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_loss  = min(history["val_loss"])
    ax.axvline(x=best_epoch, color="#ffffff", linestyle="--",
               alpha=0.4, linewidth=1)
    ax.annotate(
        f"Best: {best_loss:.5f}\n@ epoch {best_epoch}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + max(1, len(epochs) * 0.05), best_loss * 1.1),
        color="#ffd54f", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#ffd54f", lw=0.8)
    )

    # ── Sağ: Öğrenme Hızı Grafiği 
    ax2 = axes[1]
    ax2.semilogy(epochs, history["lr"], color="#ce93d8",
                 linewidth=1.8, alpha=0.9)
    ax2.set_title("Öğrenme Hızı / Learning Rate", color="#e8eaf0", pad=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LR (log scale)")
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  [Plot] Eğitim grafiği kaydedildi / Saved: {save_path}")


def plot_predictions(
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    save_path  : str,
    n_samples  : int = 500
) -> None:
    """
    Gerçek ve tahmin edilen değerleri karşılaştırmalı çizer.
    Plots actual vs predicted values for visual comparison.

    Args:
        y_true    : (N, Horizon) gerçek değerler
        y_pred    : (N, Horizon) tahmin değerleri
        n_samples : Gösterilecek örnek sayısı (okunabilirlik için)
    """
    # İlk tahmin adımını al ve zaman dizisi oluştur
    true_1h = y_true[:n_samples, 0]
    pred_1h = y_pred[:n_samples, 0]
    x_axis  = np.arange(len(true_1h))

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"Gerçek vs Tahmin / Actual vs Predicted — {config.MODEL_TYPE}\n"
        f"(İlk {n_samples} test örneği, 1 saatlik tahmin ufku)",
        fontsize=13, color="#e8eaf0", fontweight="bold"
    )

    # ── Üst: Tam Karşılaştırma 
    ax1 = axes[0]
    ax1.plot(x_axis, true_1h, color=ACCENT_REAL,
             linewidth=1.2, label="Gerçek / Actual", alpha=0.9)
    ax1.plot(x_axis, pred_1h, color=ACCENT_PRED,
             linewidth=1.2, label="Tahmin / Predicted", alpha=0.8,
             linestyle="--")
    ax1.set_title("Genel Karşılaştırma / Overall Comparison",
                  color="#e8eaf0", pad=8)
    ax1.set_xlabel("Test Örnek İndisi / Test Sample Index")
    ax1.set_ylabel("Tüketim (MWh)")
    ax1.legend(framealpha=0.2, edgecolor="#3a3d4d")
    ax1.grid(True, alpha=0.3)

    # ── Orta: Yakın Görünüm (Son 7 Gün / 168 Saat) 
    ax2 = axes[1]
    zoom = min(168, len(true_1h))
    ax2.plot(x_axis[-zoom:], true_1h[-zoom:], color=ACCENT_REAL,
             linewidth=1.5, label="Gerçek / Actual", marker="o",
             markersize=2, alpha=0.9)
    ax2.plot(x_axis[-zoom:], pred_1h[-zoom:], color=ACCENT_PRED,
             linewidth=1.5, label="Tahmin / Predicted", marker="x",
             markersize=3, alpha=0.8, linestyle="--")
    ax2.fill_between(x_axis[-zoom:], true_1h[-zoom:], pred_1h[-zoom:],
                     alpha=0.15, color="#f06292", label="Hata / Error")
    ax2.set_title(f"Son 7 Gün Yakın Görünüm / Last 7-Day Zoom ({zoom} saat)",
                  color="#e8eaf0", pad=8)
    ax2.set_xlabel("Test Örnek İndisi / Test Sample Index")
    ax2.set_ylabel("Tüketim (MWh)")
    ax2.legend(framealpha=0.2, edgecolor="#3a3d4d")
    ax2.grid(True, alpha=0.3)

    # ── Alt: Artık Hata Dağılımı / Residual Plot 
    ax3 = axes[2]
    residuals = true_1h - pred_1h
    ax3.bar(x_axis, residuals, color=np.where(residuals >= 0, ACCENT_REAL, ACCENT_PRED),
            alpha=0.6, width=1.0)
    ax3.axhline(y=0, color="#ffffff", linewidth=1, alpha=0.5)
    ax3.axhline(y=np.std(residuals), color="#ffd54f", linewidth=1,
                alpha=0.5, linestyle="--", label=f"+1σ = {np.std(residuals):.0f}")
    ax3.axhline(y=-np.std(residuals), color="#ffd54f", linewidth=1,
                alpha=0.5, linestyle="--", label=f"-1σ = {-np.std(residuals):.0f}")
    ax3.set_title("Artık Hatalar / Residuals (Gerçek - Tahmin)",
                  color="#e8eaf0", pad=8)
    ax3.set_xlabel("Test Örnek İndisi / Test Sample Index")
    ax3.set_ylabel("Hata (MWh)")
    ax3.legend(framealpha=0.2, edgecolor="#3a3d4d", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  [Plot] Tahmin grafiği kaydedildi / Saved: {save_path}")


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """
    Gerçek vs Tahmin scatter (dağılım) grafiği.
    Mükemmel tahmin → noktalar diyagonalde toplanır.
    Actual vs Predicted scatter plot.
    Perfect model → points cluster around diagonal.
    """
    true_flat = y_true[:, 0]
    pred_flat = y_pred[:, 0]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0f1117")

    ax.scatter(true_flat, pred_flat, alpha=0.3, s=8,
               color=ACCENT_PRED, edgecolors="none")

    # Mükemmel tahmin çizgisi (y=x)
    min_val = min(true_flat.min(), pred_flat.min())
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color="#ffffff", linewidth=1.5, alpha=0.5,
            linestyle="--", label="Mükemmel Tahmin / Perfect Fit (y=x)")

    ax.set_title(f"Gerçek vs Tahmin Dağılımı / Actual vs Predicted Scatter\n{config.MODEL_TYPE}",
                 color="#e8eaf0", pad=12)
    ax.set_xlabel("Gerçek Tüketim / Actual Consumption (MWh)")
    ax.set_ylabel("Tahmin Tüketim / Predicted Consumption (MWh)")
    ax.legend(framealpha=0.2, edgecolor="#3a3d4d")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  [Plot] Scatter grafiği kaydedildi / Saved: {save_path}")



# ANA DEĞERLENDİRME / Main Evaluation Pipeline


def full_evaluation(
    model       : nn.Module,
    test_loader : DataLoader,
    history     : dict,
    scaler,
    device      : torch.device
) -> dict:
    """
    Tahmin al → metrik hesapla → grafik çiz.
    Get predictions → compute metrics → plot charts.
    """
    os.makedirs(config.PLOT_DIR, exist_ok=True)

    logger.info("\n[Evaluate] Tahminler üretiliyor / Generating predictions...")
    y_true, y_pred = get_predictions(model, test_loader, device, scaler)

    logger.info("[Evaluate] Metrikler hesaplanıyor / Computing metrics...")
    metrics = compute_metrics(y_true, y_pred)

    logger.info("[Evaluate] Grafikler çiziliyor / Plotting charts...")
    plot_training_history(
        history,
        os.path.join(config.PLOT_DIR, f"{config.MODEL_NAME}_training_history.png")
    )
    plot_predictions(
        y_true, y_pred,
        os.path.join(config.PLOT_DIR, f"{config.MODEL_NAME}_predictions.png")
    )
    plot_scatter(
        y_true, y_pred,
        os.path.join(config.PLOT_DIR, f"{config.MODEL_NAME}_scatter.png")
    )

    logger.info(f"\n[Evaluate] Tüm grafikler '{config.PLOT_DIR}' dizinine kaydedildi.\n")

    return metrics
