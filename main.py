#!/usr/bin/env python3
# main.py
# ============================================================
# Projenin giriş noktası (entry point).
# Tüm adımları sırayla çalıştırır:
#   1. Veri yükleme ve ön işleme
#   2. Model oluşturma
#   3. Eğitim
#   4. Değerlendirme ve görselleştirme
#
# Project entry point. Runs all steps sequentially:
#   1. Data loading and preprocessing
#   2. Model building
#   3. Training
#   4. Evaluation and visualization
#
# Kullanım / Usage:
#   python main.py
# ============================================================

import os
import sys
import json

# Proje kökünü Python yoluna ekle / Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs import config
from src.utils     import set_seed, get_device, get_logger, count_parameters, load_checkpoint
from src.dataset   import build_dataloaders
from src.models    import build_model
from src.train     import train_model
from src.evaluate  import full_evaluation


def main():
    # ── 0. Başlangıç Ayarları / Initialization ────────────────
    logger = get_logger()
    logger.info("\n" + "═" * 65)
    logger.info("   Elektrik Tüketimi Zaman Serisi Tahmini")
    logger.info("     Electricity Consumption Time-Series Forecasting")
    logger.info("═" * 65)

    # Tekrar üretilebilirlik için seed sabitle
    set_seed(config.SEED)
    device = get_device()

    # ── 1. Veri Pipeline / Data Pipeline ──────────────────────
    logger.info("\n[Step 1/4] Veri yükleniyor / Loading data...")
    data_path = os.path.join(config.DATA_DIR, config.DATA_FILE)

    if not os.path.exists(data_path):
        logger.error(
            f"\n{'─'*55}\n"
            f"  HATA / ERROR: CSV dosyası bulunamadı!\n"
            f"  Dosya: {data_path}\n\n"
            f"  Ne yapmalısın? / What to do?\n"
            f"  1. seffaflik.epias.com.tr adresine git\n"
            f"  2. Tüketim → Gerçek Zamanlı Tüketim bölümünü aç\n"
            f"  3. İstediğin tarih aralığını seç ve CSV olarak indir\n"
            f"  4. İndirilen dosyayı '{config.DATA_DIR}/' klasörüne koy\n"
            f"  5. configs/config.py içinde DATA_FILE adını güncelle\n"
            f"{'─'*55}\n"
        )
        sys.exit(1)

    train_loader, val_loader, test_loader, scaler, (X_test, y_test) = \
        build_dataloaders(data_path)

    # ── 2. Model Oluşturma / Model Building ───────────────────
    logger.info("\n[Step 2/4] Model oluşturuluyor / Building model...")
    model = build_model(device)
    count_parameters(model)

    # ── 3. Eğitim / Training ───────────────────────────────────
    logger.info("\n[Step 3/4] Eğitim başlıyor / Starting training...")
    history = train_model(model, train_loader, val_loader, device)

    # ── 4. Değerlendirme / Evaluation ─────────────────────────
    logger.info("\n[Step 4/4] Değerlendirme / Evaluation...")

    # En iyi modeli yükle (checkpoint'ten / from checkpoint)
    if config.SAVE_BEST_MODEL and os.path.exists(config.CHECKPOINT_PATH):
        logger.info(f"  En iyi checkpoint yükleniyor / Loading best checkpoint...")
        load_checkpoint(model, None, config.CHECKPOINT_PATH, device)

    metrics = full_evaluation(model, test_loader, history, scaler, device)

    # ── Metrikleri JSON olarak kaydet / Save metrics as JSON ──
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    metrics_path = os.path.join(config.REPORT_DIR, f"{config.MODEL_NAME}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2, ensure_ascii=False)
    logger.info(f"  Metrikler kaydedildi / Metrics saved: {metrics_path}")

    # ── Özet / Summary ─────────────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("   Pipeline tamamlandı / Pipeline complete!")
    logger.info(f"  Model     : {config.MODEL_TYPE} (layers={config.NUM_LAYERS}, "
                f"hidden={config.HIDDEN_SIZE})")
    logger.info(f"  Lookback  : {config.LOOKBACK} saat / hours")
    logger.info(f"  Horizon   : {config.HORIZON} saat / hours")
    logger.info(f"  Checkpoint: {config.CHECKPOINT_PATH}")
    logger.info(f"  Grafikler / Plots : {config.PLOT_DIR}/")
    logger.info(f"  Loglar / Logs     : {config.REPORT_DIR}/train.log")
    logger.info("═" * 65 + "\n")


if __name__ == "__main__":
    main()
