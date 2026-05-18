# configs/config.py
# ============================================================
# Tüm hiperparametreleri ve proje ayarlarını merkezi olarak
# yöneten yapılandırma dosyası (Central configuration hub).
# Buradan değiştirdiğin her şey projenin tamamına yansır.
# ============================================================

import os

# --------------- Dizin Yapısı / Directory Paths ---------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
PLOT_DIR    = os.path.join(OUTPUT_DIR, "plots")
CKPT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")
REPORT_DIR  = os.path.join(OUTPUT_DIR, "reports")

# --------------- Veri / Data Settings -------------------------
# EPİAŞ'tan indirdiğin CSV dosyasının adı.
# Dosyayı 'data/' klasörüne koyup bu ismi güncelle.
DATA_FILE  = "Gercek_Zamanli_Tuketim-16052025-16052026.csv"
DATE_COL   = "Tarih"
TARGET_COL = "Tüketim Miktarı(MWh)"
# EPİAŞ CSV genellikle noktalı virgülle ayrılmış gelir:
CSV_SEP     = ";"

# --------------- Zaman Serisi Penceresi / Sliding Window ------
# LOOKBACK: Modele "geçmiş kaç saati göster?" (168 = 1 hafta)
# HORIZON : "Kaç saat ilerisi tahmin edilsin?" (24 = 1 gün)
LOOKBACK    = 168   # Geri bakış penceresi (saat cinsinden)
HORIZON     = 24    # Tahmin ufku (saat cinsinden)

# --------------- Veri Bölme / Train-Val-Test Split ------------
# Verinin %70'i eğitim, %15'i doğrulama, %15'i test için kullanılır.
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# Test oranı otomatik hesaplanır: 1 - TRAIN - VAL

# --------------- Model Seçimi / Model Selection ---------------
# "LSTM" veya "GRU" yazarak model tipini belirle.
MODEL_TYPE      = "GRU"     # "LSTM" | "GRU"
HIDDEN_SIZE     = 128           # Her RNN katmanındaki gizli birim sayısı
NUM_LAYERS      = 2             # Üst üste yığılan RNN katman sayısı
DROPOUT         = 0.3           # Dropout oranı (0 = kapalı, 0.5 = %50)
BIDIRECTIONAL   = False         # Çift yönlü RNN kullan mı?

# --------------- Eğitim / Training Settings -------------------
BATCH_SIZE      = 64            # Her adımda işlenen örnek sayısı
EPOCHS          = 100           # Maksimum epoch sayısı
LEARNING_RATE   = 1e-3          # Adam optimizer öğrenme hızı
WEIGHT_DECAY    = 1e-5          # L2 düzenlileştirme katsayısı
SCALER_TYPE     = "minmax"      # "minmax" | "standard"

# --------------- Erken Durdurma / Early Stopping --------------
# Validation loss 'PATIENCE' epoch boyunca iyileşmezse dur.
PATIENCE        = 15
MIN_DELTA       = 1e-5          # "İyileşme" için minimum fark eşiği

# --------------- Çıktı / Output Settings ----------------------
SAVE_BEST_MODEL = True          # En iyi modeli kaydet
MODEL_NAME      = f"{MODEL_TYPE}_h{HIDDEN_SIZE}_l{NUM_LAYERS}"
CHECKPOINT_PATH = os.path.join(CKPT_DIR, f"{MODEL_NAME}_best.pt")

# --------------- Seed (Tekrar Üretilebilirlik) ----------------
SEED = 42
