# src/dataset.py
# ============================================================
# EPİAŞ CSV verisini okur, temizler, ölçekler ve
# PyTorch DataLoader formatına dönüştürür.
#
# Reads EPİAŞ CSV data, cleans it, scales it, and
# converts it into PyTorch DataLoader format.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Proje kök dizinini Python yoluna ekle (import için)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import config


# ─────────────────────────────────────────────────────────────
# 1. VERİ YÜKLEME VE TEMİZLEME / Data Loading & Cleaning
# ─────────────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    EPİAŞ CSV dosyasını okur ve temel temizlik işlemlerini yapar.

    EPİAŞ'tan indirilen CSV dosyaları genellikle:
    - Noktalı virgülle (;) ayrılmış olur
    - Sayılarda binlik ayırıcı olarak nokta, ondalık olarak virgül kullanır
      örn: "1.234,56" → 1234.56
    - İlk satır başlık (header) içerir
    - Tarih ve Saat sütunları ayrı olabilir

    Loads EPİAŞ CSV and performs basic cleaning steps.
    """
    print(f"[Dataset] Veri yükleniyor / Loading data: {filepath}")

    # --- CSV'yi oku / Read CSV ---
    # ÖNEMLİ: thousands='.' parametresi "16.05.2025" tarihini
    # 16052025 sayısına dönüştürür! Bunu önlemek için Tarih
    # sütununu dtype=str ile string olarak okutuyoruz.
    # IMPORTANT: thousands='.' would parse "16.05.2025" as
    # integer 16052025. We force the date column to str dtype.
    df = pd.read_csv(
        filepath,
        sep=config.CSV_SEP,
        encoding="utf-8-sig",
        thousands=".",
        decimal=",",
        dtype={config.DATE_COL: str},  # Tarihi string olarak oku
    )

    print(f"[Dataset] Ham veri boyutu / Raw shape: {df.shape}")
    print(f"[Dataset] Sütunlar / Columns: {list(df.columns)}")

    # --- Tarih sütunu oluştur / Build datetime index ---
    # EPİAŞ formatı: "16.05.2025" + "00:00" → "%d.%m.%Y %H:%M"
    # Explicit format veriyoruz çünkü pandas otomatik algılayamıyor.
    if "Saat" in df.columns and config.DATE_COL in df.columns:
        combined = df[config.DATE_COL].astype(str).str.strip() + " " + df["Saat"].astype(str).str.strip()
        # Önce bilinen EPİAŞ formatını dene, olmazsa genel parse'a düş
        df["datetime"] = pd.to_datetime(combined, format="%d.%m.%Y %H:%M", errors="coerce")
        if df["datetime"].isna().all():
            df["datetime"] = pd.to_datetime(combined, dayfirst=True, errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(
            df[config.DATE_COL].astype(str).str.strip(),
            format="%d.%m.%Y", errors="coerce"
        )
        if df["datetime"].isna().all():
            df["datetime"] = pd.to_datetime(df[config.DATE_COL], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()

    # --- Hedef sütunu temizle / Clean target column ---
    # Bazen string olarak gelebilir, sayıya çeviriyoruz
    if df[config.TARGET_COL].dtype == object:
        df[config.TARGET_COL] = (
            df[config.TARGET_COL]
            .astype(str)
            .str.replace(".", "", regex=False)   # Binlik ayırıcı
            .str.replace(",", ".", regex=False)  # Ondalık ayırıcı
            .astype(float)
        )

    # --- Eksik değerleri doldur / Fill missing values ---
    # Elektrik verilerinde saat atlamaları olabilir.
    # Forward fill + backward fill ile dolduruyoruz.
    missing_before = df[config.TARGET_COL].isna().sum()
    df[config.TARGET_COL] = (
        df[config.TARGET_COL]
        .interpolate(method="time")   # Zaman tabanlı interpolasyon
        .ffill()
        .bfill()
    )
    missing_after = df[config.TARGET_COL].isna().sum()
    print(f"[Dataset] Eksik değer / Missing: {missing_before} → {missing_after}")

    # --- Sadece hedef sütunu al ---
    df = df[[config.TARGET_COL]].copy()
    df.columns = ["consumption"]

    print(f"[Dataset] Temizlenmiş veri boyutu / Clean shape: {df.shape}")
    print(f"[Dataset] Tarih aralığı / Date range: "
          f"{df.index.min()} → {df.index.max()}")
    print(f"[Dataset] İstatistikler / Stats:\n{df.describe()}\n")

    return df


# ─────────────────────────────────────────────────────────────
# 2. ÖLÇEKLEME / Scaling (Normalization)
# ─────────────────────────────────────────────────────────────

def get_scaler():
    """
    Config'deki scaler tipine göre uygun scaler döndürür.
    - MinMaxScaler: Veriyi [0, 1] aralığına sıkıştırır.
      → Verinin alt ve üst sınırı biliniyorsa iyi çalışır.
    - StandardScaler: Ortalama=0, Std=1 yaparak normalize eder.
      → Aykırı değerlere (outlier) daha dayanıklı.

    Returns appropriate scaler based on config.
    """
    if config.SCALER_TYPE == "minmax":
        return MinMaxScaler(feature_range=(0, 1))
    elif config.SCALER_TYPE == "standard":
        return StandardScaler()
    else:
        raise ValueError(f"Bilinmeyen scaler tipi / Unknown scaler: {config.SCALER_TYPE}")


# ─────────────────────────────────────────────────────────────
# 3. VERİ BÖLME / Train-Val-Test Split
# ─────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple:
    """
    Veriyi zamana göre sıralı şekilde böler.
    ÖNEMLI: Zaman serilerinde rastgele bölme yapılmaz!
    Gelecekten geçmişe sızmayı (data leakage) önlemek için
    verinin ilk kısmı train, ortası val, sonu test olur.

    Splits data in chronological order (no random shuffling!).
    This prevents data leakage from future to past.
    """
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end   = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    print(f"[Dataset] Train: {len(train_df):,} | "
          f"Val: {len(val_df):,} | "
          f"Test: {len(test_df):,} (saatlik kayıt / hourly records)")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────
# 4. KAYAN PENCERE / Sliding Window → [Batch, Time, Features]
# ─────────────────────────────────────────────────────────────

def create_sequences(data: np.ndarray,
                     lookback: int,
                     horizon: int) -> tuple:
    """
    Ham diziyi (1D array) LSTM/GRU'nun beklediği
    [Örnekler, Zaman Adımları, Özellikler] formatına çevirir.

    Nasıl çalışır / How it works:
    ─────────────────────────────
    Veri: [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
    lookback=3, horizon=2 ise:

    Pencere 0: X=[x0,x1,x2]  →  y=[x3,x4]
    Pencere 1: X=[x1,x2,x3]  →  y=[x4,x5]
    Pencere 2: X=[x2,x3,x4]  →  y=[x5,x6]
    ...

    Args:
        data     : (N, 1) şeklinde ölçeklenmiş numpy dizisi
        lookback : Geriye bakış pencere uzunluğu (örn: 168 saat)
        horizon  : Tahmin ufku (örn: 24 saat)

    Returns:
        X: (num_samples, lookback, 1)  → Model girdisi
        y: (num_samples, horizon)      → Model çıktısı / hedef
    """
    X, y = [], []

    for i in range(len(data) - lookback - horizon + 1):
        # Giriş penceresi: lookback kadar geçmiş değer
        X.append(data[i : i + lookback])
        # Hedef: sonraki horizon kadar değer (düzleştirilmiş)
        y.append(data[i + lookback : i + lookback + horizon, 0])

    X = np.array(X, dtype=np.float32)  # (N, lookback, 1)
    y = np.array(y, dtype=np.float32)  # (N, horizon)

    print(f"[Dataset] Sequence X: {X.shape}  (örnekler, lookback, özellikler)")
    print(f"[Dataset] Sequence y: {y.shape}  (örnekler, horizon)")

    return X, y


# ─────────────────────────────────────────────────────────────
# 5. PYTORCH DATASET SINIFI / PyTorch Dataset Class
# ─────────────────────────────────────────────────────────────

class ElectricityDataset(Dataset):
    """
    PyTorch Dataset: DataLoader ile mini-batch oluşturmak için
    gerekli __len__ ve __getitem__ metodlarını tanımlar.

    PyTorch Dataset that defines __len__ and __getitem__
    for DataLoader to create mini-batches automatically.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # numpy → torch tensor dönüşümü
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Veri setindeki toplam örnek sayısı / Total number of samples"""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        """Belirtilen indisteki örneği döndür / Return sample at index"""
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# 6. ANA PIPELINE / Main Pipeline
# ─────────────────────────────────────────────────────────────

def build_dataloaders(filepath: str) -> tuple:
    """
    Ham CSV dosyasından eğitime hazır DataLoader'lara kadar
    tüm adımları tek fonksiyonda yürütür.

    Runs the complete pipeline from raw CSV to ready DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, scaler,
        (X_test_np, y_test_np)  ← Görselleştirme için orijinal numpy dizileri
    """

    # ── Adım 1: Yükle ve temizle / Load & clean ──────────────
    df = load_and_clean(filepath)

    # ── Adım 2: Kronolojik bölme / Chronological split ───────
    train_df, val_df, test_df = split_data(df)

    # ── Adım 3: Scaler'ı SADECE train üzerinde fit et ────────
    # Kritik: Scaler'ı val/test verisine de uygulamak için
    # onlara sadece transform() çağrısı yapılır (fit değil!).
    # Bu "veri sızıntısını" (data leakage) önler.
    scaler = get_scaler()
    train_scaled = scaler.fit_transform(train_df.values)  # fit + transform
    val_scaled   = scaler.transform(val_df.values)        # sadece transform
    test_scaled  = scaler.transform(test_df.values)       # sadece transform

    # ── Adım 4: Kayan pencere ile sequence oluştur ────────────
    X_train, y_train = create_sequences(train_scaled, config.LOOKBACK, config.HORIZON)
    X_val,   y_val   = create_sequences(val_scaled,   config.LOOKBACK, config.HORIZON)
    X_test,  y_test  = create_sequences(test_scaled,  config.LOOKBACK, config.HORIZON)

    # ── Adım 5: Dataset → DataLoader ─────────────────────────
    # shuffle=True sadece train için: Modelin örnek sıralarını
    # ezberlemesini (overfitting) önler.
    train_loader = DataLoader(
        ElectricityDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows uyumluluğu için 0
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        ElectricityDataset(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        ElectricityDataset(X_test, y_test),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\n[Dataset] Train batches: {len(train_loader)} | "
          f"Val: {len(val_loader)} | Test: {len(test_loader)}\n")

    return train_loader, val_loader, test_loader, scaler, (X_test, y_test)