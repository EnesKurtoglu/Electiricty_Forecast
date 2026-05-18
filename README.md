# Electricity Consumption Forecasting
### LSTM / GRU ile Elektrik Tüketimi Zaman Serisi Tahmini

EPİAŞ Şeffaflık Platformu'ndan alınan gerçek Türkiye elektrik tüketimi verileriyle  
LSTM veya GRU modellerini eğitmek için production-ready bir PyTorch pipeline'ı.

---

##  Proje Yapısı / Project Structure

```
electricity_forecast/
├── data/                          ← EPİAŞ CSV dosyası buraya gelmeli
├── src/
│   ├── dataset.py                 ← Veri yükleme, ölçekleme, sliding window, DataLoader
│   ├── models.py                  ← LSTM / GRU model mimarisi
│   ├── train.py                   ← Eğitim döngüsü, early stopping, scheduler
│   ├── evaluate.py                ← RMSE, MAE hesaplama + Matplotlib grafikleri
│   └── utils.py                   ← Seed, device, logger, checkpoint yardımcıları
├── configs/
│   └── config.py                  ← TÜM hiperparametreler merkezi olarak burada
├── outputs/
│   ├── plots/                     ← Eğitim + tahmin grafikleri (.png)
│   ├── checkpoints/               ← En iyi model ağırlıkları (.pt)
│   └── reports/                   ← Metrikler (.json) + log dosyası
├── main.py                        ← Tek komutla her şeyi çalıştır
└── requirements.txt
```

Girdi: [64, 168, 1]
         ↓
   ┌─────────────┐
   │  LSTM       │  Layer 1: 128 nöron
   │  + Dropout  │  (katmanlar arası %30 nöron kapatılır)
   ├─────────────┤
   │  LSTM       │  Layer 2: 128 nöron  
   └─────────────┘
         ↓
   Son hidden state: [64, 128]
         ↓
   LayerNorm + Dropout(0.3)
         ↓
   Linear(128 → 64) + ReLU
         ↓
   Linear(64 → 24)
         ↓
   Çıktı: [64, 24]  ← 24 saatlik tahmin

---

##  Kurulum / Setup

```bash
# 1. Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 2. Bağımlılıkları yükle
pip install -r requirements.txt
```

---

##  Veri İndirme / Getting Data (EPİAŞ)

1. **https://seffaflik.epias.com.tr** adresine git
2. **Tüketim → Gerçek Zamanlı Tüketim** sekmesini aç
3. Tarih aralığını seç (öneri: en az 1-2 yıl → model örüntüleri daha iyi öğrenir)
4. **CSV olarak indir** butonuna tıkla
5. İndirilen dosyayı `data/` klasörüne koy
6. `configs/config.py` içinde `DATA_FILE` değişkenini güncelle:
   ```python
   DATA_FILE  = "epias_tuketim.csv"   # Dosya adın ne ise onu yaz
   DATE_COL   = "Tarih"               # CSV'deki tarih sütunu adı
   TARGET_COL = "Tüketim Miktarı (MWh)"  # CSV'deki hedef sütun adı
   ```

---

##  Konfigürasyon / Configuration (`configs/config.py`)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `MODEL_TYPE` | `"LSTM"` | `"LSTM"` veya `"GRU"` |
| `LOOKBACK` | `168` | Geriye bakış penceresi (saat) — 168 = 1 hafta |
| `HORIZON` | `24` | Tahmin ufku (saat) — 24 = 1 gün |
| `HIDDEN_SIZE` | `128` | RNN gizli katman boyutu |
| `NUM_LAYERS` | `2` | Yığılı RNN katman sayısı |
| `DROPOUT` | `0.3` | Dropout oranı |
| `BATCH_SIZE` | `64` | Mini-batch boyutu |
| `EPOCHS` | `100` | Maksimum epoch sayısı |
| `LEARNING_RATE` | `1e-3` | Optimizer öğrenme hızı |
| `PATIENCE` | `15` | Early stopping sabrı |
| `SCALER_TYPE` | `"minmax"` | `"minmax"` veya `"standard"` |
| `BIDIRECTIONAL` | `False` | Çift yönlü RNN |

---

##  Çalıştırma / Running

```bash
# Proje kök dizininden:
python main.py
```

**Çıktılar / Outputs:**
- `outputs/plots/LSTM_h128_l2_training_history.png` — Loss eğrileri
- `outputs/plots/LSTM_h128_l2_predictions.png`      — Gerçek vs Tahmin
- `outputs/plots/LSTM_h128_l2_scatter.png`          — Scatter dağılımı
- `outputs/checkpoints/LSTM_h128_l2_best.pt`        — En iyi model ağırlıkları
- `outputs/reports/LSTM_h128_l2_metrics.json`       — RMSE, MAE, MAPE
- `outputs/reports/train.log`                       — Tam eğitim logu

---

##  Model Değiştirme / Switching Models

```python
# configs/config.py içinde:
MODEL_TYPE = "GRU"    # LSTM → GRU'ya geç
```

Başka bir şey değiştirmene gerek yok, her şey otomatik güncellenir.

---

##  Model Mimarisi / Architecture

```
Girdi / Input: [Batch, Lookback=168, Features=1]
        ↓
LSTM veya GRU (num_layers=2, hidden=128, dropout=0.3)
        ↓
Son Zaman Adımı Gizli Durumu / Last Hidden State
        ↓
LayerNorm → Dropout(0.3)
        ↓
Linear(128 → 64) → ReLU
        ↓
Linear(64 → 24)
        ↓
Çıktı / Output: [Batch, Horizon=24]  ← Sonraki 24 saatin tahmini
```

---

##  İpuçları / Tips

- **GPU kullanımı**: CUDA varsa otomatik algılanır ve kullanılır.
- **Lookback değeri**: Elektrik verisinde haftalık mevsimsellik güçlüdür. `LOOKBACK=168` (1 hafta) iyi bir başlangıçtır.
- **Horizon**: Kısa vadeli tahmin (1-24 saat) daha kolaydır. Horizon büyüdükçe hata artar.
- **Overfitting**: Val loss train loss'tan çok yüksekse `DROPOUT` değerini artır ya da `HIDDEN_SIZE`/`NUM_LAYERS` azalt.
- **Underfitting**: Her iki loss da yüksekse `HIDDEN_SIZE` veya `NUM_LAYERS` artır.
