# src/models.py
# ============================================================
# LSTM ve GRU model mimarilerini tek bir sınıf altında
# parametrik olarak tanımlar.
#
# Defines LSTM and GRU architectures parametrically
# under a single unified class.
# ============================================================

import torch
import torch.nn as nn
from configs import config


# ─────────────────────────────────────────────────────────────
# RNN BLOK: Seçilebilir LSTM veya GRU Katmanı
# Selectable LSTM or GRU Layer Block
# ─────────────────────────────────────────────────────────────

class RNNBlock(nn.Module):
    """
    LSTM veya GRU mimarisini tek bir blokta kapsüller.
    Encapsulates LSTM or GRU architecture in a single block.

    Mimari / Architecture:
    ──────────────────────────────────────────────────────────
    Girdi (Input)     : [Batch, Lookback, input_size]
                         ↓
    LSTM veya GRU     : num_layers adet yığılı RNN katmanı
     (+ Bidirectional?): her katmanda dropout uygulanır
                         ↓
    Son Zaman Adımı   : [Batch, hidden_size (* 2 bi-dir)]
                         ↓
    LayerNorm         : İç kovaryanslı kayma sorununu azaltır
                         ↓
    Dropout           : Aşırı öğrenmeyi engeller
                         ↓
    FC Katmanları     : hidden → hidden//2 → horizon
    (Dense Layers)    : Her aralarında ReLU aktivasyon
                         ↓
    Çıktı (Output)    : [Batch, horizon]  ← tahmin değerleri
    ──────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        model_type  : str   = "LSTM",   # "LSTM" | "GRU"
        input_size  : int   = 1,         # Özellik sayısı (feature count)
        hidden_size : int   = 128,       # Gizli katman boyutu
        num_layers  : int   = 2,         # Yığılan RNN katman sayısı
        dropout     : float = 0.3,       # Dropout oranı
        bidirectional: bool = False,     # Çift yönlü RNN mü?
        horizon     : int   = 24,        # Tahmin adımı sayısı
    ):
        super(RNNBlock, self).__init__()

        self.model_type    = model_type.upper()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        # Çift yönlü ise çıktı boyutu 2 katına çıkar
        self.direction_factor = 2 if bidirectional else 1

        # ── RNN Katmanı / RNN Layer ──────────────────────────
        rnn_kwargs = dict(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,          # [Batch, Seq, Feature] formatı
            dropout       = dropout if num_layers > 1 else 0.0,
            # Not: PyTorch'ta dropout sadece num_layers>1 iken katmanlar
            # arasına eklenir; tek katmanda bu parametre görmezden gelinir.
            bidirectional = bidirectional,
        )

        if self.model_type == "LSTM":
            # LSTM: Uzun kısa süreli bellek hücreleri (cell state + hidden state)
            # Kapılar: input, forget, output, cell
            # Long short-term memory with cell state and 4 gates
            self.rnn = nn.LSTM(**rnn_kwargs)

        elif self.model_type == "GRU":
            # GRU: LSTM'den daha basit (reset gate + update gate)
            # LSTM'e göre daha az parametre, genellikle daha hızlı eğitim
            # Simpler than LSTM: reset gate + update gate, fewer params
            self.rnn = nn.GRU(**rnn_kwargs)

        else:
            raise ValueError(
                f"Geçersiz model tipi / Invalid model type: '{model_type}'. "
                "Kullanın / Use: 'LSTM' veya/or 'GRU'"
            )

        # RNN çıktı boyutu (bidirectional ise iki kat)
        rnn_out_size = hidden_size * self.direction_factor

        # ── Normalizasyon / Normalization ────────────────────
        # LayerNorm: Her örnek için kendi içinde normalize eder.
        # BatchNorm'dan farklı olarak batch büyüklüğünden bağımsız.
        self.layer_norm = nn.LayerNorm(rnn_out_size)

        # ── Dropout ──────────────────────────────────────────
        # Eğitim sırasında rastgele nöronları kapatır → overfitting azalır
        # Randomly drops neurons during training → reduces overfitting
        self.dropout = nn.Dropout(p=dropout)

        # ── Tam Bağlı Katmanlar / Fully Connected Layers ─────
        # hidden → hidden//2 → horizon  (iki kademeli daralma)
        fc_mid = rnn_out_size // 2

        self.fc1 = nn.Linear(rnn_out_size, fc_mid)
        self.fc2 = nn.Linear(fc_mid, horizon)
        self.relu = nn.ReLU()

        # ── Ağırlık Başlangıç Değerleri / Weight Init ────────
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier Uniform ile Linear katmanları başlatır.
        Kaybın başlangıçta çok büyük veya küçük olmasını önler.
        Initializes Linear layers with Xavier Uniform.
        Prevents vanishing/exploding gradients at the start.
        """
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)  
            elif "bias" in name:
                nn.init.zeros_(param.data)
                if "bias_ih" in name and self.model_type == "LSTM":
                    n = param.data.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri geçiş / Forward pass.

        Args:
            x: [Batch, Lookback, Features]  ← DataLoader'dan gelen veri

        Returns:
            out: [Batch, Horizon]  ← tahmin edilen gelecek değerler
        """
        # ── RNN İleri Geçiş ──────────────────────────────────
        if self.model_type == "LSTM":
            # LSTM iki durum döndürür: (output, (hidden, cell))
            rnn_out, (h_n, _) = self.rnn(x)
        else:
            # GRU tek durum döndürür: (output, hidden)
            rnn_out, h_n = self.rnn(x)

        # h_n boyutu: [num_layers * directions, Batch, hidden]
        # Sadece son katmanın çıktısını kullanıyoruz.
        # Only use last layer's output.
        if self.bidirectional:
            # İleri ve geri yöndeki son katman gizli durumlarını birleştir
            # Concatenate forward and backward hidden states of last layer
            h_forward  = h_n[-2]   # son katman ileri yön
            h_backward = h_n[-1]   # son katman geri yön
            last_hidden = torch.cat([h_forward, h_backward], dim=-1)
        else:
            last_hidden = h_n[-1]  # [Batch, hidden_size]

        # ── Normalizasyon + Dropout ───────────────────────────
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)

        # ── Tam Bağlı Katmanlar ───────────────────────────────
        out = self.relu(self.fc1(out))
        out = self.fc2(out)          # [Batch, Horizon]

        return out

    def __repr__(self) -> str:
        direction = "Bidirectional-" if self.bidirectional else ""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{direction}{self.model_type} Model | "
            f"Layers: {self.num_layers} | "
            f"Hidden: {self.hidden_size} | "
            f"Params: {total_params:,}"
        )



# MODEL FACTORY / Fabrika Fonksiyonu


def build_model(device: torch.device) -> RNNBlock:
    """
    Config dosyasındaki ayarlara göre model oluşturur ve
    belirtilen cihaza (GPU/CPU) taşır.

    Builds model based on config settings and moves it
    to the specified device (GPU/CPU).
    """
    model = RNNBlock(
        model_type    = config.MODEL_TYPE,
        input_size    = 1,                  # Tek değişken: tüketim
        hidden_size   = config.HIDDEN_SIZE,
        num_layers    = config.NUM_LAYERS,
        dropout       = config.DROPOUT,
        bidirectional = config.BIDIRECTIONAL,
        horizon       = config.HORIZON,
    ).to(device)

    print(f"\n[Model] {model}")
    print(f"[Model] Cihaz / Device: {next(model.parameters()).device}\n")

    return model
