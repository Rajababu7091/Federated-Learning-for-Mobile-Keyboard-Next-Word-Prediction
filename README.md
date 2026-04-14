# 🔐 Federated Learning for Mobile Keyboard Next-Word Prediction

> **A privacy-first, on-device next-word prediction system using Federated Learning with Attention-GRU, FedProx + FedNova, DP-SGD, Rényi DP accounting, and Trigram re-ranking.**

---

## 🚀 What Makes This Unique

| Feature | Standard FL | This Project |
|---|---|---|
| Model | Plain GRU | **Attention-GRU** (Bahdanau self-attention) |
| Aggregation | FedAvg | **FedProx + FedNova** (proximal term + normalized gradient averaging) |
| Privacy mechanism | None | **DP-SGD** (gradient clipping + Gaussian noise, not just post-hoc noise) |
| Privacy accounting | None | **Rényi DP (RDP) accountant** (same as Google DP-SGD / TF Privacy) |
| LR schedule | Fixed | **Cosine annealing** (smooth convergence across rounds) |
| Loss function | Plain CE | **Label-smoothed CE** (prevents overconfident softmax) |
| Prediction | Softmax only | **Trigram + Bigram re-ranking** (two-level corpus statistics blend) |
| Data pipeline | NumPy loops | **tf.data with prefetch** (overlaps CPU prep with GPU compute) |
| TFLite export | Dynamic quant | **INT8 full-integer quantization** with representative dataset calibration |
| Personalization | None | **On-device head fine-tuning** (MAML-inspired, early stopping) |
| Privacy audit | Simplified ε | **RDP accountant + gradient leakage cosine similarity** |
| OOV handling | UNK only | **Longest-prefix subword fallback** |

---

## 🏗️ Architecture

```
Mobile Device (Client)
┌─────────────────────────────────────────────────────┐
│  Raw text  →  Tokenizer  →  Sequence builder        │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Attention-GRU Model                        │   │
│  │  Embedding(96) → GRU(192, seq) →            │   │
│  │  BahdanauAttention(64) → Dropout →          │   │
│  │  Dense(vocab_size, softmax)                 │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  DP-SGD local training:                            │
│    • FedProx proximal term (μ=0.01)                │
│    • Per-layer gradient clipping (L2 norm=1.0)     │
│    • Gaussian noise on weight delta (σ_mult=0.1)   │
│    • Cosine LR decay across rounds                 │
│    • Label-smoothed cross-entropy (ε=0.1)          │
│                                                     │
│  ← Only clipped+noisy weight delta transmitted →   │
└─────────────────────────────────────────────────────┘
                        ↕ weight deltas only
┌─────────────────────────────────────────────────────┐
│  Federated Server                                   │
│  FedNova normalized aggregation → Global update     │
│  RDP accountant tracks cumulative ε budget          │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Major Project/
├── vocabulary.py          # Vocab + bigram + trigram index + OOV prefix fallback
├── client_data.py         # 5 simulated on-device text corpora
├── federated_train.py     # Core FL: FedProx+FedNova, DP-SGD, cosine LR, label smoothing
├── demo.py                # Interactive demo + typing simulator
├── personalization.py     # Per-client on-device head fine-tuning
├── privacy_audit.py       # RDP accountant + membership inference + gradient leakage
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the federated model
```bash
python federated_train.py
```

### 3. Run the interactive demo
```bash
python demo.py
```

### 4. Run personalization
```bash
python personalization.py
python personalization.py --retrain   # force re-fine-tune
```

### 5. Run privacy audit
```bash
python privacy_audit.py
```

---

## 🔬 Technical Details

### DP-SGD  (Differentially Private SGD)
The original code added Gaussian noise directly to the final weight vector — this is **not** proper DP-SGD. The correct approach (Abadi et al., 2016) is:

1. Clip each gradient update to bounded L2 norm (sensitivity bounding)
2. Add calibrated Gaussian noise to the clipped delta
3. Track cumulative privacy cost with an accountant

```
Δw_clipped = Δw * min(1, C / ‖Δw‖₂)        # clip
Δw_private = Δw_clipped + N(0, (σ·C)²I)     # noise
```

### Rényi Differential Privacy (RDP) Accountant
The simplified Gaussian mechanism formula used previously overestimates ε significantly. The RDP accountant (Mironov 2017) composes privacy cost across all gradient steps and converts to (ε, δ)-DP via:

```
ε = RDP(α) * T + log(1/δ) / (α - 1)
```

where T = total gradient steps, α is the optimal Rényi order (searched over {2,4,8,16,32,64}). This is the same method used in TensorFlow Privacy and Google's production DP systems.

### FedNova  (Federated Normalized Averaging)
Standard FedAvg suffers from **objective inconsistency** when clients run different numbers of local steps (heterogeneous data sizes). FedNova (Li et al., 2021) normalizes each client's update by its local step count before aggregating:

```
Δw_i_normalized = (w_i − w_global) / τ_i
w_new = w_global + Σ_i (n_i/N) * Δw_i_normalized
```

This produces a better-calibrated global gradient direction than plain FedAvg.

### Cosine LR Decay
Learning rate decays smoothly from `LR` to `LR_MIN` across rounds:
```
lr(t) = LR_MIN + (LR - LR_MIN) * 0.5 * (1 + cos(π * t / T))
```
Prevents oscillation in later rounds when the model is near convergence.

### Label Smoothing
Replaces hard one-hot targets with soft targets `(1-ε)·y + ε/K`. With ε=0.1, this prevents the model from becoming overconfident on training tokens — critical for a keyboard that must generalize to unseen word sequences.

### Trigram + Bigram Re-ranking
Final predictions blend three signals:
```
P_final = 0.80 * P_model + 0.10 * P_trigram + 0.10 * P_bigram
```
Trigrams capture stronger context (e.g. `('planning', 'a') → 'trip'`) than bigrams alone. Falls back gracefully to bigram when trigram context is unseen.

### tf.data Prefetch Pipeline
```python
tf.data.Dataset
  .from_tensor_slices((x, y))
  .shuffle(buffer_size=1024, reshuffle_each_iteration=True)
  .batch(BATCH_SIZE)
  .prefetch(tf.data.AUTOTUNE)
```
Overlaps CPU data preparation with GPU/CPU model compute, eliminating the data loading bottleneck during local training.

### INT8 TFLite Export
Uses a representative dataset from client corpora to calibrate full-integer quantization, reducing model size by ~4× vs float32 with minimal accuracy loss.

---

## 📊 Privacy Guarantees

| Guarantee | Mechanism |
|---|---|
| Raw text stays on device | Local training only |
| Gradient sensitivity bounded | L2 clipping (norm=1.0) |
| Weight updates are noisy | Gaussian DP-SGD (σ_mult=0.1) |
| Privacy cost tracked | RDP accountant (Rényi DP) |
| Client drift bounded | FedProx (μ=0.01) |
| Update dominance prevented | FedNova normalization |
| No user identifiers | Anonymous weight aggregation |

---

## 🧪 Client Personas

| Client | Domain | Vocabulary Style |
|---|---|---|
| 1 | Social / Casual | Conversational, informal |
| 2 | Professional / Work | Formal, business terminology |
| 3 | Tech / Developer | Technical, ML/software terms |
| 4 | Food / Lifestyle | Culinary, wellness vocabulary |
| 5 | Travel / General | Geographic, exploratory |

---

## 📱 Mobile Deployment

The trained model is exported as a full-integer INT8 quantized TFLite model (`keyboard_model.tflite`) suitable for:
- Android keyboard apps (via TFLite Android API)
- iOS keyboard extensions (via TFLite Swift/ObjC API)
- On-device inference with < 5ms latency

---

## 🔮 Future Extensions

- [ ] Secure Aggregation (cryptographic masking of weight updates)
- [ ] Federated Distillation (share predictions, not weights)
- [ ] Transformer-based model (replace GRU with lightweight BERT)
- [ ] Real Android keyboard integration
- [ ] Emoji and punctuation prediction
- [ ] Multi-language support
