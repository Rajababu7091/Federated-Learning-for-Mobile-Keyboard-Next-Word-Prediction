# 🔐 Federated Learning for Mobile Keyboard Next-Word Prediction

> **A privacy-first, on-device next-word prediction system using Federated Learning with Attention-GRU, FedProx + FedNova, DP-SGD, Rényi DP accounting, and Trigram re-ranking.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Live Demo

Start the server and open http://localhost:3000 to see the keyboard in action!

```bash
python app.py
```

## ✨ Features

- **🔒 Privacy-First Design**: Data never leaves the device in production
- **🧠 Attention-GRU Model**: 332,916 parameters with Bahdanau attention
- **🤝 Advanced Federated Learning**: FedProx + FedNova aggregation
- **🛡️ Differential Privacy**: DP-SGD with Rényi DP accounting
- **📊 Trigram Re-ranking**: Blends model predictions with corpus statistics
- **🎯 Personalization**: Per-client on-device fine-tuning
- **📱 Mobile-Ready**: INT8 TFLite export for deployment
- **🎨 Beautiful UI**: Real-time predictions with 4 themes

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

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/KOVVURIPCDURGAREDDY/Major-Project.git
cd Major-Project

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Open http://localhost:3000 in your browser.

## 🎯 Quick Start

### 1. Keyboard Tab
- Type in the input box to get real-time predictions
- Click predictions to add them to your sentence
- Switch between 5 personas (Social, Professional, Tech, Food, Travel)
- Use keyboard shortcuts: `Tab` (accept), `Enter` (refresh), `Esc` (clear)

### 2. Train Tab
- Click "Train Model" to start federated training
- 25 rounds, ~5-10 minutes
- Progress bar shows real-time updates

### 3. Personalize Tab
- Click "Personalize All Clients" to fine-tune per persona
- Shows accuracy improvements for each client

### 4. Privacy Audit Tab
- Click "Run Privacy Audit" for comprehensive analysis
- RDP budget, membership inference, gradient leakage

## 🔬 Technical Details

### Model Architecture
- **Type**: Attention-GRU
- **Parameters**: 332,916
- **Embedding**: 96 dimensions
- **GRU Units**: 192
- **Attention**: 64 units (Bahdanau)
- **Vocabulary**: 531 tokens
- **Context Length**: 6 tokens

### Federated Learning
- **Algorithm**: FedProx + FedNova
- **Rounds**: 25
- **Local Epochs**: 4
- **Batch Size**: 16
- **Learning Rate**: Cosine decay (8e-3 → 1e-4)
- **Loss**: Label-smoothed cross-entropy (ε=0.1)

### Privacy Guarantees
- **DP-SGD**: Gradient clipping (L2 norm=1.0) + Gaussian noise (σ=0.1)
- **RDP Accounting**: Tracks cumulative privacy budget
- **Privacy Budget**: ε tracked via Rényi DP accountant
- **Data Isolation**: Raw text never leaves device

### Prediction System
- **80%** Model probability
- **10%** Trigram statistics
- **10%** Bigram statistics
- **Latency**: 30-100ms per prediction

## 📁 Project Structure

```
Major-Project/
├── app.py                    # Flask server entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── backend/
│   ├── data/                 # Vocabulary & client datasets
│   │   ├── vocabulary.py
│   │   └── client_data.py
│   ├── model/                # Model architecture & training
│   │   ├── architecture.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── saved/            # Trained models
│   ├── federated/            # FL algorithms
│   │   ├── fedprox.py
│   │   └── fednova.py
│   ├── privacy/              # Privacy mechanisms
│   │   ├── dp_sgd.py
│   │   └── audit.py
│   ├── personalization/      # Fine-tuning
│   │   └── finetune.py
│   ├── routes/               # API endpoints
│   │   ├── predict_routes.py
│   │   ├── train_routes.py
│   │   ├── audit_routes.py
│   │   └── personalize_routes.py
│   └── state.py              # Global state management
└── frontend/
    ├── templates/
    │   └── index.html        # Main UI
    └── static/
        ├── css/
        │   └── style.css     # Styles
        ├── js/
        │   └── app.js        # Client-side logic
        └── favicon.svg       # Icon
```

## 🧪 Testing

```bash
# Run comprehensive tests
python test_comprehensive.py

# Test keyboard functionality
python test_keyboard.py

# Full system verification
python test_final_verification.py
```

## 📊 Performance

- **Vocabulary**: 531 tokens
- **Model Size**: 332,916 parameters
- **Training Time**: 5-10 minutes (25 rounds)
- **Prediction Latency**: 30-100ms
- **Accuracy**: 70-80% (varies by client)
- **TFLite Size**: ~4x smaller than float32

## 🎨 UI Themes

- 🌙 **Dark** (default)
- ☀️ **Light**
- 🌊 **Ocean**
- 🌿 **Forest**

## 🔧 API Endpoints

- `GET /` - Main UI
- `GET /api/config` - Training configuration
- `GET /api/train/status` - Training progress
- `POST /api/train` - Start training
- `POST /api/predict` - Get predictions
- `POST /api/personalize` - Start personalization
- `GET /api/personalize/status` - Personalization progress
- `POST /api/audit` - Run privacy audit
- `GET /api/audit/result` - Audit results

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:3000 app:app
```

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["python", "app.py"]
```

## 📱 Mobile Integration

The trained model is exported as INT8 TFLite for mobile deployment:

```python
# Model is saved at: backend/model/saved/keyboard_model.tflite
# Use TFLite Android/iOS API for on-device inference
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FedProx**: [Li et al., 2020](https://arxiv.org/abs/1812.06127)
- **FedNova**: [Wang et al., 2020](https://arxiv.org/abs/2007.07481)
- **DP-SGD**: [Abadi et al., 2016](https://arxiv.org/abs/1607.00133)
- **Rényi DP**: [Mironov, 2017](https://arxiv.org/abs/1702.07476)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for privacy-preserving machine learning**
