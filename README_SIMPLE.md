# 🔐 Federated Learning Keyboard

**Privacy-first next-word prediction using Federated Learning**

## ✅ Status: READY TO USE

All systems verified and operational!

## 🚀 Quick Start

```bash
# Start the server
python app.py

# Open browser
http://localhost:3000
```

## 📋 Features

- ✅ Real-time next-word predictions (30-100ms)
- ✅ 5 client personas (Social, Professional, Tech, Food, Travel)
- ✅ Attention-GRU model (332,916 parameters)
- ✅ FedProx + FedNova aggregation
- ✅ DP-SGD privacy guarantees
- ✅ Trigram + Bigram re-ranking
- ✅ On-device personalization
- ✅ Privacy audit tools

## 🎯 How to Use

1. **Keyboard Tab** - Type and get real-time predictions
2. **Train Tab** - Train the federated model (25 rounds)
3. **Personalize Tab** - Fine-tune per client
4. **Privacy Audit Tab** - Check privacy guarantees

## ⌨️ Keyboard Shortcuts

- `Tab` - Accept top prediction
- `Enter` - Refresh predictions
- `Esc` - Clear all

## 🧪 Run Tests

```bash
python test_comprehensive.py      # Component tests
python test_keyboard.py           # Keyboard functionality
python test_final_verification.py # Complete system check
```

## 📊 Model Details

- **Architecture**: Attention-GRU
- **Parameters**: 332,916
- **Vocabulary**: 531 tokens
- **Context length**: 6 tokens
- **Privacy**: DP-SGD with RDP accounting

## 🎨 UI Themes

Click theme buttons in sidebar:
- 🌙 Dark (default)
- ☀️ Light
- 🌊 Ocean
- 🌿 Forest

## 📁 Project Structure

```
Major Project/
├── app.py                    # Main server
├── backend/
│   ├── data/                 # Vocabulary & client data
│   ├── model/                # Architecture & training
│   ├── federated/            # FedProx & FedNova
│   ├── privacy/              # DP-SGD & audit
│   ├── personalization/      # Fine-tuning
│   └── routes/               # API endpoints
└── frontend/
    ├── templates/            # HTML
    └── static/               # CSS & JavaScript
```

## 🔧 Requirements

- Python 3.10+
- TensorFlow 2.21+
- Flask
- NumPy

Install: `pip install -r requirements.txt`

## 📖 Documentation

- `SYSTEM_READY.txt` - Complete system documentation
- `QUICK_START.txt` - Quick start guide
- `VERIFICATION_REPORT.txt` - Test results

## ✨ Everything Works!

All tests passed. The keyboard is fully functional and ready to demonstrate.

**Start typing and watch the magic happen!** 🎉
