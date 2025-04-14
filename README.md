
# 📦 Monthly Model Trainer

A complete Python pipeline for monthly product category prediction, integrating active learning and monitoring.

## Features
- 🔁 Monthly model retraining
- 📊 Monitoring (without labels)
- 🤖 Active learning: detect low-confidence predictions
- ✅ CLI-friendly automation

## Usage
```bash
pip install -r requirements.txt
python update_and_run.py --month 202504 --check
```

Check `/results/` for predictions and review suggestions.
