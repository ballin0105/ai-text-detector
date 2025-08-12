# AI Text Detector

A DistilBERT-based system for detecting AI-generated text.

## 🎯 Project Goal

To build a high-accuracy AI text detector capable of distinguishing between AI-generated and human-written text.

## 📁 Project Structure

```
Detector/
├── data/                   # Datasets
│   ├── raw/                # Raw data
│   └── processed/          # Processed data
├── scripts/                # Core scripts
│   ├── 01_data_collection/ # Data collection
│   ├── 02_data_generation/ # AI data generation
│   ├── 03_data_preparation/# Data preparation
│   ├── 04_model_training/  # Model training
│   └── 05_evaluation/      # Evaluation and analysis
└── utils/                  # Tools
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda create -n text_detector python=3.10
conda activate text_detector
pip install -r requirements.txt
```

### 2. Running the Pipeline
```bash
# Prepare the data
python scripts/03_data_preparation/prepare_final_dataset.py

# Train the model
python scripts/04_model_training/finetune_distilbert_model.py

# Evaluate the model
python scripts/05_evaluation/evaluate_and_analyze.py
```

## 📊 Model Performance

- Accuracy: ~99%
- F1-Score: ~0.99
- ROC-AUC: ~0.99

## 📝 License

MIT License
