# AI Text Detector

基于 DistilBERT 的 AI 生成文本检测系统

## 🎯 项目目标

构建一个高精度的AI文本检测器，能够区分AI生成的文本和人类撰写的文本。

## 📁 项目结构

```
Detector/
├── data/                    # 数据集
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后数据
├── scripts/                # 核心脚本
│   ├── 01_data_collection/ # 数据收集
│   ├── 02_data_generation/ # AI数据生成
│   ├── 03_data_preparation/# 数据准备
│   ├── 04_model_training/  # 模型训练
│   └── 05_evaluation/      # 评估分析
├── model_cache/            # 模型缓存
├── results/                # 实验结果
└── utils/                  # 工具脚本
```

## 🚀 快速开始

### 1. 环境配置
```bash
conda create -n text_detector python=3.10
conda activate text_detector
pip install -r requirements.txt
```

### 2. 运行流程
```bash
# 准备数据
python scripts/03_data_preparation/prepare_final_dataset.py

# 训练模型
python scripts/04_model_training/finetune_distilbert_model.py

# 评估模型
python scripts/05_evaluation/evaluate_and_analyze.py
```

## 📊 模型性能

- 准确率: ~95%
- F1分数: ~0.95
- ROC-AUC: ~0.98

## 📝 License

MIT License
