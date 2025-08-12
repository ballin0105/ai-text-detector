#!/usr/bin/env python3
"""
DistilBERT模型微调脚本
用于训练一个能够区分AI生成文本和人类撰写文本的分类器
"""

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# === Hugging Face 镜像配置（中国大陆用户） ===
# 如果需要使用镜像，请将 USE_MIRROR 设置为 True
USE_MIRROR = True  # 设置为 True 使用镜像源

if USE_MIRROR:
    # 方法1：设置环境变量（推荐）
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 方法2：直接修改 transformers 的下载地址
    import transformers
    # 设置镜像URL
    transformers.utils.hub.HUGGINGFACE_CO_PREFIX = "https://hf-mirror.com"
    
    print("🌐 已配置使用 Hugging Face 镜像源: https://hf-mirror.com")
    print("   如需更换镜像源，请修改脚本中的镜像配置\n")
    
    # 备选镜像源列表
    # 1. hf-mirror.com (推荐)
    # 2. mirrors.bfsu.edu.cn/hugging-face-models
    # 3. mirrors.tuna.tsinghua.edu.cn/hugging-face-models

# --- 配置参数 ---
FINAL_DATASET_PATH = '/hy-tmp/Detector/data/processed/final_labeled_dataset.csv'
# 使用本地已下载的模型
MODEL_NAME = './model_cache/distilbert-base-uncased'  # 改为本地路径
OUTPUT_DIR = '/hy-tmp/Detector/results/model'
LOGGING_DIR = '/hy-tmp/Detector/results/logs'
BEST_MODEL_DIR = '/hy-tmp/Detector/results/best_model'

# 训练超参数
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 16  # RTX 3090 可以用更大的批次
MAX_TOKEN_LENGTH = 256  # 摘要通常不会太长，256足够
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# 早停参数
EARLY_STOPPING_PATIENCE = 2

def print_separator(title="", char="=", length=60):
    """打印分隔符"""
    if title:
        print(f"\n{char * length}")
        print(f"{title.center(length)}")
        print(f"{char * length}")
    else:
        print(f"{char * length}")

def check_environment():
    """检查运行环境"""
    print_separator("环境检查")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ CUDA不可用，将使用CPU训练（速度会较慢）")
    
    # 检查数据文件
    if os.path.exists(FINAL_DATASET_PATH):
        print(f"✓ 数据集文件存在: {FINAL_DATASET_PATH}")
    else:
        print(f"✗ 数据集文件不存在: {FINAL_DATASET_PATH}")
        return False
    
    return True

def download_model_with_retry(model_name, max_retries=3):
    """
    加载本地模型或下载模型
    """
    # 检查是否使用本地路径
    if os.path.exists(model_name):
        print(f"使用本地模型: {model_name}")
        try:
            # 直接从本地加载
            tokenizer = DistilBertTokenizerFast.from_pretrained(
                model_name,
                local_files_only=True,  # 只使用本地文件
            )
            
            model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                id2label={0: "Human", 1: "AI"},
                label2id={"Human": 0, "AI": 1},
                local_files_only=True,  # 只使用本地文件
            )
            
            print("  ✓ 本地模型加载成功！")
            return tokenizer, model
            
        except Exception as e:
            print(f"  ✗ 本地模型加载失败: {str(e)}")
            raise e
    
    # 如果不是本地路径，尝试下载（原有的下载逻辑）
    print(f"开始下载模型: {model_name}")
    
    # 镜像源列表
    mirror_urls = [
        "https://hf-mirror.com",
        "https://mirrors.bfsu.edu.cn/hugging-face-models",
        "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    ]
    
    for mirror_idx, mirror_url in enumerate(mirror_urls):
        print(f"\n尝试镜像源 {mirror_idx + 1}: {mirror_url}")
        
        # 设置当前镜像
        os.environ['HF_ENDPOINT'] = mirror_url
        
        # 如果使用 hf-mirror.com，需要特殊处理
        if "hf-mirror" in mirror_url:
            # 构建完整的模型URL
            model_url = f"{mirror_url}/{model_name}"
        else:
            model_url = model_name
            
        for attempt in range(max_retries):
            try:
                print(f"  尝试 {attempt + 1}/{max_retries}...")
                
                # 尝试使用镜像URL下载
                if "hf-mirror" in mirror_url:
                    # 对于 hf-mirror，直接使用完整URL
                    tokenizer = DistilBertTokenizerFast.from_pretrained(
                        model_url,
                        cache_dir="./model_cache",
                        resume_download=True,
                        force_download=False,
                        local_files_only=False,
                        trust_remote_code=True,
                    )
                    
                    model = DistilBertForSequenceClassification.from_pretrained(
                        model_url,
                        num_labels=2,
                        id2label={0: "Human", 1: "AI"},
                        label2id={"Human": 0, "AI": 1},
                        cache_dir="./model_cache",
                        resume_download=True,
                        force_download=False,
                        local_files_only=False,
                        trust_remote_code=True,
                    )
                else:
                    # 对于其他镜像，使用模型名
                    tokenizer = DistilBertTokenizerFast.from_pretrained(
                        model_name,
                        cache_dir="./model_cache",
                        resume_download=True,
                        force_download=False,
                    )
                    
                    model = DistilBertForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=2,
                        id2label={0: "Human", 1: "AI"},
                        label2id={"Human": 0, "AI": 1},
                        cache_dir="./model_cache",
                        resume_download=True,
                        force_download=False,
                    )
                
                print("  ✓ 模型下载成功！")
                return tokenizer, model
                
            except Exception as e:
                print(f"  ✗ 下载失败: {str(e)[:200]}")
                if attempt < max_retries - 1:
                    print(f"  等待3秒后重试...")
                    import time
                    time.sleep(3)
                else:
                    print(f"  该镜像源失败，尝试下一个...")
                    break
    
    # 如果所有镜像都失败，尝试手动下载方案
    print("\n❌ 所有镜像源都下载失败！")
    print("\n📌 手动下载解决方案：")
    print("1. 访问以下任一链接下载模型文件：")
    print("   - https://hf-mirror.com/distilbert-base-uncased")
    print("   - https://huggingface.co/distilbert-base-uncased")
    print("\n2. 下载以下文件到 ./model_cache/distilbert-base-uncased/ 目录：")
    print("   - config.json")
    print("   - pytorch_model.bin")
    print("   - tokenizer_config.json")
    print("   - tokenizer.json")
    print("   - vocab.txt")
    print("\n3. 使用代理方案：")
    print("   export https_proxy=http://your-proxy:port")
    print("   export http_proxy=http://your-proxy:port")
    print("\n4. 或使用 huggingface-cli 下载：")
    print("   pip install -U huggingface_hub")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    print("   huggingface-cli download distilbert-base-uncased --local-dir ./model_cache/distilbert-base-uncased")
    
    raise Exception("无法下载模型，请尝试手动下载")

def load_and_prepare_dataset():
    """加载数据集并划分为训练集和测试集"""
    print_separator("加载和准备数据集")
    
    # 读取CSV文件
    df = pd.read_csv(FINAL_DATASET_PATH)
    print(f"总数据量: {len(df)} 条")
    
    # 检查列名并确定文本列
    text_column = None
    for col in ['abstract', 'text', 'content']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        # 使用第一个非label列作为文本列
        cols = [col for col in df.columns if col != 'label']
        if cols:
            text_column = cols[0]
    
    print(f"使用文本列: '{text_column}'")
    
    # 确保文本列是字符串类型
    df[text_column] = df[text_column].astype(str)
    
    # 如果原始列名不是'text'，重命名为'text'以统一处理
    if text_column != 'text':
        df = df.rename(columns={text_column: 'text'})
    
    # 显示标签分布
    label_counts = df['label'].value_counts().sort_index()
    print("\n标签分布:")
    print(f"  人类文本 (label=0): {label_counts.get(0, 0)} 条")
    print(f"  AI文本 (label=1): {label_counts.get(1, 0)} 条")
    
    # 创建Hugging Face Dataset对象
    dataset = Dataset.from_pandas(df[['text', 'label']])
    
    # 80/20 划分训练集和测试集
    train_test_split_dict = dataset.train_test_split(test_size=0.2, seed=42)
    
    print("\n数据集划分:")
    print(f"  训练集: {len(train_test_split_dict['train'])} 条")
    print(f"  测试集: {len(train_test_split_dict['test'])} 条")
    
    return train_test_split_dict

def tokenize_data(dataset_dict, tokenizer):
    """对数据集进行分词"""
    print_separator("数据分词")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH
        )
    
    print("正在对训练集进行分词...")
    tokenized_datasets = dataset_dict.map(
        tokenize_function, 
        batched=True,
        desc="Tokenizing"
    )
    
    print("✓ 分词完成")
    print(f"  最大序列长度: {MAX_TOKEN_LENGTH}")
    
    return tokenized_datasets

def compute_metrics(pred):
    """定义评估指标计算函数"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # 计算各项指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 分别计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'precision_human': precision_per_class[0] if len(precision_per_class) > 0 else 0,
        'precision_ai': precision_per_class[1] if len(precision_per_class) > 1 else 0,
        'recall_human': recall_per_class[0] if len(recall_per_class) > 0 else 0,
        'recall_ai': recall_per_class[1] if len(recall_per_class) > 1 else 0,
    }

def create_model_card(trainer, training_args, model_path):
    """创建模型卡片，记录模型信息"""
    model_card = {
        "model_name": "DistilBERT AI Text Detector",
        "base_model": MODEL_NAME,
        "task": "Binary Classification (AI vs Human Text)",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_parameters": {
            "num_epochs": NUM_TRAIN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_token_length": MAX_TOKEN_LENGTH,
            "warmup_steps": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
        },
        "dataset_info": {
            "path": FINAL_DATASET_PATH,
            "train_size": len(trainer.train_dataset),
            "test_size": len(trainer.eval_dataset),
        },
        "performance_metrics": None  # 将在训练后更新
    }
    
    # 保存模型卡片
    card_path = os.path.join(model_path, "model_card.json")
    with open(card_path, 'w', encoding='utf-8') as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)
    
    return card_path

def main():
    """主函数，执行模型微调全过程"""
    print_separator("DistilBERT模型微调 - AI文本检测器", "=", 60)
    
    # 0. 环境检查
    if not check_environment():
        print("\n❌ 环境检查失败，请确保数据集文件存在")
        return
    
    # 1. 加载和准备数据集
    dataset_dict = load_and_prepare_dataset()
    
    # 2. 初始化分词器和模型（使用重试机制）
    print_separator("初始化模型和分词器")
    print(f"模型名称: {MODEL_NAME}")
    
    if USE_MIRROR:
        print("使用镜像源下载模型...")
    
    try:
        tokenizer, model = download_model_with_retry(MODEL_NAME)
    except Exception as e:
        print(f"\n模型下载失败: {e}")
        return
    
    # 3. 分词
    tokenized_datasets = tokenize_data(dataset_dict, tokenizer)
    
    # 4. 配置模型设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ 模型将在 {device.upper()} 上进行训练")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    model.to(device)
    
    # 5. 定义训练参数
    print_separator("配置训练参数")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        logging_steps=50,
        eval_strategy="epoch",  # 使用新的参数名
        save_strategy="epoch",  # 每个epoch结束后保存模型
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="f1",  # 使用f1分数来判断最佳模型
        greater_is_better=True,
        save_total_limit=2,  # 只保留最好的2个checkpoint
        report_to=["tensorboard"],  # RTX 3090 可以使用 tensorboard
        fp16=True,  # RTX 3090 支持混合精度训练，加速训练
        gradient_checkpointing=False,  # RTX 3090 有足够显存，不需要
        optim="adamw_torch",  # 优化器
        seed=42,  # 随机种子
        max_grad_norm=1.0,
    )
    
    print("训练配置:")
    print(f"  训练轮数: {NUM_TRAIN_EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  最大序列长度: {MAX_TOKEN_LENGTH}")
    print(f"  混合精度训练: {training_args.fp16}")
    
    # 6. 初始化Trainer
    print_separator("初始化Trainer")
    
    # 创建早停回调
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=0.001
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    
    # 7. 开始训练！
    print_separator("开始训练", "▶", 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 训练
    train_result = trainer.train()
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator("训练完成", "✓", 60)
    
    # 8. 评估最终模型
    print_separator("模型评估")
    eval_result = trainer.evaluate()
    
    print("测试集性能:")
    for key, value in eval_result.items():
        if key.startswith('eval_'):
            metric_name = key.replace('eval_', '')
            if isinstance(value, float):
                print(f"  {metric_name:15s}: {value:.4f}")
    
    # 9. 保存最佳模型
    print_separator("保存模型")
    
    # 保存模型和分词器
    trainer.save_model(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    print(f"✓ 模型已保存至: {BEST_MODEL_DIR}")
    
    # 创建并保存模型卡片
    model_card_path = create_model_card(trainer, training_args, BEST_MODEL_DIR)
    
    # 更新模型卡片的性能指标
    with open(model_card_path, 'r') as f:
        model_card = json.load(f)
    
    model_card['performance_metrics'] = {
        k.replace('eval_', ''): v 
        for k, v in eval_result.items() 
        if k.startswith('eval_')
    }
    
    with open(model_card_path, 'w') as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 模型卡片已保存至: {model_card_path}")
    
    # 10. 总结
    print_separator("训练总结", "📊", 60)
    print(f"最佳F1分数: {eval_result.get('eval_f1', 0):.4f}")
    print(f"准确率: {eval_result.get('eval_accuracy', 0):.4f}")
    print(f"精确率: {eval_result.get('eval_precision', 0):.4f}")
    print(f"召回率: {eval_result.get('eval_recall', 0):.4f}")
    print("\n分类别性能:")
    print(f"  人类文本 - 精确率: {eval_result.get('eval_precision_human', 0):.4f}, "
          f"召回率: {eval_result.get('eval_recall_human', 0):.4f}")
    print(f"  AI文本 - 精确率: {eval_result.get('eval_precision_ai', 0):.4f}, "
          f"召回率: {eval_result.get('eval_recall_ai', 0):.4f}")
    
    print_separator("", "=", 60)
    print("🎉 恭喜！AI文本检测器训练完成！")
    print(f"📁 模型位置: {BEST_MODEL_DIR}")
    print("📝 您现在可以使用这个模型来检测AI生成的文本了")
    print_separator("", "=", 60)

if __name__ == '__main__':
    main()