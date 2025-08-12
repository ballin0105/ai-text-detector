#!/usr/bin/env python3
"""
消融实验脚本
使用传统的TF-IDF + Logistic Regression作为基线模型
用于对比深度学习模型的性能提升
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib

# --- 配置参数 ---
FINAL_DATASET_PATH = '/hy-tmp/Detector/data/processed/final_labeled_dataset.csv'
OUTPUT_DIR = '/hy-tmp/Detector/results/ablation'

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")

def run_ablation_study():
    """
    运行多个基线模型的消融研究，包括：
    1. TF-IDF + Logistic Regression
    2. TF-IDF + Random Forest
    3. TF-IDF + SVM
    """
    ensure_output_dir()
    
    print("=" * 60)
    print("消融研究 - 基线模型对比")
    print("=" * 60)

    # 1. 加载数据集
    if not os.path.exists(FINAL_DATASET_PATH):
        print(f"❌ 错误: 数据集 '{FINAL_DATASET_PATH}' 未找到。")
        return

    print("\n正在加载数据集...")
    df = pd.read_csv(FINAL_DATASET_PATH)
    
    # 检查列名并确定文本列
    text_column = None
    for col in ['abstract', 'text', 'content']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        cols = [col for col in df.columns if col != 'label']
        if cols:
            text_column = cols[0]
    
    print(f"使用文本列: '{text_column}'")
    df[text_column] = df[text_column].astype(str)

    # 2. 划分训练集和测试集（使用与深度学习模型相同的划分）
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column], df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {len(X_train)} 条")
    print(f"  测试集: {len(X_test)} 条")
    print(f"  类别分布: Human={sum(y_train==0)}/{sum(y_test==0)}, AI={sum(y_train==1)}/{sum(y_test==1)}")

    # 3. 特征提取 (TF-IDF)
    print("\n正在使用TF-IDF进行特征提取...")
    print("  配置: max_features=5000, ngram_range=(1,2)")
    
    start_time = time.time()
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),  # 使用unigram和bigram
        min_df=2,
        max_df=0.95
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)  # 修复：这里之前写成了vector.transform
    
    feature_time = time.time() - start_time
    print(f"  ✓ 特征提取完成 (耗时: {feature_time:.2f}秒)")
    print(f"  特征维度: {X_train_tfidf.shape[1]}")

    # 保存vectorizer供后续分析
    vectorizer_path = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  ✓ TF-IDF vectorizer已保存至: {vectorizer_path}")

    # 4. 训练多个基线模型
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*40}")
        print(f"训练模型: {model_name}")
        print(f"{'='*40}")
        
        # 训练模型
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test_tfidf)
        y_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Human', 'AI'], output_dict=True)
        
        # 计算AUC（如果模型支持概率预测）
        if y_proba is not None:
            auc_score = roc_auc_score(y_test, y_proba)
        else:
            auc_score = None
        
        # 保存结果
        results[model_name] = {
            'accuracy': accuracy,
            'precision_ai': report['AI']['precision'],
            'recall_ai': report['AI']['recall'],
            'f1_ai': report['AI']['f1-score'],
            'auc': auc_score,
            'train_time': train_time,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'model': model
        }
        
        # 打印结果
        print(f"\n训练时间: {train_time:.2f}秒")
        print(f"准确率: {accuracy:.4f}")
        if auc_score:
            print(f"ROC-AUC: {auc_score:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        
        # 保存模型
        model_path = os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ", "_")}_model.pkl')
        joblib.dump(model, model_path)
        print(f"✓ 模型已保存至: {model_path}")

    # 5. 生成对比图表
    print("\n" + "=" * 60)
    print("生成对比分析图表")
    print("=" * 60)
    
    # 5.1 性能对比条形图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准备数据
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision_ai', 'recall_ai', 'f1_ai']
    metric_labels = ['Accuracy', 'Precision (AI)', 'Recall (AI)', 'F1-Score (AI)']
    
    # 左图：主要指标对比
    ax1 = axes[0]
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in model_names]
        ax1.bar(x + i*width, values, width, label=label)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison - Baseline Models')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # 右图：训练时间对比
    ax2 = axes[1]
    train_times = [results[m]['train_time'] for m in model_names]
    bars = ax2.bar(model_names, train_times, color='skyblue')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    
    # 添加数值标签
    for bar, time_val in zip(bars, train_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, 'baseline_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ 对比图表已保存至: {comparison_path}")
    
    # 5.2 ROC曲线对比
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        if result['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
            auc_score = result['auc']
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Baseline Models Comparison', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_comparison_path = os.path.join(OUTPUT_DIR, 'roc_curves_comparison.png')
    plt.tight_layout()
    plt.savefig(roc_comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ ROC曲线对比已保存至: {roc_comparison_path}")
    
    # 6. 生成结果汇总表
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision (AI)': f"{result['precision_ai']:.4f}",
            'Recall (AI)': f"{result['recall_ai']:.4f}",
            'F1-Score (AI)': f"{result['f1_ai']:.4f}",
            'ROC-AUC': f"{result['auc']:.4f}" if result['auc'] else 'N/A',
            'Training Time (s)': f"{result['train_time']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, 'baseline_models_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 60)
    print("消融研究总结")
    print("=" * 60)
    print("\n基线模型性能对比:")
    print(summary_df.to_string(index=False))
    
    print(f"\n✓ 汇总表已保存至: {summary_path}")
    
    # 7. 特征重要性分析（针对Logistic Regression）
    print("\n" + "=" * 60)
    print("特征重要性分析 (TF-IDF + Logistic Regression)")
    print("=" * 60)
    
    lr_model = results['Logistic Regression']['model']
    feature_names = vectorizer.get_feature_names_out()
    
    # 获取最重要的特征（区分AI和Human的）
    coef = lr_model.coef_[0]
    top_ai_indices = np.argsort(coef)[-20:][::-1]  # Top 20 AI indicators
    top_human_indices = np.argsort(coef)[:20]  # Top 20 Human indicators
    
    print("\nTop 20 特征 - 指示AI生成文本:")
    for i, idx in enumerate(top_ai_indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:30s} (权重: {coef[idx]:.4f})")
    
    print("\nTop 20 特征 - 指示人类撰写文本:")
    for i, idx in enumerate(top_human_indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:30s} (权重: {coef[idx]:.4f})")
    
    # 保存特征重要性
    feature_importance = pd.DataFrame({
        'AI_Features': [feature_names[i] for i in top_ai_indices],
        'AI_Weights': [coef[i] for i in top_ai_indices],
        'Human_Features': [feature_names[i] for i in top_human_indices],
        'Human_Weights': [coef[i] for i in top_human_indices]
    })
    
    feature_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    feature_importance.to_csv(feature_path, index=False)
    print(f"\n✓ 特征重要性已保存至: {feature_path}")
    
    print("\n" + "=" * 60)
    print("✅ 消融研究完成！")
    print("=" * 60)
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("\n💡 对比建议:")
    print("1. 将这些基线模型的结果与DistilBERT模型进行对比")
    print("2. 关注深度学习模型在哪些指标上有显著提升")
    print("3. 分析不同方法的优缺点和适用场景")

if __name__ == '__main__':
    run_ablation_study()