#!/usr/bin/env python3
"""
模型评估与深度分析脚本 - 论文发表级别
生成高质量矢量图PDF，可直接用于学术论文
"""

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
from scipy.special import softmax

# ========== 配置matplotlib为论文发表质量 ==========
# 设置全局字体和样式
plt.rcParams.update({
    # 字体设置
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    
    # 数学文本
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    
    # 轴线设置
    'axes.linewidth': 1.0,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.grid': True,
    
    # 刻度设置
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    
    # 图例设置
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.fancybox': True,
    
    # 线条设置
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    
    # 图形输出设置
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # PDF设置
    'pdf.fonttype': 42,  # TrueType字体
    'ps.fonttype': 42,   # TrueType字体
})

# Seaborn样式设置
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# --- 配置参数 ---
FINAL_DATASET_PATH = '/hy-tmp/Detector/data/processed/final_labeled_dataset.csv'
BEST_MODEL_PATH = '/hy-tmp/Detector/results/best_model' 
OUTPUT_DIR = '/hy-tmp/Detector/results/evaluation'

MAX_TOKEN_LENGTH = 256  # 必须与训练时保持一致

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")

def create_publication_confusion_matrix(cm, output_path):
    """创建论文级别的混淆矩阵图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建注释文本（显示数量和百分比）
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # 绘制热力图
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                cbar_kws={'label': 'Number of Samples'},
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'],
                square=True,
                linewidths=1,
                linecolor='gray',
                cbar=True,
                vmin=0)
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix of AI Text Detection Model', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存为PDF矢量图
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    # 同时保存PNG供预览
    fig.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_publication_roc_curve(fpr, tpr, roc_auc, output_path):
    """创建论文级别的ROC曲线图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制ROC曲线
    ax.plot(fpr, tpr, color='#FF6B6B', lw=2.5, 
            label=f'DistilBERT (AUC = {roc_auc:.3f})', 
            linestyle='-', marker='', alpha=0.9)
    
    # 绘制对角线（随机分类器）
    ax.plot([0, 1], [0, 1], color='#4ECDC4', lw=2, 
            linestyle='--', label='Random Classifier (AUC = 0.500)', alpha=0.7)
    
    # 填充ROC曲线下方区域
    ax.fill_between(fpr, 0, tpr, alpha=0.15, color='#FF6B6B')
    
    # 设置轴标签和标题
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 设置轴范围和刻度
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(loc='lower right', frameon=True, fancybox=True, 
              shadow=True, framealpha=0.95)
    
    # 添加性能点标记（例如，最佳阈值点）
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = (fpr[optimal_idx], tpr[optimal_idx])
    ax.scatter(*optimal_threshold, color='red', s=100, zorder=5, 
               label=f'Optimal Threshold ({optimal_threshold[0]:.3f}, {optimal_threshold[1]:.3f})')
    
    # 设置等比例
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # 保存为PDF矢量图
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_publication_pr_curve(precision, recall, avg_precision, output_path):
    """创建论文级别的Precision-Recall曲线图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制PR曲线
    ax.plot(recall, precision, color='#6C5CE7', lw=2.5,
            label=f'DistilBERT (AP = {avg_precision:.3f})',
            linestyle='-', alpha=0.9)
    
    # 填充PR曲线下方区域
    ax.fill_between(recall, 0, precision, alpha=0.15, color='#6C5CE7')
    
    # 添加基线（随机分类器的精确率）
    baseline_precision = 0.5  # 假设类别平衡
    ax.axhline(y=baseline_precision, color='#FD79A8', lw=2, 
               linestyle='--', label=f'Random Classifier (AP = {baseline_precision:.3f})', 
               alpha=0.7)
    
    # 设置轴标签和标题
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curve', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 设置轴范围和刻度
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 设置图例
    ax.legend(loc='lower left', frameon=True, fancybox=True, 
              shadow=True, framealpha=0.95)
    
    # 设置等比例
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # 保存为PDF矢量图
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_combined_performance_plot(results_dict, output_path):
    """创建综合性能对比图（条形图）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：主要指标对比
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = results_dict['main_metrics']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax1.set_title('Classification Performance Metrics', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax1.set_axisbelow(True)
    
    # 右图：ROC-AUC和PR-AUC对比
    auc_metrics = ['ROC-AUC', 'PR-AUC']
    auc_values = results_dict['auc_metrics']
    colors2 = ['#9b59b6', '#1abc9c']
    
    bars2 = ax2.bar(auc_metrics, auc_values, color=colors2, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, val in zip(bars2, auc_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax2.set_title('Area Under Curve Metrics', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax2.set_axisbelow(True)
    
    # 调整布局
    plt.suptitle('AI Text Detection Model Performance Summary', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存为PDF矢量图
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_error_distribution_plot(error_df, output_path):
    """创建错误分布图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：错误类型分布
    error_counts = error_df['error_type'].value_counts()
    colors = ['#e74c3c', '#3498db']
    wedges, texts, autotexts = ax1.pie(error_counts.values, 
                                        labels=error_counts.index,
                                        colors=colors,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        explode=(0.05, 0.05))
    
    # 美化饼图文本
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax1.set_title('Error Type Distribution', fontsize=14, fontweight='bold')
    
    # 右图：置信度分布直方图
    ax2.hist(error_df['prediction_confidence'], bins=20, 
             color='#34495e', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
    ax2.set_title('Error Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax2.set_axisbelow(True)
    
    # 添加均值线
    mean_conf = error_df['prediction_confidence'].mean()
    ax2.axvline(mean_conf, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_conf:.3f}')
    ax2.legend()
    
    plt.suptitle('Error Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存为PDF矢量图
    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def create_all_figures_pdf(all_figures, output_path):
    """将所有图形合并到一个PDF文件中"""
    with PdfPages(output_path) as pdf:
        for fig in all_figures:
            pdf.savefig(fig, bbox_inches='tight')
        
        # 添加元数据
        d = pdf.infodict()
        d['Title'] = 'AI Text Detection Model Evaluation Results'
        d['Author'] = 'AI Text Detector'
        d['Subject'] = 'Model Performance Analysis'
        d['Keywords'] = 'Machine Learning, NLP, Text Classification, DistilBERT'

def analyze_model_performance():
    """
    加载最佳模型，在测试集上进行评估，并生成论文级别的可视化结果。
    """
    ensure_output_dir()
    
    # 存储所有图形对象
    all_figures = []
    
    # 1. 加载和准备数据集
    print("=" * 60)
    print("模型评估与分析 - 论文发表级别")
    print("=" * 60)
    
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
    
    # 使用相同的随机种子划分数据集
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    if text_column != 'text':
        test_df = test_df.rename(columns={text_column: 'text'})
    
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    print(f"测试集大小: {len(test_dataset)}")
    print(f"  人类文本: {sum(test_df['label'] == 0)} 条")
    print(f"  AI文本: {sum(test_df['label'] == 1)} 条")

    # 2. 加载模型和分词器
    print(f"\n正在从 '{BEST_MODEL_PATH}' 加载模型和分词器...")
    if not os.path.exists(BEST_MODEL_PATH):
        print("❌ 错误：找不到训练好的模型。")
        return
        
    tokenizer = DistilBertTokenizerFast.from_pretrained(BEST_MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(BEST_MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"✓ 模型已加载到 {device.upper()}")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_TOKEN_LENGTH
        )
    
    print("\n正在对测试集进行分词...")
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 3. 进行预测
    print("\n正在对测试集进行预测...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
    )
    predictions = trainer.predict(tokenized_test_dataset)
    
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    y_proba = softmax(predictions.predictions, axis=1)
    y_proba_ai = y_proba[:, 1]

    # 4. 生成分类报告
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=['Human', 'AI'], digits=4)
    print(report)
    
    report_dict = classification_report(y_true, y_pred, target_names=['Human', 'AI'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(OUTPUT_DIR, 'classification_report.csv')
    report_df.to_csv(report_path)

    # 5. 生成混淆矩阵（PDF）
    print("\n正在生成混淆矩阵...")
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(OUTPUT_DIR, 'figures', 'confusion_matrix.pdf')
    fig_cm = create_publication_confusion_matrix(cm, cm_path)
    print(f"✓ 混淆矩阵已保存至: {cm_path}")

    # 6. 生成ROC曲线（PDF）
    print("\n正在生成ROC/AUC曲线...")
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_ai)
    roc_auc = auc(fpr, tpr)
    roc_path = os.path.join(OUTPUT_DIR, 'figures', 'roc_curve.pdf')
    fig_roc = create_publication_roc_curve(fpr, tpr, roc_auc, roc_path)
    print(f"✓ ROC曲线已保存至: {roc_path}")
    print(f"  AUC分数: {roc_auc:.4f}")

    # 7. 生成Precision-Recall曲线（PDF）
    print("\n正在生成Precision-Recall曲线...")
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba_ai)
    avg_precision = average_precision_score(y_true, y_proba_ai)
    pr_path = os.path.join(OUTPUT_DIR, 'figures', 'pr_curve.pdf')
    fig_pr = create_publication_pr_curve(precision, recall, avg_precision, pr_path)
    print(f"✓ Precision-Recall曲线已保存至: {pr_path}")
    print(f"  平均精确率: {avg_precision:.4f}")

    # 8. 误差分析
    print("\n正在进行误差分析...")
    test_df_copy = test_df.copy()
    test_df_copy['predicted_label'] = y_pred
    test_df_copy['prediction_confidence'] = y_proba_ai
    test_df_copy['prediction_correct'] = (test_df_copy['label'] == test_df_copy['predicted_label'])
    
    error_df = test_df_copy[~test_df_copy['prediction_correct']].copy()
    error_df['error_type'] = error_df.apply(
        lambda row: 'False Positive' if row['label'] == 0 else 'False Negative',
        axis=1
    )
    
    error_df = error_df.sort_values('prediction_confidence', ascending=False)
    error_analysis_path = os.path.join(OUTPUT_DIR, 'error_analysis.csv')
    error_df.to_csv(error_analysis_path, index=False, encoding='utf-8')
    
    # 生成错误分布图（PDF）
    if len(error_df) > 0:
        error_plot_path = os.path.join(OUTPUT_DIR, 'figures', 'error_distribution.pdf')
        fig_error = create_error_distribution_plot(error_df, error_plot_path)
        print(f"✓ 错误分布图已保存至: {error_plot_path}")

    # 9. 生成综合性能图（PDF）
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    results_dict = {
        'main_metrics': [accuracy, precision_val, recall_val, f1],
        'auc_metrics': [roc_auc, avg_precision]
    }
    
    performance_path = os.path.join(OUTPUT_DIR, 'figures', 'performance_summary.pdf')
    fig_perf = create_combined_performance_plot(results_dict, performance_path)
    print(f"✓ 性能总结图已保存至: {performance_path}")
    
    # 10. 创建合并的PDF文件（包含所有图形）
    print("\n正在创建包含所有图形的PDF文件...")
    combined_pdf_path = os.path.join(OUTPUT_DIR, 'all_figures_combined.pdf')
    
    # 注意：由于我们已经关闭了各个图形，这里我们重新打开PDF文件
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    
    pdf_files = [
        os.path.join(OUTPUT_DIR, 'figures', 'confusion_matrix.pdf'),
        os.path.join(OUTPUT_DIR, 'figures', 'roc_curve.pdf'),
        os.path.join(OUTPUT_DIR, 'figures', 'pr_curve.pdf'),
        os.path.join(OUTPUT_DIR, 'figures', 'performance_summary.pdf'),
    ]
    
    if len(error_df) > 0:
        pdf_files.append(os.path.join(OUTPUT_DIR, 'figures', 'error_distribution.pdf'))
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            merger.append(pdf_file)
    
    merger.write(combined_pdf_path)
    merger.close()
    print(f"✓ 合并PDF已保存至: {combined_pdf_path}")
    
    # 11. 生成LaTeX表格代码
    print("\n生成LaTeX表格代码...")
    latex_path = os.path.join(OUTPUT_DIR, 'latex_tables.tex')
    with open(latex_path, 'w') as f:
        f.write("% Classification Report Table\n")
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\caption{Classification Report}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Class & Precision & Recall & F1-Score & Support \\\\\n")
        f.write("\\hline\n")
        f.write(f"Human & {report_dict['Human']['precision']:.3f} & ")
        f.write(f"{report_dict['Human']['recall']:.3f} & ")
        f.write(f"{report_dict['Human']['f1-score']:.3f} & ")
        f.write(f"{int(report_dict['Human']['support'])} \\\\\n")
        f.write(f"AI & {report_dict['AI']['precision']:.3f} & ")
        f.write(f"{report_dict['AI']['recall']:.3f} & ")
        f.write(f"{report_dict['AI']['f1-score']:.3f} & ")
        f.write(f"{int(report_dict['AI']['support'])} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Accuracy & \\multicolumn{{3}}{{c}}{{ {accuracy:.3f} }} & {len(y_true)} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        f.write("% Performance Metrics Table\n")
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Metrics}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("Metric & Score \\\\\n")
        f.write("\\hline\n")
        f.write(f"Accuracy & {accuracy:.3f} \\\\\n")
        f.write(f"Precision & {precision_val:.3f} \\\\\n")
        f.write(f"Recall & {recall_val:.3f} \\\\\n")
        f.write(f"F1-Score & {f1:.3f} \\\\\n")
        f.write(f"ROC-AUC & {roc_auc:.3f} \\\\\n")
        f.write(f"PR-AUC & {avg_precision:.3f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ LaTeX表格代码已保存至: {latex_path}")
    
    print("\n" + "=" * 60)
    print("✅ 评估完成！")
    print("=" * 60)
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("\n📊 论文级别图形文件:")
    print(f"  PDF矢量图: {OUTPUT_DIR}/figures/*.pdf")
    print(f"  PNG预览图: {OUTPUT_DIR}/figures/*.png")
    print(f"  合并PDF: {OUTPUT_DIR}/all_figures_combined.pdf")
    print(f"  LaTeX代码: {OUTPUT_DIR}/latex_tables.tex")
    print("\n这些图形可以直接用于学术论文发表！")

if __name__ == '__main__':
    # 检查是否安装了PyPDF2
    try:
        import PyPDF2
    except ImportError:
        print("请安装PyPDF2以合并PDF文件: pip install PyPDF2")
    
    analyze_model_performance()