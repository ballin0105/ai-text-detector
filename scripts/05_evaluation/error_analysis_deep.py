#!/usr/bin/env python3
"""
简化版深度错误分析脚本（无需NLTK）
详细分析模型预测错误的样本，找出共同特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# 配置参数
ERROR_ANALYSIS_PATH = '/hy-tmp/Detector/results/evaluation/error_analysis.csv'
OUTPUT_DIR = '/hy-tmp/Detector/results/evaluation/error_deep_analysis'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def simple_tokenize(text):
    """简单的分词函数"""
    # 移除标点，转小写，分词
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words

def sentence_split(text):
    """简单的句子分割"""
    # 使用常见的句子结束符分割
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def analyze_text_features(text):
    """分析文本的各种特征（简化版）"""
    features = {}
    
    # 基本统计
    features['length'] = len(text)
    words = simple_tokenize(text)
    features['word_count'] = len(words)
    sentences = sentence_split(text)
    features['sentence_count'] = len(sentences)
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
    
    # 词汇多样性
    features['unique_words'] = len(set(words))
    features['lexical_diversity'] = features['unique_words'] / features['word_count'] if features['word_count'] > 0 else 0
    
    # 标点符号使用
    features['punctuation_count'] = len(re.findall(r'[.,;:!?\-\(\)]', text))
    features['punctuation_ratio'] = features['punctuation_count'] / features['word_count'] if features['word_count'] > 0 else 0
    
    # 大写词比例（可能表示专有名词、缩写等）
    all_words = re.findall(r'\b[A-Za-z]+\b', text)
    features['capitalized_words'] = len([w for w in all_words if w[0].isupper()])
    features['capital_ratio'] = features['capitalized_words'] / len(all_words) if all_words else 0
    
    # 数字和符号
    features['contains_numbers'] = bool(re.search(r'\d', text))
    features['number_count'] = len(re.findall(r'\d+', text))
    
    return features

def analyze_linguistic_patterns(text):
    """分析语言模式（简化版）"""
    patterns = {}
    
    # 常见的AI/学术写作模式
    ai_keywords = [
        'we propose', 'we present', 'we introduce', 'our approach', 'our method',
        'in this paper', 'in this work', 'furthermore', 'moreover', 'specifically',
        'comprehensive', 'significant', 'novel', 'state-of-the-art', 'experimental results',
        'we demonstrate', 'extensive experiments', 'outperforms', 'baseline'
    ]
    
    patterns['ai_pattern_count'] = 0
    patterns['ai_patterns_found'] = []
    for keyword in ai_keywords:
        if keyword in text.lower():
            patterns['ai_pattern_count'] += text.lower().count(keyword)
            patterns['ai_patterns_found'].append(keyword)
    
    # 人类写作可能的特征
    human_keywords = [
        'surprisingly', 'interestingly', 'we find', 'we show', 'turns out',
        'it seems', 'perhaps', 'maybe', 'probably', 'might', 'could be'
    ]
    
    patterns['human_pattern_count'] = 0
    patterns['human_patterns_found'] = []
    for keyword in human_keywords:
        if keyword in text.lower():
            patterns['human_pattern_count'] += text.lower().count(keyword)
            patterns['human_patterns_found'].append(keyword)
    
    # 检查特殊符号
    patterns['question_marks'] = text.count('?')
    patterns['exclamation_marks'] = text.count('!')
    patterns['parentheses'] = text.count('(')
    patterns['quotes'] = text.count('"')
    
    # 检查开头模式
    text_lower = text.strip().lower()
    patterns['starts_with_we'] = text_lower.startswith('we ')
    patterns['starts_with_this'] = text_lower.startswith('this ')
    patterns['starts_with_in'] = text_lower.startswith('in ')
    patterns['starts_with_the'] = text_lower.startswith('the ')
    
    return patterns

def load_and_analyze_errors():
    """加载并分析错误样本"""
    print("=" * 60)
    print("深度错误分析（简化版）")
    print("=" * 60)
    
    # 加载错误分析文件
    if not os.path.exists(ERROR_ANALYSIS_PATH):
        print(f"错误：找不到文件 {ERROR_ANALYSIS_PATH}")
        print("请先运行评估脚本生成 error_analysis.csv")
        return None, None, None
    
    error_df = pd.read_csv(ERROR_ANALYSIS_PATH)
    
    print(f"\n总错误数: {len(error_df)}")
    print(f"错误率: {len(error_df)/1000*100:.1f}% (假设测试集1000个样本)")
    
    # 分类错误类型
    false_positives = error_df[error_df['label'] == 0]  # Human误判为AI
    false_negatives = error_df[error_df['label'] == 1]  # AI误判为Human
    
    print(f"\nFalse Positives (Human→AI): {len(false_positives)}")
    print(f"False Negatives (AI→Human): {len(false_negatives)}")
    
    return error_df, false_positives, false_negatives

def display_detailed_analysis(error_df, false_positives, false_negatives):
    """详细显示每个错误样本的分析"""
    
    print("\n" + "=" * 60)
    print("错误样本详细分析")
    print("=" * 60)
    
    # 保存详细分析到文件
    detailed_analysis_path = os.path.join(OUTPUT_DIR, 'detailed_error_analysis.txt')
    
    with open(detailed_analysis_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("深度错误分析报告\n")
        f.write("="*80 + "\n\n")
        
        # 分析False Positives (Human→AI)
        f.write("\n" + "="*60 + "\n")
        f.write("FALSE POSITIVES (人类文本被误判为AI)\n")
        f.write(f"共 {len(false_positives)} 个样本\n")
        f.write("="*60 + "\n\n")
        
        for i, (idx, row) in enumerate(false_positives.iterrows(), 1):
            f.write(f"\n【样本 {i}】 (原索引: {idx}, 置信度: {row['prediction_confidence']:.3f})\n")
            f.write("-" * 40 + "\n")
            
            text = row['text']
            f.write(f"文本内容:\n{text}\n\n")
            
            # 分析特征
            features = analyze_text_features(text)
            patterns = analyze_linguistic_patterns(text)
            
            f.write("文本特征:\n")
            f.write(f"  - 总长度: {features['length']} 字符\n")
            f.write(f"  - 词数: {features['word_count']}\n")
            f.write(f"  - 句子数: {features['sentence_count']}\n")
            f.write(f"  - 平均句长: {features['avg_sentence_length']:.1f} 词\n")
            f.write(f"  - 独特词汇数: {features['unique_words']}\n")
            f.write(f"  - 词汇多样性: {features['lexical_diversity']:.3f}\n")
            f.write(f"  - 标点符号比例: {features['punctuation_ratio']:.3f}\n")
            f.write(f"  - 大写词比例: {features['capital_ratio']:.3f}\n")
            
            f.write("\n语言模式:\n")
            f.write(f"  - AI模式词汇数: {patterns['ai_pattern_count']}\n")
            if patterns['ai_patterns_found']:
                f.write(f"    发现的AI模式: {', '.join(patterns['ai_patterns_found'])}\n")
            f.write(f"  - 人类模式词汇数: {patterns['human_pattern_count']}\n")
            if patterns['human_patterns_found']:
                f.write(f"    发现的人类模式: {', '.join(patterns['human_patterns_found'])}\n")
            
            f.write("\n可能的误判原因:\n")
            reasons = []
            if patterns['ai_pattern_count'] > 3:
                reasons.append("使用了大量学术/AI写作常见词汇")
            if features['lexical_diversity'] < 0.5:
                reasons.append("词汇多样性较低，文本可能显得公式化")
            if patterns['starts_with_we'] or patterns['starts_with_this']:
                reasons.append("使用了典型的学术论文开头模式")
            if features['avg_sentence_length'] > 25:
                reasons.append("句子较长且结构复杂")
            if not patterns['question_marks'] and not patterns['exclamation_marks']:
                reasons.append("缺乏疑问句或感叹句等人类特征")
            
            for reason in reasons:
                f.write(f"  - {reason}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        # 分析False Negatives (AI→Human)
        f.write("\n" + "="*60 + "\n")
        f.write("FALSE NEGATIVES (AI文本被误判为人类)\n")
        f.write(f"共 {len(false_negatives)} 个样本\n")
        f.write("="*60 + "\n\n")
        
        for i, (idx, row) in enumerate(false_negatives.iterrows(), 1):
            f.write(f"\n【样本 {i}】 (原索引: {idx}, 置信度: {row['prediction_confidence']:.3f})\n")
            f.write("-" * 40 + "\n")
            
            text = row['text']
            f.write(f"文本内容:\n{text}\n\n")
            
            # 分析特征
            features = analyze_text_features(text)
            patterns = analyze_linguistic_patterns(text)
            
            f.write("文本特征:\n")
            f.write(f"  - 总长度: {features['length']} 字符\n")
            f.write(f"  - 词数: {features['word_count']}\n")
            f.write(f"  - 句子数: {features['sentence_count']}\n")
            f.write(f"  - 平均句长: {features['avg_sentence_length']:.1f} 词\n")
            f.write(f"  - 独特词汇数: {features['unique_words']}\n")
            f.write(f"  - 词汇多样性: {features['lexical_diversity']:.3f}\n")
            
            f.write("\n语言模式:\n")
            f.write(f"  - AI模式词汇数: {patterns['ai_pattern_count']}\n")
            f.write(f"  - 人类模式词汇数: {patterns['human_pattern_count']}\n")
            if patterns['human_patterns_found']:
                f.write(f"    发现的人类模式: {', '.join(patterns['human_patterns_found'])}\n")
            f.write(f"  - 疑问句: {patterns['question_marks']} 个\n")
            f.write(f"  - 感叹句: {patterns['exclamation_marks']} 个\n")
            
            f.write("\n可能的误判原因:\n")
            reasons = []
            if patterns['human_pattern_count'] > 2:
                reasons.append("包含了较多人类写作特征词汇")
            if features['lexical_diversity'] > 0.6:
                reasons.append("词汇多样性高，显得更自然")
            if patterns['question_marks'] > 0:
                reasons.append("包含疑问句")
            if features['avg_sentence_length'] < 20:
                reasons.append("句子较短，更像人类的简洁表达")
            if patterns['ai_pattern_count'] < 2:
                reasons.append("缺少典型的AI/学术写作模式")
            
            for reason in reasons:
                f.write(f"  - {reason}\n")
            
            f.write("\n" + "="*60 + "\n")
    
    print(f"\n✓ 详细分析已保存至: {detailed_analysis_path}")
    
    # 打印简要总结到控制台
    print("\n### 错误样本简要预览 ###\n")
    
    if len(false_positives) > 0:
        print("FALSE POSITIVES (Human→AI):")
        for i, (idx, row) in enumerate(false_positives.head(3).iterrows(), 1):
            print(f"\n{i}. 置信度: {row['prediction_confidence']:.3f}")
            print(f"   文本预览: {row['text'][:200]}...")
    
    if len(false_negatives) > 0:
        print("\nFALSE NEGATIVES (AI→Human):")
        for i, (idx, row) in enumerate(false_negatives.head(3).iterrows(), 1):
            print(f"\n{i}. 置信度: {row['prediction_confidence']:.3f}")
            print(f"   文本预览: {row['text'][:200]}...")

def create_feature_comparison_chart(error_df):
    """创建特征对比图表（论文级别）"""
    
    print("\n正在生成特征对比图表...")
    
    # 设置matplotlib为论文发表质量
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42  # TrueType字体
    mpl.rcParams['ps.fonttype'] = 42   # TrueType字体
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    
    # 收集特征
    all_features = []
    for idx, row in error_df.iterrows():
        features = analyze_text_features(row['text'])
        features['error_type'] = 'FP' if row['label'] == 0 else 'FN'
        features['confidence'] = row['prediction_confidence']
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 子图标签样式
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    label_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    
    # 1. 文本长度对比 - 子图(a)
    ax = axes[0, 0]
    fp_data = features_df[features_df['error_type'] == 'FP']['length']
    fn_data = features_df[features_df['error_type'] == 'FN']['length']
    
    if len(fp_data) > 0 and len(fn_data) > 0:
        bp = ax.boxplot([fp_data, fn_data], 
                        labels=['False Positive\n(Human→AI)', 'False Negative\n(AI→Human)'],
                        patch_artist=True)
        # 设置箱体颜色
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
    elif len(fp_data) > 0:
        bp = ax.boxplot([fp_data], labels=['False Positive\n(Human→AI)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
    elif len(fn_data) > 0:
        bp = ax.boxplot([fn_data], labels=['False Negative\n(AI→Human)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
    
    ax.set_ylabel('Text Length (characters)', fontweight='bold')
    ax.set_title('Text Length Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, subplot_labels[0], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top', bbox=label_props)
    
    # 2. 词汇多样性对比 - 子图(b)
    ax = axes[0, 1]
    fp_diversity = features_df[features_df['error_type'] == 'FP']['lexical_diversity']
    fn_diversity = features_df[features_df['error_type'] == 'FN']['lexical_diversity']
    
    if len(fp_diversity) > 0 and len(fn_diversity) > 0:
        bp = ax.boxplot([fp_diversity, fn_diversity], 
                        labels=['False Positive\n(Human→AI)', 'False Negative\n(AI→Human)'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
    elif len(fp_diversity) > 0:
        bp = ax.boxplot([fp_diversity], labels=['False Positive\n(Human→AI)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
    elif len(fn_diversity) > 0:
        bp = ax.boxplot([fn_diversity], labels=['False Negative\n(AI→Human)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
    
    ax.set_ylabel('Lexical Diversity', fontweight='bold')
    ax.set_title('Vocabulary Diversity Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, subplot_labels[1], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top', bbox=label_props)
    
    # 3. 平均句长对比 - 子图(c)
    ax = axes[1, 0]
    fp_sent = features_df[features_df['error_type'] == 'FP']['avg_sentence_length']
    fn_sent = features_df[features_df['error_type'] == 'FN']['avg_sentence_length']
    
    if len(fp_sent) > 0 and len(fn_sent) > 0:
        bp = ax.boxplot([fp_sent, fn_sent], 
                        labels=['False Positive\n(Human→AI)', 'False Negative\n(AI→Human)'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
    elif len(fp_sent) > 0:
        bp = ax.boxplot([fp_sent], labels=['False Positive\n(Human→AI)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
    elif len(fn_sent) > 0:
        bp = ax.boxplot([fn_sent], labels=['False Negative\n(AI→Human)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
    
    ax.set_ylabel('Average Sentence Length (words)', fontweight='bold')
    ax.set_title('Sentence Length Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, subplot_labels[2], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top', bbox=label_props)
    
    # 4. 置信度分布 - 子图(d)
    ax = axes[1, 1]
    fp_conf = features_df[features_df['error_type'] == 'FP']['confidence']
    fn_conf = features_df[features_df['error_type'] == 'FN']['confidence']
    
    if len(fp_conf) > 0:
        n1, bins1, patches1 = ax.hist(fp_conf, alpha=0.6, label='False Positive', 
                                      color='red', bins=10, edgecolor='black', linewidth=1)
    if len(fn_conf) > 0:
        n2, bins2, patches2 = ax.hist(fn_conf, alpha=0.6, label='False Negative', 
                                      color='blue', bins=10, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Prediction Confidence', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Confidence Distribution of Errors', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.98, subplot_labels[3], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top', bbox=label_props)
    
    # 设置总标题
    plt.suptitle('Error Sample Feature Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局以适应总标题
    
    # 保存为PDF矢量图（论文用）
    pdf_path = os.path.join(OUTPUT_DIR, 'error_features_comparison.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ 特征对比图(PDF)已保存至: {pdf_path}")
    
    # 同时保存PNG预览
    png_path = os.path.join(OUTPUT_DIR, 'error_features_comparison.png')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✓ 特征对比图(PNG)已保存至: {png_path}")
    
    plt.close()
    
    # 打印统计摘要
    print("\n### 特征统计摘要 ###")
    
    if len(features_df[features_df['error_type'] == 'FP']) > 0:
        fp_stats = features_df[features_df['error_type'] == 'FP']
        print("\nFalse Positives (Human→AI):")
        print(f"  平均长度: {fp_stats['length'].mean():.0f} 字符 (±{fp_stats['length'].std():.0f})")
        print(f"  平均词数: {fp_stats['word_count'].mean():.0f} (±{fp_stats['word_count'].std():.0f})")
        print(f"  平均词汇多样性: {fp_stats['lexical_diversity'].mean():.3f} (±{fp_stats['lexical_diversity'].std():.3f})")
        print(f"  平均句长: {fp_stats['avg_sentence_length'].mean():.1f} 词 (±{fp_stats['avg_sentence_length'].std():.1f})")
        print(f"  平均预测置信度: {fp_stats['confidence'].mean():.3f}")
    
    if len(features_df[features_df['error_type'] == 'FN']) > 0:
        fn_stats = features_df[features_df['error_type'] == 'FN']
        print("\nFalse Negatives (AI→Human):")
        print(f"  平均长度: {fn_stats['length'].mean():.0f} 字符 (±{fn_stats['length'].std():.0f})")
        print(f"  平均词数: {fn_stats['word_count'].mean():.0f} (±{fn_stats['word_count'].std():.0f})")
        print(f"  平均词汇多样性: {fn_stats['lexical_diversity'].mean():.3f} (±{fn_stats['lexical_diversity'].std():.3f})")
        print(f"  平均句长: {fn_stats['avg_sentence_length'].mean():.1f} 词 (±{fn_stats['avg_sentence_length'].std():.1f})")
        print(f"  平均预测置信度: {fn_stats['confidence'].mean():.3f}")

def generate_insights_report(error_df, false_positives, false_negatives):
    """生成洞察报告"""
    
    print("\n" + "=" * 60)
    print("关键洞察与改进建议")
    print("=" * 60)
    
    insights = []
    
    # 分析共同特征
    if len(false_positives) > 0:
        fp_features = [analyze_text_features(text) for text in false_positives['text']]
        fp_patterns = [analyze_linguistic_patterns(text) for text in false_positives['text']]
        
        avg_length = np.mean([f['length'] for f in fp_features])
        avg_diversity = np.mean([f['lexical_diversity'] for f in fp_features])
        total_ai_patterns = sum([p['ai_pattern_count'] for p in fp_patterns])
        
        insights.append(f"【False Positives 分析】")
        insights.append(f"  • 平均文本长度: {avg_length:.0f} 字符")
        insights.append(f"  • 平均词汇多样性: {avg_diversity:.3f}")
        insights.append(f"  • 发现 {total_ai_patterns} 个AI写作模式")
        
        if avg_length > 1500:
            insights.append("  • 倾向于较长的文本")
        if avg_diversity < 0.5:
            insights.append("  • 词汇重复率高，写作风格规范化")
    
    if len(false_negatives) > 0:
        fn_features = [analyze_text_features(text) for text in false_negatives['text']]
        fn_patterns = [analyze_linguistic_patterns(text) for text in false_negatives['text']]
        
        avg_length = np.mean([f['length'] for f in fn_features])
        avg_diversity = np.mean([f['lexical_diversity'] for f in fn_features])
        total_human_patterns = sum([p['human_pattern_count'] for p in fn_patterns])
        
        insights.append(f"\n【False Negatives 分析】")
        insights.append(f"  • 平均文本长度: {avg_length:.0f} 字符")
        insights.append(f"  • 平均词汇多样性: {avg_diversity:.3f}")
        insights.append(f"  • 发现 {total_human_patterns} 个人类写作模式")
        
        if avg_diversity > 0.6:
            insights.append("  • 词汇多样性高，更像自然语言")
    
    # 打印洞察
    for insight in insights:
        print(insight)
    
    print("\n### 改进建议 ###")
    suggestions = [
        "1. 数据增强：增加边界案例的训练样本",
        "2. 特征工程：考虑加入文本长度、词汇多样性作为辅助特征",
        "3. 模型改进：使用更大的预训练模型或集成学习",
        "4. 后处理：对低置信度预测进行二次验证",
        "5. 领域适应：针对特定写作风格进行微调"
    ]
    
    for suggestion in suggestions:
        print(suggestion)
    
    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, 'insights_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("错误分析洞察报告\n")
        f.write("=" * 40 + "\n\n")
        
        for insight in insights:
            f.write(insight + "\n")
        
        f.write("\n改进建议:\n")
        for suggestion in suggestions:
            f.write(suggestion + "\n")
    
    print(f"\n✓ 洞察报告已保存至: {report_path}")

def main():
    """主函数"""
    # 加载错误样本
    error_df, false_positives, false_negatives = load_and_analyze_errors()
    
    if error_df is None:
        return
    
    # 详细分析每个错误样本
    display_detailed_analysis(error_df, false_positives, false_negatives)
    
    # 创建特征对比图表
    create_feature_comparison_chart(error_df)
    
    # 生成洞察报告
    generate_insights_report(error_df, false_positives, false_negatives)
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)
    print(f"\n所有分析结果已保存到: {OUTPUT_DIR}")
    print("\n生成的文件:")
    print("  📄 detailed_error_analysis.txt - 每个错误样本的详细分析")
    print("  📊 error_features_comparison.pdf - 特征对比图表(矢量图)")
    print("  📊 error_features_comparison.png - 特征对比图表(预览)")
    print("  📝 insights_report.txt - 洞察与改进建议")

if __name__ == '__main__':
    main()