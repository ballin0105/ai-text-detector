#!/usr/bin/env python3
"""
准备最终训练数据集
将AI生成的摘要和人类撰写的摘要合并成一个带标签的训练数据集
"""

import pandas as pd
import os
import sys

# --- 配置参数 ---
HUMAN_DATA_PATH = '/hy-tmp/Detector/data/raw/human_abstracts.csv'
AI_DATA_PATH = '/hy-tmp/Detector/data/raw/ai_abstracts.csv'

OUTPUT_DIR = '/hy-tmp/Detector'
FINAL_DATASET_PATH = os.path.join(OUTPUT_DIR, '../data/processed/final_labeled_dataset.csv')

def check_files_exist():
    """检查输入文件是否存在"""
    missing_files = []
    
    if not os.path.exists(HUMAN_DATA_PATH):
        missing_files.append(HUMAN_DATA_PATH)
    if not os.path.exists(AI_DATA_PATH):
        missing_files.append(AI_DATA_PATH)
    
    if missing_files:
        print("错误: 缺少以下文件:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    return True

def prepare_dataset():
    """
    合并人类和AI生成的摘要，添加标签，并创建一个用于训练的最终数据集。
    
    处理步骤:
    1. 读取两个CSV文件
    2. 为数据添加标签（人类=0, AI=1）
    3. 合并数据集
    4. 清洗数据（移除空行和失败的生成）
    5. 打乱数据顺序
    6. 保存最终数据集
    """
    
    print("=" * 60)
    print("开始准备最终训练数据集")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not check_files_exist():
        print("\n请确保两个CSV文件都已准备好。")
        sys.exit(1)
    
    print("\n正在加载数据集...")
    print("-" * 40)
    
    try:
        # 加载数据集
        df_human = pd.read_csv(HUMAN_DATA_PATH, encoding='utf-8')
        df_ai = pd.read_csv(AI_DATA_PATH, encoding='utf-8')
        
        print(f"✓ 成功加载人类摘要: {len(df_human)} 条")
        print(f"✓ 成功加载AI摘要: {len(df_ai)} 条")
        
    except Exception as e:
        print(f"错误: 读取CSV文件时出错 - {e}")
        sys.exit(1)
    
    # --- 添加标签 ---
    print("\n添加标签...")
    print("-" * 40)
    # 人类撰写: label = 0
    # AI生成:   label = 1
    df_human['label'] = 0
    df_ai['label'] = 1
    print("✓ 人类摘要标记为: 0")
    print("✓ AI摘要标记为: 1")
    
    # --- 合并数据集 ---
    print("\n合并数据集...")
    print("-" * 40)
    df_combined = pd.concat([df_human, df_ai], ignore_index=True)
    print(f"✓ 合并完成，总共 {len(df_combined)} 条数据")
    
    # --- 数据清洗 ---
    print("\n清洗数据...")
    print("-" * 40)
    initial_count = len(df_combined)
    
    # 检查是否有'abstract'列，如果没有则检查其他可能的列名
    if 'abstract' in df_combined.columns:
        abstract_col = 'abstract'
    elif 'text' in df_combined.columns:
        abstract_col = 'text'
    elif 'content' in df_combined.columns:
        abstract_col = 'content'
    else:
        # 如果找不到明确的文本列，使用第一个非标签列
        cols = [col for col in df_combined.columns if col != 'label']
        if cols:
            abstract_col = cols[0]
            print(f"  使用列 '{abstract_col}' 作为文本内容列")
        else:
            print("错误: 无法找到文本内容列")
            sys.exit(1)
    
    # 移除内容为空的行
    df_combined.dropna(subset=[abstract_col], inplace=True)
    removed_na = initial_count - len(df_combined)
    if removed_na > 0:
        print(f"  移除了 {removed_na} 行空数据")
    
    # 移除在AI生成过程中可能标记为失败的行
    if 'GENERATION_FAILED' in df_combined[abstract_col].values:
        before_remove = len(df_combined)
        df_combined = df_combined[df_combined[abstract_col] != 'GENERATION_FAILED']
        removed_failed = before_remove - len(df_combined)
        if removed_failed > 0:
            print(f"  移除了 {removed_failed} 行生成失败的数据")
    
    # 移除空字符串
    before_remove = len(df_combined)
    df_combined = df_combined[df_combined[abstract_col].str.strip() != '']
    removed_empty = before_remove - len(df_combined)
    if removed_empty > 0:
        print(f"  移除了 {removed_empty} 行空字符串")
    
    print(f"✓ 清洗完成，剩余 {len(df_combined)} 条有效数据")
    
    # --- 打乱数据顺序 ---
    print("\n打乱数据顺序...")
    print("-" * 40)
    # 使用 frac=1 表示100%抽样，相当于完全打乱
    # random_state=42 确保每次打乱的结果都一样，保证实验可复现
    df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    print("✓ 数据已随机打乱（random_state=42）")
    
    # --- 数据统计 ---
    print("\n数据集统计:")
    print("-" * 40)
    label_counts = df_shuffled['label'].value_counts().sort_index()
    print(f"  人类摘要 (label=0): {label_counts.get(0, 0)} 条")
    print(f"  AI摘要 (label=1): {label_counts.get(1, 0)} 条")
    print(f"  总计: {len(df_shuffled)} 条")
    
    # 计算标签分布比例
    if len(df_shuffled) > 0:
        human_ratio = label_counts.get(0, 0) / len(df_shuffled) * 100
        ai_ratio = label_counts.get(1, 0) / len(df_shuffled) * 100
        print(f"  比例: 人类 {human_ratio:.1f}% / AI {ai_ratio:.1f}%")
    
    # --- 保存最终数据集 ---
    print("\n保存最终数据集...")
    print("-" * 40)
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 保存数据
        df_shuffled.to_csv(FINAL_DATASET_PATH, index=False, encoding='utf-8')
        print(f"✓ 成功保存到: {FINAL_DATASET_PATH}")
        
        # 显示文件大小
        file_size = os.path.getsize(FINAL_DATASET_PATH) / (1024 * 1024)  # 转换为MB
        print(f"  文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"错误: 保存文件时出错 - {e}")
        sys.exit(1)
    
    # --- 完成 ---
    print("\n" + "=" * 60)
    print("✅ 任务完成！")
    print("=" * 60)
    print(f"\n最终的训练数据集已准备就绪:")
    print(f"  📁 {FINAL_DATASET_PATH}")
    print(f"  📊 包含 {len(df_shuffled)} 条数据")
    print("\n现在您可以使用这个数据集进行模型训练了！")
    print("=" * 60)
    
    return df_shuffled

def display_sample(df, n=5):
    """显示数据集的样本"""
    print(f"\n数据集前 {n} 条样本预览:")
    print("-" * 60)
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        # 获取文本列名
        text_cols = [col for col in df.columns if col != 'label']
        if text_cols:
            text_col = text_cols[0]
            text_preview = str(row[text_col])[:100] + "..." if len(str(row[text_col])) > 100 else str(row[text_col])
            print(f"样本 {i+1}:")
            print(f"  标签: {row['label']} ({'人类' if row['label'] == 0 else 'AI'})")
            print(f"  文本: {text_preview}")
            print()

if __name__ == '__main__':
    # 执行数据集准备
    final_df = prepare_dataset()
    
    # 可选：显示一些样本
    if input("\n是否显示数据集样本？(y/n): ").lower() == 'y':
        display_sample(final_df, n=5)