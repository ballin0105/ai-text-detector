#!/usr/bin/env python3
"""
脚本：检查两个CSV文件的条目数量
用于验证 ai_abstracts.csv 和 human_abstracts.csv 是否都包含2500条记录
"""

import csv
import os
import sys

def count_csv_rows(filepath):
    """
    统计CSV文件的行数（不包括表头）
    
    Args:
        filepath: CSV文件路径
    
    Returns:
        int: 数据行数（不包括表头）
        None: 如果文件读取失败
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            # 跳过表头
            next(csv_reader, None)
            # 计算数据行数
            row_count = sum(1 for row in csv_reader)
            return row_count
    except FileNotFoundError:
        print(f"错误：文件 {filepath} 不存在")
        return None
    except Exception as e:
        print(f"读取文件 {filepath} 时出错：{e}")
        return None

def main():
    # 定义文件路径
    ai_file = "/hy-tmp/Detector/ai_abstracts.csv"
    human_file = "/hy-tmp/Detector/human_abstracts.csv"
    
    print("=" * 60)
    print("CSV文件条目数量检查")
    print("=" * 60)
    
    # 检查AI abstracts文件
    print(f"\n检查文件: {ai_file}")
    ai_count = count_csv_rows(ai_file)
    if ai_count is not None:
        print(f"  条目数量: {ai_count}")
        if ai_count == 2500:
            print(f"  ✓ 符合预期（2500条）")
        else:
            print(f"  ✗ 不符合预期（期望2500条，实际{ai_count}条）")
    
    # 检查Human abstracts文件
    print(f"\n检查文件: {human_file}")
    human_count = count_csv_rows(human_file)
    if human_count is not None:
        print(f"  条目数量: {human_count}")
        if human_count == 2500:
            print(f"  ✓ 符合预期（2500条）")
        else:
            print(f"  ✗ 不符合预期（期望2500条，实际{human_count}条）")
    
    # 总结
    print("\n" + "=" * 60)
    print("检查总结:")
    print("=" * 60)
    
    if ai_count is not None and human_count is not None:
        if ai_count == 2500 and human_count == 2500:
            print("✓ 两个文件都包含2500条记录")
            sys.exit(0)
        else:
            print("✗ 文件条目数量不符合预期:")
            if ai_count != 2500:
                print(f"  - ai_abstracts.csv: {ai_count}条（期望2500条）")
            if human_count != 2500:
                print(f"  - human_abstracts.csv: {human_count}条（期望2500条）")
            sys.exit(1)
    else:
        print("✗ 无法完成检查，请查看上述错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()