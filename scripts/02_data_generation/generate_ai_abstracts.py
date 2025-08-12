import google.generativeai as genai
import pandas as pd
import os
import time
from tqdm import tqdm
import json
from datetime import datetime
import argparse

# --- 配置参数 ---

# 输入和输出文件路径
INPUT_FILENAME = '../data/raw/human_abstracts.csv'
OUTPUT_FILENAME = '../data/raw/ai_abstracts.csv'
CHECKPOINT_FILE = 'generation_checkpoint.json'

# 使用的模型
MODEL_NAME = 'gemini-2.5-pro'  # 或 'gemini-1.5-pro' 如果需要更高质量

# 指令模板
PROMPT_TEMPLATE = """You are an expert academic researcher in the field of Computer Vision.
Write a professional, high-quality scientific abstract for a research paper titled: "{title}".

The abstract should be:
- Clear, concise, and well-structured
- Approximately 150-250 words
- Written in a formal, academic tone
- Include: context/motivation, approach/methodology, key findings/contributions, and significance
- Use present tense for general truths and past tense for specific experiments

Do NOT add any introductory phrases like 'Here is an abstract...' or any concluding remarks. Just provide the abstract text itself."""

# 备选的更详细的提示模板
DETAILED_PROMPT_TEMPLATE = """You are an expert academic researcher in Computer Vision. Generate a high-quality scientific abstract for a paper titled: "{title}".

Structure the abstract to include:
1. Background/Context (1-2 sentences)
2. Problem/Gap addressed (1 sentence)
3. Proposed method/approach (2-3 sentences)
4. Key experiments/results (1-2 sentences)
5. Main contributions/significance (1 sentence)

Style requirements:
- Use technical terminology appropriately
- Be specific about methods and results
- Avoid vague statements
- Total length: 150-250 words

Generate only the abstract text without any meta-commentary."""

def load_checkpoint():
    """加载检查点，用于断点续传"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed_indices': [], 'ai_abstracts': []}

def save_checkpoint(checkpoint):
    """保存检查点"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def validate_abstract(abstract):
    """验证生成的摘要是否符合要求"""
    if not abstract or abstract == 'GENERATION_FAILED':
        return False
    
    # 检查长度
    word_count = len(abstract.split())
    if word_count < 100 or word_count > 350:
        return False
    
    # 检查是否包含不需要的前缀
    unwanted_prefixes = ['Here is', 'Below is', 'The abstract is', 'Abstract:']
    for prefix in unwanted_prefixes:
        if abstract.startswith(prefix):
            return False
    
    return True

def generate_ai_abstracts(use_detailed_prompt=False, resume=True, max_retries=3):
    """
    读取包含论文标题的CSV文件，使用Gemini API为每个标题生成新的摘要
    
    参数:
    - use_detailed_prompt: 是否使用更详细的提示模板
    - resume: 是否从上次中断的地方继续
    - max_retries: 每个标题的最大重试次数
    """
    
    # 检查API密钥
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("错误: 请先设置名为 'GEMINI_API_KEY' 的环境变量")
        print("Linux/Mac: export GEMINI_API_KEY='your-key-here'")
        print("Windows: set GEMINI_API_KEY=your-key-here")
        return
    
    genai.configure(api_key=api_key)
    
    # 检查输入文件
    if not os.path.exists(INPUT_FILENAME):
        print(f"错误: 输入文件 '{INPUT_FILENAME}' 未找到")
        return
    
    # 加载数据
    df_human = pd.read_csv(INPUT_FILENAME)
    print(f"加载了 {len(df_human)} 篇论文")
    
    # 加载检查点（如果启用断点续传）
    checkpoint = load_checkpoint() if resume else {'processed_indices': [], 'ai_abstracts': []}
    processed_indices = set(checkpoint['processed_indices'])
    ai_abstracts = checkpoint['ai_abstracts']
    
    # 初始化模型
    model = genai.GenerativeModel(MODEL_NAME)
    
    # 选择提示模板
    prompt_template = DETAILED_PROMPT_TEMPLATE if use_detailed_prompt else PROMPT_TEMPLATE
    
    # 统计信息
    success_count = len(ai_abstracts)
    fail_count = 0
    
    print(f"\n开始生成AI摘要...")
    if processed_indices:
        print(f"从检查点恢复，已处理 {len(processed_indices)} 条")
    
    # 使用tqdm显示进度
    for idx, row in tqdm(df_human.iterrows(), total=len(df_human), desc="生成进度"):
        # 跳过已处理的
        if idx in processed_indices:
            continue
        
        title = row['title']
        paper_id = row['id']
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                prompt = prompt_template.format(title=title)
                
                # 调用API
                response = model.generate_content(prompt)
                ai_abstract = response.text.strip()
                
                # 验证生成的摘要
                if validate_abstract(ai_abstract):
                    ai_abstracts.append({
                        'id': paper_id,
                        'title': title,
                        'abstract': ai_abstract,
                        'word_count': len(ai_abstract.split()),
                        'generated_at': datetime.now().isoformat()
                    })
                    success_count += 1
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"\n摘要验证失败，重试 {attempt + 1}/{max_retries}...")
                        time.sleep(2)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n错误: {e}，重试 {attempt + 1}/{max_retries}...")
                    time.sleep(5)  # 错误后等待更长时间
                else:
                    print(f"\n为标题 '{title[:50]}...' 生成失败")
                    ai_abstracts.append({
                        'id': paper_id,
                        'title': title,
                        'abstract': 'GENERATION_FAILED',
                        'word_count': 0,
                        'generated_at': datetime.now().isoformat()
                    })
                    fail_count += 1
        
        # 更新检查点
        processed_indices.add(idx)
        checkpoint['processed_indices'] = list(processed_indices)
        checkpoint['ai_abstracts'] = ai_abstracts
        
        # 每10条保存一次检查点
        if len(processed_indices) % 10 == 0:
            save_checkpoint(checkpoint)
        
        # API速率限制
        time.sleep(1)
    
    # 创建最终的DataFrame
    df_ai = pd.DataFrame(ai_abstracts)
    
    # 确保所有论文都有对应的生成结果
    if len(df_ai) < len(df_human):
        print(f"\n警告: 只生成了 {len(df_ai)} 个摘要，原始数据有 {len(df_human)} 条")
    
    # 保存结果
    df_ai.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')
    
    # 删除检查点文件
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    # 统计信息
    print("\n" + "="*60)
    print("AI摘要生成完成！")
    print(f"总计处理: {len(df_ai)} 篇论文")
    print(f"成功生成: {success_count} 篇")
    print(f"生成失败: {fail_count} 篇")
    print(f"结果保存到: {OUTPUT_FILENAME}")
    
    if fail_count > 0:
        print(f"\n失败的条目已标记为 'GENERATION_FAILED'")
        failed_df = df_ai[df_ai['abstract'] == 'GENERATION_FAILED']
        print(f"失败的论文ID: {failed_df['id'].tolist()[:5]}...")  # 显示前5个
    
    # 生成质量报告
    if success_count > 0:
        valid_abstracts = df_ai[df_ai['abstract'] != 'GENERATION_FAILED']
        avg_length = valid_abstracts['word_count'].mean()
        print(f"\n生成质量统计:")
        print(f"平均词数: {avg_length:.1f}")
        print(f"最短摘要: {valid_abstracts['word_count'].min()} 词")
        print(f"最长摘要: {valid_abstracts['word_count'].max()} 词")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='使用Gemini生成AI论文摘要')
    parser.add_argument('--detailed', action='store_true', help='使用更详细的提示模板')
    parser.add_argument('--no-resume', action='store_true', help='不使用断点续传')
    parser.add_argument('--retries', type=int, default=3, help='每个标题的最大重试次数')
    
    args = parser.parse_args()
    
    generate_ai_abstracts(
        use_detailed_prompt=args.detailed,
        resume=not args.no_resume,
        max_retries=args.retries
    )

if __name__ == '__main__':
    main()