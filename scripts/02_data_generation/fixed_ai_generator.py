import requests
import json
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import os

# ===== 配置 =====
API_KEY = "sk-CkeD2M8VvmLtzBClduppBPw1ySKm51PefqLnKsOo4wg5g0zS"  # <-- 替换为您的API密钥
BASE_URL = "https://www.chataiapi.com"
MODEL = "gemini-2.0-flash"  # 或其他模型

# 文件名
INPUT_FILE = "human_abstracts.csv"
OUTPUT_FILE = "ai_abstracts.csv"
CHECKPOINT_FILE = "generation_checkpoint.json"

print("="*60)
print("AI 摘要生成器")
print(f"模型: {MODEL}")
print("="*60)

# 检查输入文件
if not os.path.exists(INPUT_FILE):
    print(f"错误: 找不到文件 {INPUT_FILE}")
    exit(1)

# 加载数据
df = pd.read_csv(INPUT_FILE)
print(f"✓ 成功加载 {len(df)} 篇论文")

# API配置
url = BASE_URL + "/v1/chat/completions"
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# 提示模板
PROMPT = """You are an expert academic researcher in Computer Vision.
Write a professional scientific abstract (150-250 words) for a paper titled: "{title}".

The abstract should include:
1. Background/Context (1-2 sentences)
2. Problem addressed (1 sentence)
3. Proposed method (2-3 sentences)
4. Key results (1-2 sentences)
5. Significance (1 sentence)

Write only the abstract text, no introductions or labels."""

# 加载检查点（断点续传）
checkpoint = {'processed': 0, 'results': []}
if os.path.exists(CHECKPOINT_FILE):
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        print(f"从检查点恢复，已处理 {checkpoint['processed']} 篇")
    except:
        pass

results = checkpoint['results']
start_idx = checkpoint['processed']

# 统计
success_count = len([r for r in results if r['abstract'] != 'GENERATION_FAILED'])
fail_count = len([r for r in results if r['abstract'] == 'GENERATION_FAILED'])

print(f"\n开始生成 AI 摘要...")
print(f"将从第 {start_idx + 1} 篇开始\n")

# 使用tqdm显示进度
for idx in tqdm(range(start_idx, len(df)), initial=start_idx, total=len(df), desc="生成进度"):
    row = df.iloc[idx]
    
    try:
        # 构造请求
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert academic researcher in Computer Vision."
                },
                {
                    "role": "user", 
                    "content": PROMPT.format(title=row['title'])
                }
            ],
            "max_tokens": 400,
            "temperature": 0.7
        }
        
        # 发送请求
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # 提取摘要内容
            if 'choices' in data and len(data['choices']) > 0:
                abstract = data['choices'][0]['message']['content'].strip()
                
                # 验证摘要长度
                word_count = len(abstract.split())
                if word_count < 50:  # 太短
                    print(f"\n警告: 摘要太短 ({word_count} 词)")
                    abstract = 'GENERATION_FAILED'
                    fail_count += 1
                else:
                    success_count += 1
            else:
                print(f"\n警告: 响应格式异常")
                abstract = 'GENERATION_FAILED'
                fail_count += 1
                
        else:
            print(f"\n错误: API返回 {response.status_code}")
            abstract = 'GENERATION_FAILED'
            fail_count += 1
            
    except Exception as e:
        print(f"\n错误: {str(e)[:100]}")
        abstract = 'GENERATION_FAILED'
        fail_count += 1
    
    # 保存结果
    results.append({
        'id': row['id'],
        'title': row['title'],
        'abstract': abstract
    })
    
    # 更新检查点（每10篇）
    if (idx + 1) % 10 == 0:
        checkpoint = {
            'processed': idx + 1,
            'results': results
        }
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f)
    
    # 每100篇显示统计
    if (idx + 1) % 100 == 0:
        print(f"\n进度: {idx + 1}/{len(df)} | 成功: {success_count} | 失败: {fail_count}")
    
    # API速率限制
    time.sleep(1)

# 保存最终结果
df_output = pd.DataFrame(results)
df_output.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

# 删除检查点
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

# 统计信息
print("\n" + "="*60)
print("✓ 生成完成！")
print(f"总计: {len(df)} 篇")
print(f"成功: {success_count} 篇")
print(f"失败: {fail_count} 篇")
print(f"输出文件: {OUTPUT_FILE}")

# 计算有效摘要的统计
valid_df = df_output[df_output['abstract'] != 'GENERATION_FAILED']
if len(valid_df) > 0:
    lengths = valid_df['abstract'].apply(lambda x: len(x.split()))
    print(f"\n摘要统计:")
    print(f"- 平均长度: {lengths.mean():.1f} 词")
    print(f"- 最短: {lengths.min()} 词")
    print(f"- 最长: {lengths.max()} 词")

print("="*60)