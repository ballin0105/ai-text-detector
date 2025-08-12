import requests
import json
import pandas as pd

# ===== 配置 =====
API_KEY = "sk-CkeD2M8VvmLtzBClduppBPw1ySKm51PefqLnKsOo4wg5g0zS"  # <-- 替换为您的API密钥
BASE_URL = "https://www.chataiapi.com"
MODEL = "gemini-2.0-flash"

print("="*60)
print("摘要生成测试")
print(f"模型: {MODEL}")
print("="*60)

# API配置
url = BASE_URL + "/v1/chat/completions"
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# 加载前几篇论文进行测试
df = pd.read_csv("human_abstracts.csv")
print(f"加载了 {len(df)} 篇论文，将测试前3篇\n")

# 测试提示模板
PROMPT = """You are an expert academic researcher in Computer Vision.
Write a professional scientific abstract (150-250 words) for a paper titled: "{title}".

The abstract should include:
1. Background/Context (1-2 sentences)
2. Problem addressed (1 sentence)
3. Proposed method (2-3 sentences)
4. Key results (1-2 sentences)
5. Significance (1 sentence)

Write only the abstract text, no introductions or labels."""

# 测试前3篇
for i in range(min(3, len(df))):
    row = df.iloc[i]
    print(f"\n{'='*60}")
    print(f"测试 {i+1}/3")
    print(f"论文ID: {row['id']}")
    print(f"标题: {row['title']}")
    print("-"*60)
    
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
        
        # 显示请求内容
        print(f"请求内容预览:")
        print(f"User prompt: {PROMPT.format(title=row['title'])[:100]}...")
        
        # 发送请求
        print(f"\n发送请求...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"响应状态: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # 显示响应结构
            print(f"响应keys: {list(data.keys())}")
            
            if 'choices' in data and len(data['choices']) > 0:
                abstract = data['choices'][0]['message']['content'].strip()
                word_count = len(abstract.split())
                
                print(f"\n生成的摘要 ({word_count} 词):")
                print("-"*60)
                print(abstract)
                print("-"*60)
                
                # 显示token使用情况
                if 'usage' in data:
                    usage = data['usage']
                    print(f"\nToken使用:")
                    print(f"- Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(f"- Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                    print(f"- Total tokens: {usage.get('total_tokens', 'N/A')}")
            else:
                print(f"\n错误: 响应中没有choices")
                print(f"完整响应: {json.dumps(data, indent=2)}")
        else:
            print(f"\nAPI错误: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"\n发生异常: {e}")
        import traceback
        traceback.print_exc()

# 估算成本
print(f"\n{'='*60}")
print("成本估算:")
print("- 每篇论文约使用 200-400 tokens")
print("- 2500篇预计使用 50-100万 tokens")
print(f"- Gemini Flash 价格约 $0.075/1M tokens")
print(f"- 预计总成本: ${0.075 * 0.5:.2f} - ${0.075 * 1:.2f}")
print("="*60)