#!/bin/bash
# 模型下载辅助脚本
# 用于从镜像源下载 DistilBERT 模型

echo "========================================"
echo "    DistilBERT 模型下载脚本"
echo "========================================"

# 设置模型名称和本地目录
MODEL_NAME="distilbert-base-uncased"
LOCAL_DIR="./model_cache/${MODEL_NAME}"

# 创建目录
mkdir -p ${LOCAL_DIR}

echo ""
echo "方案1: 使用 huggingface-cli (推荐)"
echo "----------------------------------------"
echo "正在安装/更新 huggingface-cli..."
pip install -U huggingface_hub -i https://pypi.org/simple

# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

echo "正在从镜像源下载模型..."
huggingface-cli download ${MODEL_NAME} \
    --local-dir ${LOCAL_DIR} \
    --local-dir-use-symlinks False \
    --resume-download

if [ $? -eq 0 ]; then
    echo "✓ 模型下载成功！"
    echo "模型保存在: ${LOCAL_DIR}"
else
    echo ""
    echo "方案2: 使用 wget 直接下载"
    echo "----------------------------------------"
    echo "正在尝试使用 wget 下载必要文件..."
    
    # 基础URL
    BASE_URL="https://hf-mirror.com/${MODEL_NAME}/resolve/main"
    
    # 需要下载的文件列表
    FILES=(
        "config.json"
        "pytorch_model.bin"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.txt"
    )
    
    # 下载每个文件
    for file in "${FILES[@]}"; do
        echo "下载: $file"
        wget -c "${BASE_URL}/${file}" -O "${LOCAL_DIR}/${file}"
    done
    
    echo ""
    echo "方案3: 使用 Python 脚本下载"
    echo "----------------------------------------"
    cat << 'EOF' > download_model.py
import os
import requests
from tqdm import tqdm

def download_file(url, dest):
    """下载文件with进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))

# 模型文件
model_name = "distilbert-base-uncased"
local_dir = f"./model_cache/{model_name}"
os.makedirs(local_dir, exist_ok=True)

# 镜像源URL
base_url = f"https://hf-mirror.com/{model_name}/resolve/main"

# 需要下载的文件
files = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json", 
    "tokenizer.json",
    "vocab.txt"
]

print(f"开始下载 {model_name} 模型...")
for file_name in files:
    url = f"{base_url}/{file_name}"
    dest = os.path.join(local_dir, file_name)
    print(f"\n下载: {file_name}")
    try:
        download_file(url, dest)
        print(f"✓ {file_name} 下载成功")
    except Exception as e:
        print(f"✗ {file_name} 下载失败: {e}")

print("\n下载完成！")
EOF
    
    python download_model.py
fi

echo ""
echo "========================================"
echo "下载完成后，请重新运行训练脚本："
echo "python finetune_distilbert_model.py"
echo "========================================"