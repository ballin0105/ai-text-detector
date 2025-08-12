# code/check_gpu.py

import torch

def check_pytorch_gpu():
    """
    检查PyTorch是否能正确识别并使用NVIDIA GPU。
    """
    print("--- PyTorch GPU 环境诊断 ---")
    
    # 1. 检查CUDA是否可用
    is_available = torch.cuda.is_available()
    
    if is_available:
        print("\n✅ 恭喜！PyTorch已成功检测到您的GPU。")
        
        # 2. 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"   - 检测到的GPU数量: {gpu_count}")
        
        # 3. 获取当前GPU的名称
        current_gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu_index)
        print(f"   - 当前使用GPU: Device {current_gpu_index}")
        print(f"   - GPU型号: {gpu_name}")
        
        # 4. 获取当前GPU的显存大小
        total_mem = torch.cuda.get_device_properties(current_gpu_index).total_memory
        total_mem_gb = round(total_mem / (1024**3), 2)
        print(f"   - 显存大小: {total_mem_gb} GB")
        
        print("\n您的环境已为GPU训练准备就绪！")
        
    else:
        print("\n❌ 注意：PyTorch未能检测到可用的GPU。")
        print("   - 可能的原因包括：")
        print("     1. 您没有安装NVIDIA GPU驱动程序。")
        print("     2. 您安装的PyTorch版本是CPU-only版本。")
        print("     3. 您可能需要重启电脑或重新激活Conda环境。")
        print("   - 脚本将会在CPU上运行，速度会非常慢。")
        
    print("\n--- 诊断结束 ---")

if __name__ == '__main__':
    check_pytorch_gpu()