import os
import shutil
import re

def rename_and_copy_images_ordered(source_folder, target_folder, start_number=500):
    """
    按照数字顺序重命名图片并复制到另一个文件夹
    
    Args:
        source_folder: 源图片文件夹路径
        target_folder: 目标文件夹路径
        start_number: 起始编号，默认为500
    """
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误：源文件夹 {source_folder} 不存在")
        return
    
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)
    
    # 获取源文件夹中所有文件
    try:
        files = os.listdir(source_folder)
    except PermissionError:
        print(f"错误：没有权限访问源文件夹 {source_folder}")
        return
    
    # 过滤出图片文件
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print("源文件夹中没有找到图片文件")
        return
    
    # 提取数字并排序
    def extract_number(filename):
        # 匹配文件名中的数字部分，支持负数
        match = re.search(r'(-?\d+)', filename)
        if match:
            return int(match.group(1))
        return float('inf')  # 如果没有数字，放到最后
    
    # 按照数字顺序排序
    image_files.sort(key=extract_number)
    
    print(f"找到 {len(image_files)} 个图片文件")
    print("排序后的文件顺序:")
    for i, filename in enumerate(image_files):
        print(f"  {i}: {filename}")
    
    # 重命名并复制图片
    current_number = start_number
    copied_count = 0
    
    for old_name in image_files:
        # 获取文件扩展名
        file_extension = os.path.splitext(old_name)[1]
        
        # 构建新文件名（保留原始的后缀部分，如 _0）
        # 提取原始文件名中的后缀部分（如果有的话）
        match = re.search(r'(-?\d+)(.*)', old_name.replace(file_extension, ''))
        if match:
            original_suffix = match.group(2)  # 获取 _0 这样的后缀
        else:
            original_suffix = ""
        
        # 构建新文件名：数字 + 原始后缀 + 扩展名
        new_name = f"{current_number}{original_suffix}{file_extension}"
        
        # 完整的源文件路径和目标文件路径
        old_path = os.path.join(source_folder, old_name)
        new_path = os.path.join(target_folder, new_name)
        
        try:
            # 复制文件到目标文件夹并重命名
            shutil.copy2(old_path, new_path)
            print(f"复制并重命名: {old_name} -> {new_name}")
            current_number += 1
            copied_count += 1
        except Exception as e:
            print(f"复制 {old_name} 失败: {e}")
    
    print(f"成功复制 {copied_count} 个文件到 {target_folder}，从 {start_number} 开始")

# 使用示例
if __name__ == "__main__":
    source_folder = "/home/ubuntu/Documents/code/github/FPV/PiLoT/outputs/feicuiwan_sim_seq7"
    target_folder = "/home/ubuntu/Documents/code/github/FPV/PiLoT/outputs/renamed_feicuiwan_sim_seq7"
    if target_folder not in os.listdir():
        os.mkdir(target_folder)
    
    rename_and_copy_images_ordered(source_folder, target_folder, start_number=500)