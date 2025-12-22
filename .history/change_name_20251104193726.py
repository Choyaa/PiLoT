import os
import shutil

def rename_and_copy_images(source_folder, target_folder, start_number=500):
    """
    重命名图片并复制到另一个文件夹
    
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
    
    # 过滤出图片文件并按文件名排序
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    image_files.sort()
    
    if not image_files:
        print("源文件夹中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 重命名并复制图片
    current_number = start_number
    copied_count = 0
    
    for old_name in image_files:
        # 获取文件扩展名
        file_extension = os.path.splitext(old_name)[1]
        
        # 构建新文件名
        new_name = f"{current_number}{file_extension}"
        
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
    target_folder = "/home/ubuntu/Documents/code/github/FPV/PiLoT/outputs/renamed_feicuiwan_sim_seq7"  # 修改为你想要的目标文件夹
    if target_folder not in os.listdir():
        os.mkdir(target_folder)
    
    rename_and_copy_images(source_folder, target_folder, start_number=500)