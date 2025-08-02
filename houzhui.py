import os
import re

def remove_pattern_from_filename(folder_path, pattern):
    # 确保文件夹路径存在
    if not os.path.exists(folder_path):
        print(f"文件夹路径 {folder_path} 不存在")
        return

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否匹配指定的模式
        if os.path.isfile(file_path) and re.search(pattern, filename, re.IGNORECASE):
            # 去除文件名中的匹配部分
            new_filename = re.sub(pattern, '', filename, flags=re.IGNORECASE)
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"已从 {filename} 中去除模式匹配部分，新文件名为 {new_filename}")

# 用法示例
folder_path = r"D:\桌面\RD"  # 替换为实际的文件夹路径
pattern_to_remove = r'_fake_bi'  # 替换为要去除的模式，这里使用了正则表达式的形式
remove_pattern_from_filename(folder_path, pattern_to_remove)
