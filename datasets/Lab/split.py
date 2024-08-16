import os
import random
from sklearn.model_selection import train_test_split

def split_data(input_path, base_output_dir):
    # 读取文件
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 打乱顺序以保证随机性
    random.shuffle(lines)
    
    # 划分数据集
    train, temp = train_test_split(lines, test_size=0.3, random_state=42)
    valid, test = train_test_split(temp, test_size=0.33, random_state=42)
    
    # 创建文件夹
    fold_dir = os.path.join(base_output_dir, '1')
    os.makedirs(fold_dir, exist_ok=True)
    
    # 写入文件
    write_to_file(os.path.join(fold_dir, 'train_links'), train)
    write_to_file(os.path.join(fold_dir, 'valid_links'), valid)
    write_to_file(os.path.join(fold_dir, 'test_links'), test)

def write_to_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# 路径设置
input_path = 'C:\\Users\\ASUS\\Documents\\DXD\\VSC\\OpenEA\\datasets\\tongji_DL_Lab\\TONGJI_DL_LAB\\ent_links'
base_output_dir = 'C:\\Users\\ASUS\\Documents\\DXD\\VSC\\OpenEA\\datasets\\tongji_DL_Lab\\TONGJI_DL_LAB\\721_5fold'

# 执行函数
split_data(input_path, base_output_dir)