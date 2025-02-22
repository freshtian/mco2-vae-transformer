import numpy as np
import pandas as pd
import os

# 定义文件夹路径
npy_folder_path = './'
csv_folder_path = './final-csv/'

# 确保CSV文件夹存在
os.makedirs(csv_folder_path, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(npy_folder_path):
    if filename.endswith('.npy'):
        # 构建完整路径
        npy_file_path = os.path.join(npy_folder_path, filename)

        # 1. 加载.npy文件
        data = np.load(npy_file_path)

        # 2. 展平数组
        data_flattened = data.reshape(-1, data.shape[-1])

        # 3. 将NumPy数组转换为Pandas DataFrame
        df = pd.DataFrame(data_flattened)

        # 4. 构建CSV文件路径
        csv_filename = filename.replace('.npy', '.csv')
        csv_file_path = os.path.join(csv_folder_path, csv_filename)

        # 5. 保存为CSV文件
        df.to_csv(csv_file_path, index=False)

        print(f"文件 {filename} 已成功保存为 {csv_filename}")

print("所有文件转换完成")
