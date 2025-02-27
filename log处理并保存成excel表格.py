import re
import os
import pandas as pd

# 定义输入和输出文件路径
input_file_path = r'新第七跑.log'
output_file_path = r'C:\Users\Yang Yuhao\Desktop\3.xlsx'

# 定义字段关键字和对应的存储列表
fields = {
    '角点点云坐标': [],
    '角点NEZ坐标': [],
    '偏航角': [],
    '俯仰角': []
}

# 定义正则表达式来匹配时间戳
timestamp_pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}'

# 定义要删除的字段前缀
prefix_patterns = {
    '角点点云坐标': r'-INFO - 角点点云坐标：',
    '角点NEZ坐标': r'-INFO - 角点NEZ坐标：',
    '偏航角': r'-INFO - 偏航角\(yaw\)：',
    '俯仰角': r'-INFO - 俯仰角\(pitch\)：'
}

# 检查输入文件是否存在
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"输入文件 {input_file_path} 不存在")

# 读取输入文件
with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        # 判断该行属于哪个字段，并处理对应数据
        if '角点点云坐标' in line:
            # 去掉字段前缀
            data_without_prefix = re.sub(prefix_patterns['角点点云坐标'], '', line).strip()
            fields['角点点云坐标'].append(data_without_prefix)
        elif '角点NEZ坐标' in line:
            # 去掉时间戳部分和字段前缀
            data_without_timestamp = re.sub(timestamp_pattern, '', line).strip()
            data_without_prefix = re.sub(prefix_patterns['角点NEZ坐标'], '', data_without_timestamp).strip()
            fields['角点NEZ坐标'].append(data_without_prefix)
        elif '偏航角' in line:
            # 去掉时间戳部分和字段前缀
            data_without_timestamp = re.sub(timestamp_pattern, '', line).strip()
            data_without_prefix = re.sub(prefix_patterns['偏航角'], '', data_without_timestamp).strip()
            fields['偏航角'].append(data_without_prefix)
        elif '俯仰角' in line:
            # 去掉时间戳部分和字段前缀
            data_without_timestamp = re.sub(timestamp_pattern, '', line).strip()
            data_without_prefix = re.sub(prefix_patterns['俯仰角'], '', data_without_timestamp).strip()
            fields['俯仰角'].append(data_without_prefix)

# 获取最大行数
max_rows = max(len(fields['角点点云坐标']), len(fields['角点NEZ坐标']),
               len(fields['偏航角']), len(fields['俯仰角']))

# 补齐每一列数据，使所有列长度一致
for key in fields:
    while len(fields[key]) < max_rows:
        fields[key].append('')

# 创建 DataFrame
df = pd.DataFrame(fields)

# 保存为 Excel 文件
df.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"数据已成功写入 {output_file_path}")
