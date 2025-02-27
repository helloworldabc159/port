import re
import os
import pandas as pd

# 定义输入和输出文件路径
input_file_path = r'kk1测试.log'
output_file_path = r'C:\Users\wxw\Desktop\yolov5-master\cjw.xlsx'

# 定义字段关键字和对应的存储列表
fields = {
    '时间': [],
    '角点点云坐标_X': [],
    '角点点云坐标_Y': [],
    '角点点云坐标_Z': [],
    '角点NEZ坐标_N': [],
    '角点NEZ坐标_E': [],
    '角点NEZ坐标_Z': [],
    '角点NEZ坐标_其他': [],
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

# 定义正则表达式来匹配中括号
brackets_pattern = r'[\[\]]'

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
            # 去掉中括号
            data_without_brackets = re.sub(brackets_pattern, '', data_without_prefix)
            # 匹配时间戳和三个坐标值
            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)', data_without_brackets)
            if match:
                fields['时间'].append(match.group(1))
                fields['角点点云坐标_X'].append(float(match.group(2)))
                fields['角点点云坐标_Y'].append(float(match.group(3)))
                fields['角点点云坐标_Z'].append(float(match.group(4)))
            else:
                fields['时间'].append(None)
                fields['角点点云坐标_X'].append(None)
                fields['角点点云坐标_Y'].append(None)
                fields['角点点云坐标_Z'].append(None)
        elif '角点NEZ坐标' in line:
            # 去掉时间戳部分、字段前缀和中括号
            data_without_timestamp = re.sub(timestamp_pattern, '', line).strip()
            data_without_prefix = re.sub(prefix_patterns['角点NEZ坐标'], '', data_without_timestamp).strip()
            data_without_brackets = re.sub(brackets_pattern, '', data_without_prefix)
            # 将数据拆分为四个部分
            nez_parts = data_without_brackets.split(',')
            # 分别存入四个字段
            fields['角点NEZ坐标_N'].append(float(nez_parts[0].strip()) if len(nez_parts) > 0 else None)
            fields['角点NEZ坐标_E'].append(float(nez_parts[1].strip()) if len(nez_parts) > 1 else None)
            fields['角点NEZ坐标_Z'].append(float(nez_parts[2].strip()) if len(nez_parts) > 2 else None)
            fields['角点NEZ坐标_其他'].append(float(nez_parts[3].strip()) if len(nez_parts) > 3 else None)
        elif '偏航角' in line:
            # 去掉时间戳部分、字段前缀和中括号
            data_without_timestamp = re.sub(timestamp_pattern, '', line).strip()
            data_without_prefix = re.sub(prefix_patterns['偏航角'], '', data_without_timestamp).strip()
            data_without_brackets = re.sub(brackets_pattern, '', data_without_prefix)
            fields['偏航角'].append(float(data_without_brackets))
        elif '俯仰角' in line:
            # 去掉时间戳部分、字段前缀和中括号
            data_without_timestamp = re.sub(timestamp_pattern, '', line).strip()
            data_without_prefix = re.sub(prefix_patterns['俯仰角'], '', data_without_timestamp).strip()
            data_without_brackets = re.sub(brackets_pattern, '', data_without_prefix)
            fields['俯仰角'].append(float(data_without_brackets))

# 获取最大行数
max_rows = max(len(fields['时间']), len(fields['角点点云坐标_X']),
               len(fields['角点点云坐标_Y']), len(fields['角点点云坐标_Z']),
               len(fields['角点NEZ坐标_N']), len(fields['角点NEZ坐标_E']),
               len(fields['角点NEZ坐标_Z']), len(fields['角点NEZ坐标_其他']),
               len(fields['偏航角']), len(fields['俯仰角']))

# 补齐每一列数据，使所有列长度一致
for key in fields:
    while len(fields[key]) < max_rows:
        fields[key].append(None)

# 创建 DataFrame
df = pd.DataFrame(fields)

# 保存为 Excel 文件
df.to_excel(output_file_path, index=False, engine='openpyxl')

print(f"数据已成功写入 {output_file_path}")
