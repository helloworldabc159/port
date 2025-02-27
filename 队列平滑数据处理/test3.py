import pandas as pd
import re

def extract_numbers_from_log(log_file):
    """
    从日志文件中提取数字。

    参数:
    log_file (str): 日志文件的路径。

    返回:
    list: 提取的数字列表。
    """
    numbers = []
    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用正则表达式提取数字
            match = re.search(r'INFO - ([\d\.]+)', line)
            if match:
                numbers.append(float(match.group(1)))  # 将提取的字符串转换为浮点数
    return numbers

def save_numbers_to_excel(numbers, output_file):
    """
    将提取的数字保存到 Excel 文件。

    参数:
    numbers (list): 要保存的数字列表。
    output_file (str): 输出 Excel 文件的路径。
    """
    df = pd.DataFrame(numbers, columns=['Values'])  # 创建 DataFrame
    df.to_excel(output_file, index=False)  # 写入 Excel 文件

# 示例调用
log_file_path = '20号数据俯仰角1.log'  # 日志文件路径
output_excel_path = '20号数据俯仰角1.xlsx'  # 输出 Excel 文件路径

# 提取数字并保存到 Excel
numbers = extract_numbers_from_log(log_file_path)
save_numbers_to_excel(numbers, output_excel_path)

print(f'提取的数字已保存到 {output_excel_path}')
