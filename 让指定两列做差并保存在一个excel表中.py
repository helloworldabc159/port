import pandas as pd

# 指定Excel文件路径
excel_file_path = r'C:\Users\Yang Yuhao\Desktop\ran改成0.25数据\三对比5.xlsx'

try:
    # 读取Excel文件
    df = pd.read_excel(excel_file_path)

    # 指定要计算差值的列名
    column1 = 'Z旧'  # 第一列的名称
    column2 = 'Z真'  # 第二列的名称

    # 计算两列的差值
    df['Difference'] = df[column1] - df[column2]

    # 将结果保存到新的Excel文件中
    output_file_path = r'C:\Users\Yang Yuhao\Desktop\ran改成0.25数据\旧z误差.xlsx'
    df.to_excel(output_file_path, index=False)

    print(f'Difference calculated and saved to {output_file_path}')

except PermissionError:
    print(f"Permission denied: unable to write to {output_file_path}")
    print("Please check if the file is open in another program or if you have write permissions to the folder.")
