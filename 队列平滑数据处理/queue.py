import logging

import pandas as pd
import queue
import SmoothAngle
import test3


# 配置日志记录器
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
# 创建文件处理器，使用追加模式
file_handler = logging.FileHandler('20号数据俯仰角1.log', mode='a')
file_handler.setLevel(logging.INFO)
# 创建日志格式
formatter = logging.Formatter('%(asctime)s.%(msecs)03d- %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到记录器
# logger.addHandler(console_handler)
logger.addHandler(file_handler)
# 读入数据
file_path = r'HHG3.xlsx'  # 替换为你的 Excel 文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称
column_name = '俯仰角(pitch)'  # 替换为你要读取的列名

# 读取数据
data = pd.read_excel(file_path, sheet_name=sheet_name)

# 创建队列
q = queue.Queue(maxsize=10)
# 循环读取一列数据
# ------------------原始方法-------------------------
# i= 1
# for value in data[column_name]:
#     if q.qsize() < 10:
#         q.put(value)
#         logger.info(value)
#         print(f'第{i}次的值{value}')
#         i = i+1
#         continue
#     else:
#         # 计算队列中元素的和
#         average_10_num =  sum(list(q.queue))/10.0
#         print(f'第{i}的值{average_10_num}')
#         logger.info(average_10_num)
#         q.get()
#         q.put(value)
#         i = i+1

#-----------------调用函数----------------------------
for value in data[column_name]:
    value =  SmoothAngle.smoothangle(q,value)
    logger.info(value)




