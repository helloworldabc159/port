def smoothangle(q,value):
    if q.qsize()<10:
        q.put(value)
        return value
    else:
        # 弹出队列
        q.get()
        # 将当前值输入队列
        q.put(value)
        # 算平均值来代替原值
        average_10_num = sum(list(q.queue))/10.0
        return average_10_num

