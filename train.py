#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import numpy as np
import sys 
import matplotlib.pyplot as plt
import pandas as pd
import paddle
import paddle.fluid as fluid
import math




datapath="data/data84434/mycar.csv" #训练集
testdatapath="data/data84434/test.csv"#验证集
params_dirname = "mycar.model" #params_dirname用于定义模型保存路径。
BATCH_SIZE = 100 #定义每次读取数据量大小
num_epochs = 20  #训练轮次
learning_rate = 0.005  #学习率
use_cuda = True #True=GPU ，False=cpu
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

# * 上牌时间(0-140) time
# * 里程数(0-250000) meter
# * 汽车厂商(1-10) broad    
# * 汽车类型(1-4) Model
# * 汽车排量(0-6.2)displacement
# * 最大马力(20-585)horsepower
# * 变速箱(1-3)Gearbox
# * 重大事故(0-1) accident
# * 泡水火烧(0-1) soak
# * 轻微碰撞(0-1) slight_collision
# * 漆面修复次数(0-3) Paint_repair
# * 钣金修复次数(0-3) Sheet_metal_repair
# * 外观件更换次数(0-3)  Appearance_replacement
# * 
# 
# 
# 
# 

# 读取处理数据 定义模型

data = pd.read_csv(datapath)#读取数据
row = data.shape[0]
column = data.shape[1]
data_2=data
global x_raw,train_data,test_data
x_raw = data_2.T.copy()
maximums, minimums, avgs= data_2.max(axis=0), data_2.min(axis=0), data_2.mean(axis=0)
feature_num = data_2.shape[1]
for i in range(feature_num-1):#售价不用改
    data_2.iloc[:, [i]] = (data_2.iloc[:, [i]]- minimums[i])/(maximums[i]-minimums[i])#归一化


print(data_2.head())

#划分训练集 测试集
data_3=np.array(data_2)
print(data_3[0])
ratio = 0.8
offset = int(data_3.shape[0]*ratio)
train_data = data_3[:offset].copy()
test_data = data_3[offset:].copy()
print(train_data)
print(len(train_data))

#读取器
def read_data(data_set):

    def reader():
        """
 
        Return：
            data[:-1],data[-1:] --使用yield返回生成器
                data[:-1]表示前n-1个元素，也就是训练数据，
                data[-1:]表示最后一个元素，也就是对应的标签(价格)
        """
        for data in data_set:
            yield data[:-1],data[-1:]
    return reader
test_array = ([10,100],[20,200])
print("test_array for read_data:")
for value in read_data(test_array)():
    print(value)   

# 设置训练reader
train_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(train_data), 
        buf_size=1000)#表示每次缓存BUF_SIZE个数据项，并进行打乱
        ,batch_size=BATCH_SIZE)

#设置测试 reader
test_reader = paddle.batch(
    paddle.reader.shuffle(
        read_data(test_data), 
        buf_size=1000),
    batch_size=BATCH_SIZE)

#定义训练模型

x = fluid.layers.data(name='x', shape=[13], dtype='float32')
# 标签数据，fluid.layers.data表示数据层,name=’y’：名称为y,输出类型为tensor
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
# 输出层，fluid.layers.fc表示全连接层，input=x: 该层输入数据为x
y_first = fluid.layers.fc(input=x, size=6, act=None, bias_attr=True)
y_predict = fluid.layers.fc(input=y_first, size=1, act=None, bias_attr=True)


# 定义损失函数为均方差损失函数,并且求平均损失，返回值名称为avg_loss
loss = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_loss = fluid.layers.mean(loss)
# 定义执行器(参数随机初始化):
exe = fluid.Executor(place)
# 配置训练程序:
main_program = fluid.default_main_program() # 获取默认/全局主函数
startup_program = fluid.default_startup_program() # 获取默认/全局启动程序
test_program = main_program.clone(for_test=True)

sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate)
# sgd_optimizer = fluid.optimizer.SGD(learning_rate)  # 随机梯度下降
# sgd_optimizer = fluid.optimizer.Adam(learning_rate)
sgd_optimizer.minimize(avg_loss)


def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(
            program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]  # 累加测试过程中的损失值
        count += 1 # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated] # 计算平均损失





# 训练




#用于画图展示训练cost
from paddle.utils.plot import Ploter
train_prompt = "Train cost"
test_prompt = "Test cost"
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

# 训练主循环
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe.run(startup_program)
exe_test = fluid.Executor(place)



for pass_id in range(num_epochs):

    for data_train in train_reader():
        avg_loss_value, = exe.run(main_program,
                                  fetch_list=[avg_loss],
                                  feed=feeder.feed(data_train))
        if step % 10 == 0:  # 每10个批次记录并输出一下训练损失.
            plot_prompt.append(train_prompt, step, avg_loss_value[0])
            plot_prompt.plot()

        if step % 100 == 0:  # 每100批次记录并输出一下测试损失
            test_metics = train_test(executor=exe_test,
                                     program=test_program,
                                     reader=test_reader,
                                     fetch_list=[avg_loss.name],
                                     feeder=feeder)
            plot_prompt.append(test_prompt, step, test_metics[0])
            plot_prompt.plot()
            print(pass_id)
            print("%s, Step %d, Cost %f" %(test_prompt, step, test_metics[0]))

        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")

        #保存训练参数到之前给定的路径中
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)


# 验证




infer_exe = fluid.Executor(place)    #创建推测用的executor
inference_scope = fluid.core.Scope() #Scope指定作用域

with fluid.scope_guard(inference_scope):#修改全局/默认作用域（scope）, 运行时的所有变量都将分配给新的scope。
    #从指定目录中加载 预测用的model(inference model)
    [inference_program,                             #推理的program
     feed_target_names,                             #需要在推理program中提供数据的变量名称
     fetch_targets] = fluid.io.load_inference_model(   #fetch_targets: 推断结果
                                    params_dirname,    #model_save_dir:模型训练路径 
                                    infer_exe)         #infer_exe: 预测用executor
   
    datatest = pd.read_csv(testdatapath)
    row2 = datatest.shape[0]
    print(row2)
    data_2= datatest.loc[:,['time', 'meter', 'broad', 'Model', 'displacement',
           'horsepower','Gearbox','accident','soak','slight_collision',
           'Paint_repair','Sheet_metal_repair','Appearance_replacement']]
    data_y=np.array(datatest.loc[:,['price']])

    maximums, minimums, avgs= data.max(axis=0), data.min(axis=0), data.mean(axis=0)
    feature_num = data_2.shape[1]
    for i in range(feature_num):#售价不用改
        data_2.iloc[:, [i]] = (data_2.iloc[:, [i]]- minimums[i])/(maximums[i]-minimums[i])
    
    data_3=np.array(data_2)
    step=0
    array_lost=[]
    for number in range(row2):
        tensor_x = []
        tensor_x = [np.array(data_3[number]).astype("float32")]
        #print(tensor_x)
        results = infer_exe.run(inference_program,                              #预测模型
                            feed={feed_target_names[0]: np.array(tensor_x)},  #喂入要预测的x值
                            fetch_list=fetch_targets)                       #得到推测结果 
        step+=1               
        
        for idx, val in enumerate(results[0]):
            if val[0]<0.05 :
               
                val2=0.05
            else:
                val2=val[0]
                lost=abs(val2-data_y[number][0])#相差的绝对值
                array_lost.append(lost)#只统计大于0.05的数据
                
            print("%d: 预测%.2f 实际%.2f △=%.2f" % (number,val2,data_y[number],lost))
          
    mean_lost=np.mean(array_lost)
    print("平均偏差：")
    print("±△=",mean_lost)


# 预测




raw_x= [41,45058,9,2,4,498,2,0,0,0,1,3,0] #填入要预测的二手车的数据
'''['time', 'meter', 'broad', 'Model', 'displacement',
           'horsepower','Gearbox','accident','soak','slight_collision',
           'Paint_repair','Sheet_metal_repair','Appearance_replacement']
'''
print("平均偏差：±△=",mean_lost)
tensor_x = []
data = pd.read_csv(datapath)

maximums, minimums, avgs= data.max(axis=0), data.min(axis=0), data.mean(axis=0)
print(len(raw_x))
for i in range(len(raw_x)):
    tensor_x.append((raw_x[i] - minimums[i])/(maximums[i]-minimums[i]))


tensor_x = [np.array(tensor_x).astype("float32")]

print(tensor_x)
results = infer_exe.run(inference_program,                              #预测模型
                    feed={feed_target_names[0]: np.array(tensor_x)},  #喂入要预测的x值
                    fetch_list=fetch_targets)                       #得到推测结果             
for idx, val in enumerate(results[0]):
    if val[0]<0.05 :
        val2=0.05
    else:
        val2=val[0]

    print(" 预测：现价=（%.4f ±%.4f）*原价" % (val2,mean_lost))





