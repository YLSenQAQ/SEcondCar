# 神经网络二手车价格预测模型
## 1.使用AIstudio平台进行训练
#### [点击打开AIstudio网址](https://aistudio.baidu.com/aistudio/index)
#### 1.1创建项目,选择 PaddlePaddle版本为1.5.1,Python版本 3.7以上,导入ipynb文件：
```
Aistudio_train.ipynb
```

#### 1.2在data中导入数据集,修改path路径为对应的数据集路径
```
datapath="data/data84434/mycar.csv" #训练集路径
testdatapath="data/data84434/test.csv"#验证集路径
```
#### 1.3根据实际情况选择使用CPU或GPU

```
use_cuda = True #True=显卡 ，False=cpu
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
```
#### 1.4修改训练参数

```
params_dirname="mycar.model"#定义模型保存路径
BATCH_SIZE=100                       #定义每次读取数据量大小
num_epochs=40                          #训练轮次
learning_rate=0.001                     #学习率
use_cuda=True                           #是否使用显卡进行训练

```
#### 1.5填入要预测的二手车数据

```
raw_x= [41,45058,9,2,4,498,2,0,0,0,1,3,0] #填入要预测的二手车的数据['time', 'meter', 'broad', 'Model', 'displacement','horsepower','Gearbox','accident','soak','slight_collision','Paint_repair','Sheet_metal_repair','Appearance_replacement']
```
## 2.使用本地平台进行训练
#### 2.1使用Python3.7以上版本打开python文件：

```
train.py
```
#### 2.2安装相应库
##### paddle 1.5.1版本
##### pandas
##### matplotlib
##### numpy
##### 若遇到连接超时问题,可选择使用国内的清华源：

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple　要下的包名
```

#### 2.3导入数据集与于所在目录，并修改path路径为对应的数据集路径
```
datapath="data/data84434/mycar.csv" #训练集路径
testdatapath="data/data84434/test.csv"#验证集路径
```
#### 2.4根据实际情况选择使用CPU或GPU

```
use_cuda = True #True=GPU ，False=cpu
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
```
#### 2.5修改训练参数

```
params_dirname="mycar.model"#定义模型保存路径
BATCH_SIZE=100                       #定义每次读取数据量大小
num_epochs=40                          #训练轮次
learning_rate=0.001                     #学习率
use_cuda=True                           #是否使用显卡进行训练

```
#### 2.6填入要预测的二手车数据

```
raw_x= [41,45058,9,2,4,498,2,0,0,0,1,3,0]#填入要预测的二手车的数据['time', 'meter', 'broad', 'Model', 'displacement','horsepower','Gearbox','accident','soak','slight_collision','Paint_repair','Sheet_metal_repair','Appearance_replacement']
```