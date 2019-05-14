---
layout: post
title: 深度学习方法与实践
author: lyeeer
---

深度学习课程笔记

（0429 主要介绍tensorflow的相关信息；0506 主要介绍人工神经网络原理；0513主要介绍卷积神经网络）

参考邱锡鹏《神经网络与深度学习》；nudt深度学习方法与实践课程

# '深度学习方法与实践 '笔记

----------------

------------------------

## 深度学习工具与实践

### tensorflow基础：概念与编程模型

主流：tensorflow、caffee、pyTorch

tensorflow就是python中调用的一个库

输入、输出、模型计算过程用计算图graph描述

tensor（张量、数据）+flow（流动）.

**tensorflow中graph和session缺一不可**

**画个图--定义计算图（graph）：描述所有数学计算的图，有向无环**

边:张量，节点之间传递的数据可以看成n维数组

终端节点：输入和模型参数

计算流程：算子（一个计算过程）、节点、数据传递

如何构建？像写函数一样

静态图（tf、caffe，预先定义计算图，运行时反复使用）vs动态图（pytorch，每次运行时重构计算图，灵活便于修改）

**session：会话，运行模型**

给定graph的输入、指定结果的获取方式，并启动数据在graph中的流动

如with tf.device('/cpu:0')

1.明确调用会话生成、关闭函数

2.调用python的上下文管理器使用会话（更常用）

​	with tf.Session() as sess:

-------------------

### tensorflow机器学习编程框架

**使用tensorflow**

使用tensorflow实现线性回归（股票预测）

​	原来：使用sklearn进行线性回归

**建图：**

​	1.创建数据，定义输入输出

​			feed和fetch

​			placeholder：占位符，规定属性但不包括数据			

​	2.定义模型主要部分计算图

​	3.定义损失函数

​		loss = tf.reduce_mean(tf.square(real_label−y_label)) # 定义目标函数loss

​	4.定义优化器和优化目标

​		train = tf.train.GradientDescentOptimizer(0.2).minimize (loss) # 定义优化器及优化目标(最小化loss), 其中0.2为 学习率

**执行：**

​	1.初始化参数

常用的TF初始化值函数: 

tf.constant(const)：常量初始化 

tf.random_normal()：正态分布初始化 

tf.truncated_normal(mean=0.0,stddev=1.0,seed=None,dtype= dtypes.float32)：截取的正态分布初始化 

tf.random_uniform()：均匀分布初始化

全局初始化：tf.global_variables_initializer 

​	2.定义训练脚本并执行

​		session.run()

-------------------------------------------

### tensorflow线性回归

<img src="{{ site.baseurl }}/images/0429-001.png" style="width: 400px;"/>

**模型存储**

ckpt模式：（方便灵活）

1. Metagraph: .meta文件 protocolbuffer保存graph.例如variables,operations,collections
2. Checkpointfile: .ckpt文件 
   1. 2个二进制文件：包含所有的weights,biases,gradients和其他 variables的值。 
   2. mymodel.data-00000-of-00001训练的变量值 
   3. mymodel.index 
3. checkpoint’简单保存最近一次保存checkpoint文件的记录

怎么存？

`Saver=tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)`

` Saver.save(sess,ckpt_file_path,global_step)#global_step为存储步长`

**模型恢复**

重复定义计算图为默认图，用tf.train.Saver()中的restore工具恢复默认图 

`saver.restore(sess,tf.train.latest_checkpoint(’./ckpt’))`

PB模式：（适用于模型封装和移植）

MetaGraph的protocolbuffer格式的文件，包括计算图，数据流， 以及相关的变量等

具有语言独立性，可独立运行，任何语言都可以解析 

存储示例：

`constant_graph = graph_util.convert_variables_to_constants(sess, sess. 	graph_def, ['op_to_store'])` 

`with tf.gfile.FastGFile('./tmp2/pbplus.pb', mode=' wb') as f:` 			`f.write(constant_graph.SerializeToString())`

------------------------------------

### Eager Execution

EagerExecution采用直接定义输入变量的模式，不使用placeholder 

当启动EagerExecution时，运算会即刻执行，无需Session.run()就可以把它们的值返回到Python

`import tensorflow.contrib.eager as tfe`

`tfe.enable_eager_execution()`

EagerExecution中不能自动调用GPU资源 :如果要在EagerExecution中使用GPU计算资源，则需要显式地将 tensor移动到指定的GPU中

`a = a.gpu() # copies tensor to default GPU (GPU0)` 

-------------------------

-------------------------------

## 人工神经网络原理

### 起源与发展

仿生意义：神经元&树突&轴突&突触&状态

模仿大脑意识和思考

***spike neural network***

构建类脑的生物网络结构（构建大量小的神经元实现功能，如10^8--10^10量级）

***deep neural network***

数学建模神经网络机理获得脑智能

### 人工神经网络的建模（发展历史）

1943年，MP抽象神经网络原型（MP神经 元中的激活函数f 为0或1的阶跃函数，而现代神经元中的激活函数通常要求是 连续可导的函数）

感知机时代（解决二分类/线性可分问题）---只有一个神经元的ANN，权重（突触）、偏置（阈值）及激活函数（细胞体），输出为+1或-1

​		*局限性：如果不是二分类/线性回归，不能确保收敛

***多层感知机***

​		*全连接多层神经网络

​		*多层神经网络的前向传播

***神经网络线性变换的意义***

​		*线性变换--旋转

​		*降维

​		*缩放与平移

​		*非线性变换--弯曲

​		*非线性映射--激活函数（relu/sigmoid）

***神经网络分层的意义***

​		*大多数问题是非线性的

​		*逐层才能以简单的方法实现复杂问题（尤其非线性）

​		*层数增加，对原始输入的“扭曲力”会增加

***更复杂的分类：softmax与交叉熵***

​		softmax表达样本属于某类概率，softmax函数是将多个标量映射为一个概率分布。 

​		概率的优化目标：极大似然、分布距离度量（熵/散度）

对应分布为p(x)的随机变量，熵H(p)表示其最优编码长度。交叉熵（Cross Entropy）是按照概率分布q的最优编码对真实分布为p的信息进行编码的长度。

在给定p的情况下，如果q和p越接近，交叉熵越小；如果q和p越远，交叉熵就越大。

### tensorflow神经网络训练方法

e.g. mnist手写数字识别：

​	1.数据预处理

​	2.定义所需的神经网络（构建计算图）

​	3.会话中循环执行训练网络

​	4.评估网络性能

-----------------

-----------------

## 卷积神经网络

***传统全连接神经网络的局限性*** :参数太多、局部不变性特征（如对于自然图像中的物体，尺度缩放/平移/旋转等操作不影响语义信息）

***人类视觉认知的规律***

感受野（Receptive Field） 主要是指听觉、视觉等神经系统中一些神经元的特性，即神经元只接受其所支 配的刺激区域内的信号

***卷积***

物理上的意义就是图像滤波。二维卷积示例：

<img src="{{ site.baseurl }}/images/0429-002.png" style="width: 400px;"/>

在图像处理中，卷积经常作为特征提取的有效方法。一幅图像在经过卷积 操作后得到结果称为特征映射（Feature Map），每个特征映射可以作为一类抽取的图像特征

滤波器的步长（Stride）是指滤波器在滑动时的时间间隔。改变输出特征的大小，隔多少个元素来取。

零填充（Zero Padding）是在输入向量两端进行补零。

***卷积层***

重要性质：局部连接（在卷积层（假设是第l层）中的每一个神经元都只和下一层（第l−1 层）中某个局部窗口内的神经元相连，构成一个局部连接网络）、权重共享（作为参数的滤波器w(l) 对于第l层的所有的 神经元都是相同的）

***tensorflow使用卷积***

conv2d

***非线性激活函数***

relu/sigmod

***池化层***

池化过程定义了固定大小的池化窗口，其作用是进行特征选择，降低特征数量，并从而减少参数数量。即对于每个区域进行下采样得到一个值，作为这个区域的概括。

maximum pooling/mean polling

典型的汇聚层是将每个特征映射划分为2×2大小的不重叠区域，然后使用最大汇聚的方式进行下采样。汇聚层也可以看做是一个特殊的卷积层，卷积核大小为m×m，步长为s×s，卷积核为max函数或mean函数。过大的采样区域会急剧减少神经元的数量，会造成过多的信息损失。 

全连接层（本质为矩阵乘法）把高维的变成全局的特征，再连接softmax进行分类

典型的卷积网络结构：

<img src="{{ site.baseurl }}/images/0429-003.png" style="width: 400px;"/>

### homework：用tensorboard看背后流程

可以把前面的tf.xx操作看成建立graph，像是先构建一个水管，这个时候并不会运行。后面的session相当于从水管里注水运行操作，但是要sess.run()一次才会动一次。fetch的tensor涉及到的子图才会动

如果是variable初始化的赋值只有tf.global_variables_initializer()管。数据驱动图，从一次fetch的最高节点倒推到设计的最初节点，从下往上这个计算

python面向对象技术

鸭子模式（duck）