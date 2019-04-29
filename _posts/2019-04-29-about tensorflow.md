---
layout: post
title: 深度学习工具与实践
author: lyeeer
---

深度学习课程（主要介绍tensorflow的相关信息）

# 深度学习工具与实践



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

`Saver=tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2) Saver.save(sess,ckpt_file_path,global_step)#global_step为存储步长`

**模型恢复**

重复定义计算图为默认图，用tf.train.Saver()中的restore工具恢复默认图 

`saver.restore(sess,tf.train.latest_checkpoint(’./ckpt’))`

PB模式：（适用于模型封装和移植）

MetaGraph的protocolbuffer格式的文件，包括计算图，数据流， 以及相关的变量等

具有语言独立性，可独立运行，任何语言都可以解析 

存储示例：

`constant_graph = graph_util.convert_variables_to_constants(sess, sess. 			graph_def, ['op_to_store'])` 
`	with tf.gfile.FastGFile('./tmp2/pbplus.pb', mode=' wb') as f: 										f.write(constant_graph.SerializeToString())`

------------------------------------

### Eager Execution

EagerExecution采用直接定义输入变量的模式，不使用placeholder 

当启动EagerExecution时，运算会即刻执行，无需Session.run()就可以把它们的值返回到Python

`import tensorflow.contrib.eager as tfe`
`tfe.enable_eager_execution()`

EagerExecution中不能自动调用GPU资源 :如果要在EagerExecution中使用GPU计算资源，则需要显式地将 tensor移动到指定的GPU中`a = a.gpu() # copies tensor to default GPU (GPU0)` 

