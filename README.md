# 论文复现 PINN
[*论文：Physics-informed deep learning for incompressible laminar flows*](https://github.com/Raocp/PINN-laminar-flow)

与经典PINN框架的区别：
1. 采用流函数的梯度信息求速度，自然地满足连续方程
2. 引入柯西应力张量

![模型框架](https://github.com/yunbattle1994/PINN-incompressive-laminar-cylinder/blob/main/image/fig1.jpg)



## 稳态问题
![计算域](.//计算域.PNG)

动力粘性 0.02

密度    1.0

入口流速

![稳态入口流速公式](.//稳态入口流速公式.PNG)


![稳态结果](.//稳态结果.PNG)

## 瞬态问题

动力粘性 0.005

密度    1.0

入口流速

![瞬态入口流速公式](.//瞬态入口流速公式.PNG)



![瞬态结果](.//瞬态结果.PNG)

## 文件结构
data
  - 2D_cylinder
    - mixed
      - steady_data.mat     *训练数据*
      - steady_Fluent.mat   _fluent的计算结果_
      - unsteady_data.mat   _训练数据_
    - paddle_openfoam
      - inital     
      - probe               _监督点数据_
      - steady_data.mat     
      - domain_cylinder.csv   
      - domain_inlet.csv
      - domain_outlet.csv
      - domain_train.csv  _计算residual的数据_


pinn_cylinder
  - basic-model.py         基础模型
  - run_steady.py          稳态
  - run_unsteady.py        瞬态
  - run_steady_classical.py 采用经典pinn预测稳态
  - visual_data.py         画图




