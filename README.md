# 论文复现 PINN
[*论文：Physics-informed deep learning for incompressible laminar flows*](https://github.com/Raocp/PINN-laminar-flow)

与经典PINN框架的区别：
1. 采用流函数的梯度信息求速度

2. 引入柯西应力张量

![计算域](https://github.com/yunbattle1994/PINN-incompressive-laminar-cylinder/blob/main/image/fig2.jpg)

![模型框架](https://github.com/yunbattle1994/PINN-incompressive-laminar-cylinder/blob/main/image/fig1.jpg)



## 稳态问题


动力粘性  $0.02 kg/(m \cdot s)$

密度    $1.0 kg/m^3$

入口流速

$u(0, y)=4U_{max}(H-y)y/H^2$

$U_{max}=1.0 m/s$

![稳态结果](https://github.com/yunbattle1994/PINN-incompressive-laminar-cylinder/blob/main/image/fig4.jpg)

## 瞬态问题

动力粘性 $0.005 kg/(m \cdot s)$

密度   $1.0 kg/m^3$

入口流速

$u(0, y)=4[\sin(\frac{2 \pi t}{T} + \frac{3 \pi }{2}) + 1]U_{max}(H-y)y/H^2$

$U_{max} = 0.5 m/s$

$T = 1.0 s$

![瞬态结果](https://github.com/yunbattle1994/PINN-incompressive-laminar-cylinder/blob/main/image/fig6.jpg)

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
  - basic-model.py         _PINN 基础模型_
  - run_steady.py          _稳态复现_
  - run_unsteady.py        _瞬态复现_
  - run_steady_classical.py _采用经典PINN预测稳态_
  - visual_data.py         _画图




