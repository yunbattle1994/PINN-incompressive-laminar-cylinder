import matplotlib.tri
import numpy as np
import torch
import torch.nn as nn
from basic_model import DeepModel_single, DeepModel_multi, gradients
from visual_data import matplotlib_vision
import random
import time
from tqdm import trange
import matplotlib.pyplot as plt
import os


#################################### 定义网络框架 ###################################################################
# 输入 inn_var : x, y, t
# 输出 out_var : p, u, v, s11, s12, s22
class Net(DeepModel_single):
    def __init__(self, planes, rho, miu):
        super(Net, self).__init__(planes, active=nn.Tanh())
        self.rho = rho
        self.miu = miu

    # 将网络的直接输出 psi ,p, s11, s12, s22 转化为 p, u, v, s11, s12, s22
    def output_transform(self, inn_var, out_var):
        psi, p, s11, s22, s12 = \
            out_var[..., 0:1], out_var[..., 1:2], out_var[..., 2:3], out_var[..., 3:4], out_var[..., 4:5]

        w = gradients(psi, inn_var)
        u, v = w[..., 1:2], -w[..., 0:1]
        return torch.cat((p, u, v, s11, s12, s22), dim=-1)

    # 计算残差
    def equation(self, inn_var, out_var):
        p, u, v, s11, s12, s22 = out_var[..., (0,)], out_var[..., (1,)], out_var[..., (2,)], \
                                 out_var[..., (3,)], out_var[..., (4,)], out_var[..., (5,)]
        dpda, duda, dvda = gradients(p, inn_var), gradients(u, inn_var), gradients(v, inn_var)
        dpdy = dpda[..., 1:2]
        dudx, dudy, dudt, dvdx, dvdy, dvdt = duda[..., 0:1], duda[..., 1:2], duda[..., 2:3], \
                                             dvda[..., 0:1], dvda[..., 1:2], dvda[..., 2:3]

        s11_1 = gradients(s11, inn_var)[..., 0:1]
        s12_2 = gradients(s12, inn_var)[..., 1:2]
        s22_2 = gradients(s22, inn_var)[..., 1:2]
        s12_1 = gradients(s12, inn_var)[..., 0:1]

        eq_p = p + (s11 + s22) / 2
        eq_u = self.rho * dudt + self.rho * (u*dudx + v*dudy) - s11_1 - s12_2
        eq_v = self.rho * dvdt + self.rho * (u*dvdx + v*dvdy) - s12_1 - s22_2
        eq_s11 = -p + 2*self.miu*dudx - s11
        eq_s22 = -p + 2*self.miu*dvdy - s22
        eq_s12 = self.miu*(dudy+dvdx) - s12
        eqs = torch.cat((eq_p, eq_u, eq_v, eq_s11, eq_s12, eq_s22), dim=-1)

        return eqs, torch.cat((dpdy, dudy, dvdy), dim=-1)


######################## 获取 nodes 在 box 流域内的边界节点  ########################
def BCS_ICS(nodes, box):
    BCS = []
    Num_Nodes = nodes.shape[0]
    Index = np.arange(Num_Nodes)

    BCS.append(Index[nodes[:, 0] == box[0]])  # inlet
    BCS.append(Index[nodes[:, 0] == box[2]])  # outlet
    BCS.append(Index[nodes[:, 1] == box[1]])  # top
    BCS.append(Index[nodes[:, 1] == box[3]])  # bottom
    BCS.append(Index[np.abs((nodes[:, 0]-0)**2 + (nodes[:, 1]-0)**2 - (D/2)**2) < 1e-7])  # cylinder wall

    if nodes.shape[-1] == 3:
        BCS.append(Index[nodes[:, 2] == 0])  # initial

    return BCS


######################## 读取数据  ########################
def read_data(**kwargs):
    import scipy.io as sio
    if kwargs['steady']:
        data = sio.loadmat('..\\data\\2D_cylinder\\mixed\\steady_data.mat')
        INLET, OUTLET, WALL= data['INLET'][..., :2], data['OUTLET'], data['WALL']
        num = INLET.shape[0] + OUTLET.shape[0] + WALL.shape[0]
        XY_c  = data['XY_c'][:-num]
        data = sio.loadmat('..\\data\\2D_cylinder\\mixed\\steady_Fluent.mat')
        fields_fluent= np.squeeze(data['field']).T[..., (0, 1, 4, 2, 3)]
        return XY_c, INLET, OUTLET, WALL, fields_fluent
    else:
        data = sio.loadmat('..\\data\\2D_cylinder\\mixed\\unsteady_data.mat')
        INLET, OUTLET, WALL, INITIAL= data['INB'][..., :3], data['OUTB'], data['WALL'], data['IC']
        num = INLET.shape[0] + OUTLET.shape[0] + WALL.shape[0]
        XY_c  = data['XY_c'][:-num]
    return XY_c, INLET, OUTLET, WALL, INITIAL

def read_paddle_data(num_time):
    file_path = '..\\data\\2D_cylinder\\paddle_openfoam\\'
    cyl = np.loadtxt(file_path + 'domain_cylinder.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)] #xypuv
    inlet = np.loadtxt(file_path + 'domain_inlet.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)]
    outlet = np.loadtxt(file_path + 'domain_outlet.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)]
    train = np.loadtxt(file_path + 'domain_train.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)]
    initial = np.loadtxt(file_path + 'initial\\ic0.1.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)]

    # plt.figure(1)
    # plt.plot(cyl[:, 0], cyl[:, 1], 'r.')
    # plt.plot(inlet[:, 0], inlet[:, 1], 'b.')
    # plt.plot(outlet[:, 0], outlet[:, 1], 'b.')
    # plt.plot(train[:, 0], train[:, 1], 'g.')
    # plt.show()
    # plt.figure(2)
    # plt.plot(train[:, 0], train[:, 1], 'k.')
    # plt.show()
    # plt.figure(2)
    # plt.subplot(221)
    # plt.scatter(initial[:, 0], initial[:, 1], s=0.1, c=initial[:, 2])
    # plt.subplot(222)
    # plt.scatter(initial[:, 0], initial[:, 1], s=0.1, c=initial[:, 3])
    # plt.subplot(223)
    # plt.scatter(initial[:, 0], initial[:, 1], s=0.1, c=initial[:, 4])
    # plt.show()
    # plt.figure(3)
    # plt.plot(train[:, 0], train[:, 1], 'g.')
    probe = []
    times_list_all = []
    dirs = os.listdir(file_path + 'probe\\')
    #####获取时间
    times_list = np.arange(1, 51)#np.random.choice(times_list_all, num_time)

    for time in times_list:
        data = np.loadtxt(file_path + '/probe/probe0.' + str(time) + '.csv', skiprows=1, delimiter=',')[..., (5, 6, 0, 1, 2,)]
        t_len = data.shape[0]
        supervised_t = np.array([time] * t_len).reshape((-1, 1))
        data = np.concatenate((data[..., (0, 1)], supervised_t, data[..., (2, 3, 4)]), axis=1)
        probe.append(data)

    full_supervised_data = np.concatenate(probe)

    inlet = replicate_time_list(times_list, inlet.shape[0],  inlet)
    outlet = replicate_time_list(times_list, outlet.shape[0],  outlet)
    initial = replicate_time_list([0.1], initial.shape[0],  initial)
    cyl = replicate_time_list(times_list, cyl.shape[0],  cyl)
    train = replicate_time_list(times_list, train.shape[0],  train)

    return train,  inlet, outlet, cyl, full_supervised_data, initial


def replicate_time_list(time_list, domain_shape, spatial_data):
    all_t = []
    count = 0
    all_data = []
    for t in time_list:
        tmp_t = [t] * domain_shape
        all_t.append(tmp_t)
        tmp = spatial_data
        all_data.append(tmp)
    replicated_t = np.array(all_t).reshape(-1, 1)
    spatial_data = np.concatenate(all_data)

    spatial_data = np.concatenate((spatial_data[..., (0, 1)], replicated_t, spatial_data[..., (2, 3, 4)]), axis=1)
    return spatial_data


################################## 单次训练步骤  ##################################


def train(inn_var, BCs, out_true, model, Loss, optimizer, scheduler, log_loss):
    BC_in = torch.tensor(BCs[0][..., 0:3], dtype=torch.float32).to(device)
    BC_out = torch.tensor(BCs[1][..., 0:3], dtype=torch.float32).to(device)
    BC_wall = torch.tensor(BCs[2][..., 0:3], dtype=torch.float32).to(device)
    BC_initial = torch.tensor(BCs[-1][..., 0:3], dtype=torch.float32).to(device)
    field_supervised = torch.tensor(out_true[..., 0:3], dtype=torch.float32).to(device)

    BC_in_m = torch.tensor(BCs[0][..., 3:], dtype=torch.float32).to(device)
    BC_out_m = torch.tensor(BCs[1][..., 3:], dtype=torch.float32).to(device)
    BC_wall_m = torch.tensor(BCs[2][..., 3:], dtype=torch.float32).to(device)
    BC_initial_m = torch.tensor(BCs[-1][..., 3:], dtype=torch.float32).to(device)
    field_supervised_m = torch.tensor(out_true[..., 3:], dtype=torch.float32).to(device)

    def closure():
        inn_var.requires_grad_(True)
        BC_in.requires_grad_(True)
        BC_out.requires_grad_(True)
        BC_wall.requires_grad_(True)
        BC_initial.requires_grad_(True)
        field_supervised.requires_grad_(True)
        optimizer.zero_grad()

        out_var = model(inn_var)
        out_var = model.output_transform(inn_var, out_var)
        res_i, _ = model.equation(inn_var, out_var)
        out_var = out_var[..., 0:3]

        ##inlet loss  u,v

        pred_in = model(BC_in)
        pred_in = model.output_transform(BC_in, pred_in)
        bcs_loss_in = Loss(pred_in[..., (1, 2)], BC_in_m[..., (1, 2)])

        ##outlet loss p
        pred_out = model(BC_out)
        pred_out = model.output_transform(BC_out, pred_out)
        bcs_loss_out = Loss(pred_out[..., 0], BC_out_m[..., 0])
        ##wall loss u,v
        pred_wall = model(BC_wall)
        pred_wall = model.output_transform(BC_wall, pred_wall)
        bcs_loss_wall = Loss(pred_wall[..., (1, 2)], BC_wall_m[..., (1, 2)])
        ##initial loss u v p
        pred_initial = model(BC_initial)
        pred_initial = model.output_transform(BC_initial, pred_initial)
        bcs_loss_initial = Loss(pred_initial[..., (0, 1, 2)], BC_initial_m[..., (0, 1, 2)])

        bcs_loss = bcs_loss_in * 10 + bcs_loss_out + bcs_loss_wall * 10 + bcs_loss_initial * 10

        ## supervised loss
        pred_field = model(field_supervised)
        pred_field = model.output_transform(field_supervised, pred_field)
        supervised_loss = Loss(pred_field[..., (0, 1, 2)], field_supervised_m[..., (0, 1, 2)])

        eqs_loss = (res_i ** 2).mean()

        loss_batch = bcs_loss * 1. + eqs_loss + supervised_loss *10

        loss_batch.backward()

        # data_loss = Loss(out_var, out_true)
        log_loss.append([eqs_loss.item(), bcs_loss.item(),
                         bcs_loss_wall.item(), bcs_loss_in.item(),
                         bcs_loss_out.item(), bcs_loss_initial.item(), supervised_loss.item()])

        return loss_batch

    optimizer.step(closure)
    scheduler.step()


################################## 预测  ##################################
def inference(inn_var, model):
    inn_var = inn_var.cuda()
    inn_var.requires_grad_(True)
    out_var = model(inn_var)
    out_var = model.output_transform(inn_var, out_var)
    equation, _ = model.equation(inn_var, out_var)
    return out_var.detach().cpu(), equation.detach().cpu()


if __name__ == '__main__':


    name = 'trans-cylinder-2d-mixed-'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    #################### 定义问题相关参数 ####################
    Rho, Miu, D = 1.0, 0.02,  2
    num_time = 50         #####随机抽取时间步个数
    # U_max, tmax = 0.5, 0.5  # 入口流速的最大值以及非定常周期 tmax = T/2
    # Box = [0, 0, 1.1, 0.41]  # 矩形流域
    
    #################### 读入数据 ####################
    data_ori = read_paddle_data(num_time)
    data = list(map(np.random.permutation, data_ori)) # np.random.shuffle & random.shuffle 返回None,此外， python 3 中map返回的是迭代器
    input = data[0]
    input = torch.tensor(input[:, :3], dtype=torch.float32).to(device)
    BCs = (data[1], data[2], data[3], data[5]) ## 边界数据
    field = data[4]   ##检测的流场点


    # 采用三角形 对非结构化网格建立节点连接关系
    triang = matplotlib.tri.Triangulation(data[-1][:, 0], data[-1][:, 1])
    triang.set_mask(np.hypot(data[-1][triang.triangles, 0].mean(axis=1),
                             data[-1][triang.triangles, 1].mean(axis=1)) < D/2)

    # plt.figure(1, figsize=(20, 5))
    # t = plt.tricontourf(triang, data[-1][:, 3])
    # plt.axis('equal')
    # plt.show()

    #################### 定义损失函数、优化器以及网络结构 ####################
    L2Loss = nn.MSELoss().cuda()
    Net_model = Net(planes=[3] + 5 * [50] + [5], rho=Rho, miu=Miu).to(device)

    Optimizer_1 = torch.optim.Adam(Net_model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    Optimizer_2 = torch.optim.LBFGS(Net_model.parameters(), lr=0.01, max_iter=20)

    Boundary_epoch = [100000, 200000, 300000]

    Scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(Optimizer_1, milestones=Boundary_epoch, gamma=0.1)
    Scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(Optimizer_2, milestones=Boundary_epoch, gamma=0.1)

    Visual = matplotlib_vision('/', field_name=('p', 'u', 'v'), input_name=('x', 'y'))

    ################################### 训练 #####################################
    star_time = time.time()
    log_loss = []
    epoch_start = 0;
    """load a pre-trained model"""
    # checkpoint = torch.load(work_path + '\\latest_model.pth')
    # epoch_start = checkpoint['epoch']
    # Net_model.loadmodel(work_path + '\\latest_model.pth')
    #

    for epoch in range(epoch_start, Boundary_epoch[-1]):

        #如果GPU内存不充足，可以分批次进行训练


        if epoch < 300000:

            iter = 1
            for i in range(iter):
                data_itr = list(map(lambda x: x[i * int(x.shape[0] / iter):(i + 1) * int(x.shape[0] / iter)], data))
                input = data_itr[0]
                input = torch.tensor(input[:, :3], dtype=torch.float32).to(device)
                BCs = (data_itr[1], data_itr[2], data_itr[3], data_itr[5])  ## 边界数据
                field = data_itr[4]  ##检测的流场点
                train(input, BCs, field, Net_model, L2Loss, Optimizer_1, Scheduler_1, log_loss)
            learning_rate = Optimizer_1.state_dict()['param_groups'][0]['lr']
        if epoch >= 300000:
            iter = 1
            for i in range(iter):
                data_itr = list(map(lambda x: x[i * int(x.shape[0] / iter):(i + 1) * int(x.shape[0] / iter)], data))
                input = data_itr[0]
                input = torch.tensor(input[:, :3], dtype=torch.float32).to(device)
                BCs = (data_itr[1], data_itr[2], data_itr[3], data_itr[5])  ## 边界数据
                field = data_itr[4]  ##检测的流场点
                train(input, BCs, field, Net_model, L2Loss, Optimizer_2, Scheduler_2, log_loss)
            learning_rate = Optimizer_2.state_dict()['param_groups'][0]['lr']

        if epoch > 0 and epoch % 500 == 0:
            print('epoch: {:6d}, lr: {:.1e}, cost: {:.2e}, dat_loss: {:.2e}, eqs_loss: {:.2e}, bcs_loss: {:.2e}'.
                  format(epoch, learning_rate, time.time() - star_time,
                         log_loss[-1][-1], log_loss[-1][0], log_loss[-1][1],))
            star_time = time.time()

            # 损失曲线
            plt.figure(1, figsize=(15, 5))
            plt.clf()
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'bcs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
            plt.savefig(os.path.join(work_path, 'log_loss.svg'))

            # 详细的损失曲线
            plt.figure(2, figsize=(15, 10))
            plt.clf()
            plt.subplot(211)
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            plt.subplot(212)
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'bcs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'wall_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'in_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'out_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 5], 'ini_loss')
            plt.savefig(os.path.join(work_path, 'detail_loss.svg'))


            # 根据模型预测流场， 若有真实场，则与真实场对比
            input_visual_p = torch.tensor(data[-1][..., :3], dtype=torch.float32)  # 取初场的空间坐标
            input_visual_p[:, -1] = input_visual_p[:, -1]    # 时间取最大
            field_visual_p, _ = inference(input_visual_p, Net_model)
            field_visual_t = data[-1][..., 3:]
            field_visual_p = field_visual_p.cpu().numpy()[..., 0:3]
            # field_visual_t = field_visual_p

            plt.figure(3, figsize=(30, 8))
            plt.clf()
            Visual.plot_fields_tr(field_visual_t, field_visual_p, input_visual_p.detach().cpu().numpy(), triang)
            # plt.savefig(res_path + 'field_' + str(t) + '-' + str(epoch) + '.jpg')
            plt.savefig(os.path.join(work_path, 'global_' + str(epoch) + '.jpg'), dpi=200)
            plt.savefig(os.path.join(work_path, 'global_now.jpg'))

            torch.save({'epoch': epoch, 'model': Net_model.state_dict(), }, os.path.join(work_path, 'latest_model4.pth'))
