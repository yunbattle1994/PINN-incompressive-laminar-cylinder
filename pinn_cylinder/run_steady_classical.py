import matplotlib.tri
import numpy as np
import torch
import torch.nn as nn
from basic_model import DeepModel_single, DeepModel_multi, gradients
from visual_data import matplotlib_vision

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


    def equation(self, inn_var, out_var):
        p, u, v = out_var[..., 0:1], out_var[..., 1:2], out_var[..., 2:3]

        duda = gradients(u, inn_var)
        dudx, dudy = duda[..., 0:1], duda[..., 1:2]
        dvda = gradients(v, inn_var)
        dvdx, dvdy = dvda[..., 0:1], dvda[..., 1:2]
        d2udx2 = gradients(dudx, inn_var)[..., 0:1]
        d2udy2 = gradients(dudy, inn_var)[..., 1:2]
        d2vdx2 = gradients(dvdx, inn_var)[..., 0:1]
        d2vdy2 = gradients(dvdy, inn_var)[..., 1:2]
        dpda = gradients(p, inn_var)
        dpdx, dpdy = dpda[..., 0:1], dpda[..., 1:2]

        eq0 = dudx + dvdy
        eq1 = self.rho * (u * dudx + v * dudy) + dpdx - self.miu * (d2udx2 + d2udy2)
        eq2 = self.rho * (u * dvdx + v * dvdy) + dpdy - self.miu * (d2vdx2 + d2vdy2)
        eqs = torch.cat((eq0, eq1, eq2), dim=-1)
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
    BCS.append(Index[np.abs((nodes[:, 0]-0.2)**2 + (nodes[:, 1]-0.2)**2 - (D/2)**2) < 1e-7])  # cylinder wall

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


################################## 单次训练步骤  ##################################
def train(inn_var, BCs, out_true, model, Loss, optimizer, scheduler, log_loss):

    BC_in = BCs[0]
    BC_out = BCs[1]
    BC_bot = BCs[2]
    BC_top = BCs[3]
    BC_wall = BCs[4]

    def closure():
        inn_var.requires_grad_(True)
        optimizer.zero_grad()
        out_var = model(inn_var)
        # out_var = model.output_transform(inn_var, out_var)
        res_i, _ = model.equation(inn_var, out_var)
        out_var = out_var[..., 0:3]
        y_in = inn_var.detach()[BC_in, 1:2]

        bcs_loss_in = Loss(out_var[BC_in, 1:], torch.cat((4*U_max*y_in*((Box[-1] - Box[1])-y_in)/((Box[-1] - Box[1])**2), 0*y_in), dim=-1))
        bcs_loss_out = (out_var[BC_out, 0] ** 2).mean()
        bcs_loss_wall = (out_var[BC_wall, 1:] ** 2).mean()
        bcs_loss_top = (out_var[BC_top, 1:]**2).mean()
        bcs_loss_bot = (out_var[BC_bot, 1:]**2).mean()
        bcs_loss = bcs_loss_in + bcs_loss_out + bcs_loss_wall + bcs_loss_top + bcs_loss_bot

        eqs_loss = (res_i ** 2).mean()

        loss_batch = bcs_loss * 2. + eqs_loss
        loss_batch.backward()

        # data_loss = Loss(out_var, out_true)
        log_loss.append([eqs_loss.item(), bcs_loss.item(),
                         bcs_loss_wall.item(), bcs_loss_top.item(), bcs_loss_bot.item(), bcs_loss_in.item(), bcs_loss_out.item(),
                         0])

        return loss_batch

    optimizer.step(closure)
    scheduler.step()


################################## 预测  ##################################
def inference(inn_var, model):
    inn_var = inn_var.cuda()
    inn_var.requires_grad_(True)
    out_var = model(inn_var)
    # out_var = model.output_transform(inn_var, out_var)
    # equation, _ = model.equation(inn_var, out_var)
    return out_var.detach().cpu(), 0 #, equation.detach().cpu()


if __name__ == '__main__':

    name = 'steady-cylinder-2d-old'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #################### 定义问题相关参数 ####################
    U_max = 1.0 # 入口流速的最大值
    Rho, Miu, D = 1.0, 0.02, 0.1
    Box = [0, 0, 1.1, 0.41]  # 矩形流域
    
    
    #################### 读入数据 ####################
    data = read_data(steady=True)
    data = list(map(np.random.permutation, data)) # np.random.shuffle & random.shuffle 返回None,此外， python 3 中map返回的是迭代器
    XY_c = np.concatenate(data[:-1], 0)
    fields_fluent = data[-1]
    BCs = BCS_ICS(XY_c[:, :2], Box)
    input = torch.tensor(XY_c[:, :2], dtype=torch.float32).to(device)
    field = torch.tensor(fields_fluent, dtype=torch.float32)
    # 采用三角形 对非结构化网格建立节点连接关系
    triang = matplotlib.tri.Triangulation(fields_fluent[:, 0], fields_fluent[:, 1])
    triang.set_mask(np.hypot(fields_fluent[triang.triangles, 0].mean(axis=1) - 0.2,
                             fields_fluent[triang.triangles, 1].mean(axis=1) - 0.2) < D/2)
    # plt.figure(1, figsize=(20, 5))
    # t = plt.tricontourf(triang, fields_fluent[:, 2])
    # plt.axis('equal')
    # plt.show()

    #################### 定义损失函数、优化器以及网络结构 ####################
    L2Loss = nn.MSELoss().cuda()
    Net_model = Net(planes=[2] + 8 * [40] + [5], rho=Rho, miu=Miu).to(device)
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    Boundary_epoch = [200000, 250000, 300000]
    Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=Boundary_epoch, gamma=0.1)
    Visual = matplotlib_vision('/', field_name=('p', 'u', 'v'), input_name=('x', 'y'))


    ################################### 训练 #####################################
    star_time = time.time()
    log_loss = []
    """load a pre-trained model"""
    # Net_model.loadmodel(res_path + 'latest_model.pth')
    # Training
    for epoch in range(Boundary_epoch[-1]):

        #如果GPU内存不充足，可以分批次进行训练
        #iter = 10
        # for i in range(iter):
        #     XY_c = np.concatenate(list(map(lambda x: x[i*int(x.shape[0]/iter):(i+1)*int(x.shape[0]/iter)], data[:-1])), 0)
        #     BCs = BCS_ICS(XY_c, Box)
        #     input = torch.tensor(XY_c[:, :3], dtype=torch.float32).to(device)
        #     train(input, BCs, field, Net_model, L2Loss, Optimizer, Scheduler, log_loss)


        learning_rate = Optimizer.state_dict()['param_groups'][0]['lr']
        train(input, BCs, field, Net_model, L2Loss, Optimizer, Scheduler, log_loss)

        if epoch > 0 and epoch % 2000 == 0:
            print('epoch: {:6d}, lr: {:.1e}, cost: {:.2e}, dat_loss: {:.2e}, eqs_loss: {:.2e}, bcs_loss: {:.2e}'.
                  format(epoch, learning_rate, time.time() - star_time,
                         log_loss[-1][-1], log_loss[-1][0], log_loss[-1][1],))

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
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'top_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'bot_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 5], 'in_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 6], 'out_loss')
            plt.savefig(os.path.join(work_path, 'detail_loss.svg'))

            # 根据模型预测流场， 若有真实场，则与真实场对比
            input_visual_p = field[..., :2]
            field_visual_p, _ = inference(input_visual_p, Net_model)
            field_visual_t = field[..., 2:].cpu().numpy()
            field_visual_p = field_visual_p.cpu().numpy()[..., 0:3]

            plt.figure(3, figsize=(30, 8))
            plt.clf()
            Visual.plot_fields_tr(field_visual_t, field_visual_p, input_visual_p.detach().cpu().numpy(), triang)
            # plt.savefig(res_path + 'field_' + str(t) + '-' + str(epoch) + '.jpg')
            plt.savefig(os.path.join(work_path, 'global_' + str(epoch) + '.jpg'), dpi=200)
            plt.savefig(os.path.join(work_path, 'global_now.jpg'))

            torch.save({'epoch': epoch, 'model': Net_model.state_dict(), }, os.path.join(work_path, 'latest_model.pth'))
