import h5py
import numpy as np
import torch
import torch.nn as nn
from process_data import data_norm, data_sampler
from basic_model import gradients, DeepModel_single, DeepModel_multi
import visual_data
import matplotlib.pyplot as plt
import time
import os

pi = np.pi
# 模型建立
Re = 10.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def read_data():
    data = h5py.File('data\\cyl_Re10.mat', 'r')
    nodes = np.array(data['grids_']).squeeze().transpose((2, 1, 0)) # [Nx, Ny, Nf]
    field = np.array(data['fields_']).squeeze().transpose((2, 1, 0)) # [Nx, Ny, Nf]

    return nodes[:, :, 1:].astype(np.float32), field[:, :, :].astype(np.float32)  # Nx / 2


def BCS_ICS(nodes):
    BCS = []
    ICS = []
    Num_Nodes = nodes.shape[0] * nodes.shape[1]
    Index = np.arange(Num_Nodes).reshape((nodes.shape[0], nodes.shape[1]))

    BCS.append(np.concatenate((Index[:93, -1], Index[284:, -1]), axis=0))
    BCS.append(Index[98:274, -1])
    BCS.append(Index[:, 0])
    BCS.append(Index[(0,), 5])

    INN = np.setdiff1d(Index.flatten(), np.concatenate(BCS[:-1], axis=0))

    return INN, BCS

# class Net(DeepModel_multi):
class Net(DeepModel_single):
    def __init__(self, planes, data_norm):
        super(Net, self).__init__(planes, data_norm, active=nn.Tanh())
        self.Re = Re

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
        eq1 = (u * dudx + v * dudy) + dpdx - 1 / self.Re * (d2udx2 + d2udy2)
        eq2 = (u * dvdx + v * dvdy) + dpdy - 1 / self.Re * (d2vdx2 + d2vdy2)
        eq3 = dudx + dvdy
        eqs = torch.cat((eq1, eq2, eq3), dim=-1)
        return eqs


def train(inn_var, BCs, ICs, out_true, model, Loss, optimizer, scheduler, log_loss):

    inn = BCs[0].sampling(Nx='all'); ind_inner = inn.shape[0]
    BC_in = BCs[1].sampling(Nx='all'); ind_BC_in = BC_in.shape[0] + ind_inner
    BC_out = BCs[2].sampling(Nx='all'); ind_BC_out = BC_out.shape[0] + ind_BC_in
    BC_wall = BCs[3].sampling(Nx='all'); ind_BC_wall = BC_wall.shape[0] + ind_BC_out
    BC_meas = BCs[4].sampling(Nx='all'); ind_BC_meas = BC_meas.shape[0] + ind_BC_wall

    inn_var = torch.cat((inn_var[inn], inn_var[BC_in], inn_var[BC_out], inn_var[BC_wall], inn_var[BC_meas]), dim=0)
    out_true = torch.cat((out_true[inn], out_true[BC_in], out_true[BC_out], out_true[BC_wall], out_true[BC_meas]), dim=0)
    inn_var = inn_var.cuda()
    out_true = out_true.cuda()

    def closure():
        inn_var.requires_grad_(True)
        optimizer.zero_grad()
        out_var = model(inn_var)
        res_i = model.equation(inn_var, out_var)

        bcs_loss_1 = Loss(out_var[ind_inner:ind_BC_in, 1:], out_true[ind_inner:ind_BC_in, 1:])
        bcs_loss_2 = Loss(out_var[ind_BC_in:ind_BC_out, 0], out_true[ind_BC_in:ind_BC_out, 0])
        bcs_loss_3 = Loss(out_var[ind_BC_out:ind_BC_wall, 1:], out_true[ind_BC_out:ind_BC_wall, 1:])
        bcs_loss_4 = Loss(out_var[ind_BC_wall:ind_BC_meas, 0], out_true[ind_BC_wall:ind_BC_meas, 0])
        eqs_loss_1 = Loss(res_i[:, 1:2], torch.zeros((res_i.shape[0], 1), dtype=torch.float32).cuda())
        eqs_loss_2 = Loss(res_i[:, 2:3], torch.zeros((res_i.shape[0], 1), dtype=torch.float32).cuda())
        eqs_loss_0 = Loss(res_i[:, 0:1], torch.zeros((res_i.shape[0], 1), dtype=torch.float32).cuda())


        loss_BCs = bcs_loss_1 + bcs_loss_2 + bcs_loss_3 #+ bcs_loss_4
        loss_Eqs =  eqs_loss_1 + eqs_loss_2 + eqs_loss_0
        loss_batch = loss_BCs * 100. + loss_Eqs
        loss_BCs.backward()

        data_loss = Loss(out_var, out_true)
        log_loss.append([eqs_loss_0.item(), eqs_loss_1.item(), eqs_loss_2.item(),
                         bcs_loss_1.item(), bcs_loss_2.item(), bcs_loss_3.item(), bcs_loss_4.item(),
                         data_loss.item()])

        return loss_batch

    optimizer.step(closure)
    scheduler.step()

def inference(inn_var, model):
    inn_var.requires_grad_(True)
    out_pred = model(inn_var)
    equation = model.equation(inn_var, out_pred)
    return out_pred.detach().cpu(), equation.detach().cpu()


if __name__ == '__main__':

    nodes, field = read_data()
    INN, BCS = BCS_ICS(nodes)

    res_path = 'res\\cylinder_2d_measure0\\'
    isCreated = os.path.exists(res_path)
    if not isCreated:
        os.makedirs(res_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Nx, Ny, Nf, Ni = field.shape[0], field.shape[1], field.shape[2], nodes.shape[-1]
    nodes = nodes.reshape(-1, Ni)
    field = field.reshape(-1, Nf)
    input_norm = data_norm(nodes, method='mean-std')
    field_norm = data_norm(field, method='mean-std')

    input_visual = nodes.reshape((Nx, Ny, Ni))
    field_visual = field.reshape((Nx, Ny, Nf))

    # Training Data
    input = torch.tensor(nodes, dtype=torch.float32)
    field = torch.tensor(field, dtype=torch.float32)

    NumNodes = Nx * Ny
    BC_in = data_sampler(BCS[0], NumNodes, time=0)
    BC_out = data_sampler(BCS[1], NumNodes, time=0)
    BC_cyl = data_sampler(BCS[2], NumNodes, time=0)
    BC_meas = data_sampler(BCS[3], NumNodes, time=0)
    IN_cyl = data_sampler(INN, NumNodes, time=0)
    BCs = [IN_cyl, BC_in, BC_out, BC_cyl, BC_meas]

    L1Loss = nn.L1Loss().cuda()
    HBLoss = nn.SmoothL1Loss().cuda()
    L2Loss = nn.MSELoss().cuda()

    Net_model = Net(planes=[Ni, 64, 64, 64, 64, 64, Nf], data_norm=(input_norm, field_norm)).to(device)
    Optimizer0 = torch.optim.SGD(Net_model.parameters(), lr=0.05, momentum=0.9, )
    Optimizer1 = torch.optim.Adam(Net_model.parameters(), lr=0.0005, betas=(0.7, 0.9))
    Optimizer2 = torch.optim.LBFGS(Net_model.parameters(), lr=1, max_iter=100, history_size=50,)
    Boundary_epoch = [100000, 150000, 180000]
    Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer1, milestones=Boundary_epoch, gamma=0.1)
    Visual = visual_data.matplotlib_vision('/', field_name=('p', 'u', 'v'), input_name=('x', 'y'))
    #
    star_time = time.time()
    log_loss = []

    """load a pre-trained model"""
    # Net_model.loadmodel(res_path + 'latest_model.pth')
    # Training
    for iter in range(Boundary_epoch[-1]):

        learning_rate = Optimizer1.state_dict()['param_groups'][0]['lr']
        train(input, BCs, None, field, Net_model, L2Loss, Optimizer1, Scheduler, log_loss)

        if iter > 0 and iter % 2000 == 0:
            print('iter: {:6d}, lr: {:.1e}, cost: {:.2f} \n'
                  'mass_loss: {:.2e}, fx_loss: {:.2e}, fy_loss: {:.2e}, dat_loss: {:.2e}, '
                  'BCS_loss_in: {:.2e}, BCS_loss_out: {:.2e}, BCS_loss_wall: {:.2e}, BCS_loss_meas: {:.2e}'.
                  format(iter, learning_rate, time.time()-star_time,
                         log_loss[-1][0], log_loss[-1][1], log_loss[-1][2], log_loss[-1][-1],
                         log_loss[-1][3], log_loss[-1][4], log_loss[-1][5], log_loss[-1][6]))

            plt.figure(1, figsize=(20, 10))
            plt.clf()
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'mass_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'force_x_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'force_y_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'BCS_loss_in')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'BCS_loss_out')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 5], 'BCS_loss_wall')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 6], 'BCS_loss_measure')
            plt.savefig(res_path + 'log_loss.svg')

            star_time = time.time()

        if iter > 0 and iter % 2000 == 0:
            input_visual_p = torch.tensor(input_visual, dtype=torch.float32)
            field_visual_p, equation_visual_p = inference(input_visual_p.to(device), Net_model)
            field_visual_t = field_visual
            field_visual_p = field_visual_p.cpu().numpy()

            plt.figure(2, figsize=(30, 12))
            plt.clf()
            Visual.plot_fields_ms(field_visual_t, field_visual_p, input_visual_p[:, :, :],
                                  cmin_max=[[-5, -4], [6, 4]])
            # plt.savefig(res_path + 'field_' + str(t) + '-' + str(iter) + '.jpg')
            plt.savefig(res_path + 'local_' + str(iter) + '.jpg', dpi=200)

            plt.figure(3, figsize=(30, 20))
            plt.clf()
            Visual.plot_fields_ms(field_visual_t, field_visual_p, np.array(input_visual_p))
            plt.savefig(res_path + 'full_' + str(iter) + '.jpg', dpi=200)


            plt.figure(2, figsize=(30, 12))
            plt.clf()
            Visual.plot_fields_ms(equation_visual_p[:, 1:-1, :].numpy(), equation_visual_p[:, 1:-1, :].numpy(), input_visual_p[:, 1:-1, :].numpy(),
                                  cmin_max=[[-5, -4], [6, 4]])
            # plt.savefig(res_path + 'field_' + str(t) + '-' + str(iter) + '.jpg')
            plt.savefig(res_path + 'local_eq_' + str(iter) + '.jpg', dpi=200)

            plt.figure(3, figsize=(30, 20))
            plt.clf()
            Visual.plot_fields_ms(equation_visual_p[:, 1:-1, :].numpy(), equation_visual_p[:, 1:-1, :].numpy(), input_visual_p[:, 1:-1, :].numpy())
            plt.savefig(res_path + 'full_eq_' + str(iter) + '.jpg', dpi=200)


            torch.save({'epoch': iter, 'model': Net_model.state_dict(), }, res_path + 'latest_model.pth')