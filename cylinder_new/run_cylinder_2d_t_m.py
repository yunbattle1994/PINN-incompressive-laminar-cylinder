import h5py
import numpy as np
import torch
import torch.nn as nn
from process_data import data_norm, data_sampler
from basic_model import gradients, DeepModel_single
import visual_data
import matplotlib.pyplot as plt
import time
import os

pi = np.pi
# 模型建立
Re = 250.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def read_data():
    data = h5py.File('data\\cyl_Re250.mat', 'r')

    nodes = np.array(data['grids_']).squeeze().transpose((3, 2, 1, 0)) # [Nx, Ny, Nf]
    field = np.array(data['fields_']).squeeze().transpose((3, 2, 1, 0)) # [Nt, Nx, Ny, Nf]
    times = np.array(data['dynamics_']).squeeze().transpose((1, 0))[3::4, (0,)] # (800, 3) -> (200, 1)
    nodes = nodes[0]
    times = times - times[0, 0]

    # plt.figure(1)
    # plt.plot(nodes[:, :, 1], nodes[:, :, 2], 'k.')
    # plt.show()

    return times[:], nodes[:, :64, 1:], field[:, :, :64, :]  # Nx / 2


def BCS_ICS(nodes):
    BCS = []
    ICS = []
    Num_Nodes = nodes.shape[0] * nodes.shape[1]
    Index = np.arange(Num_Nodes).reshape((nodes.shape[0], nodes.shape[1]))

    BCS.append(np.concatenate((Index[:93, -1], Index[284:, -1]), axis=0)) #in  out
    BCS.append(Index[93:284, -1]) #
    BCS.append(Index[:, 0])
    BCS.append(Index[::30, 1])

    ICS.append(Index.reshape(-1))

    INN = np.setdiff1d(ICS[0], np.concatenate(BCS[:-1], axis=0))


    # plt.figure(1)
    # plt.plot(nodes[:, :, 1], nodes[:, :, 2], 'k.')



    return INN, BCS, ICS

class Net(DeepModel_single):
    def __init__(self, planes, data_norm):
        super(Net, self).__init__(planes, data_norm, active=nn.Tanh())
        self.Re = Re

    def equation(self, inn_var, out_var):
        # a = grad(psi.sum(), in_var, create_graph=True, retain_graph=True)[0]
        p, u, v = out_var[:, 0:1], out_var[:, 1:2], out_var[:, 2:3]

        duda = gradients(u, inn_var)
        dudx, dudy, dudt = duda[:, 0:1], duda[:, 1:2], duda[:, 2:3]
        dvda = gradients(v, inn_var)
        dvdx, dvdy, dvdt = dvda[:, 0:1], dvda[:, 1:2], dvda[:, 2:3]
        d2udx2 = gradients(dudx, inn_var)[:, 0:1]
        d2udy2 = gradients(dudy, inn_var)[:, 1:2]
        d2vdx2 = gradients(dvdx, inn_var)[:, 0:1]
        d2vdy2 = gradients(dvdy, inn_var)[:, 1:2]
        dpda = gradients(p, inn_var)
        dpdx, dpdy = dpda[:, 0:1], dpda[:, 1:2]

        eq1 = dudt + (u * dudx + v * dudy) + dpdx - 1 / self.Re * (d2udx2 + d2udy2)
        eq2 = dvdt + (u * dvdx + v * dvdy) + dpdy - 1 / self.Re * (d2vdx2 + d2vdy2)
        eq3 = dudx + dvdy
        eqs = torch.cat((eq1, eq2, eq3), dim=1)
        return eqs


def train(inn_var, BCs, ICs, out_true, model, Loss, optimizer, scheduler, log_loss):

    inn = BCs[0].sampling(Nx=6000, Nt=20); ind_inner = inn.shape[0]
    BC_in = BCs[1].sampling(Nx='all', Nt=20); ind_BC_in = BC_in.shape[0] + ind_inner
    BC_out = BCs[2].sampling(Nx='all', Nt=20); ind_BC_out = BC_out.shape[0] + ind_BC_in
    BC_wall = BCs[3].sampling(Nx='all', Nt=20); ind_BC_wall = BC_wall.shape[0] + ind_BC_out
    BC_meas = BCs[4].sampling(Nx='all', Nt=50); ind_BC_meas = BC_meas.shape[0] + ind_BC_wall
    
    IC_0 = ICs[0].sampling(Nx='all'); ind_IC_0 = IC_0.shape[0] + ind_BC_meas

    inn_var = torch.cat((inn_var[inn], inn_var[BC_in], inn_var[BC_out], inn_var[BC_wall], inn_var[BC_meas], inn_var[IC_0]), dim=0)
    out_true = torch.cat((out_true[inn], out_true[BC_in], out_true[BC_out], out_true[BC_wall], out_true[BC_meas], out_true[IC_0]), dim=0)
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
        ics_loss_0 = Loss(out_var[ind_BC_meas:ind_IC_0, :], out_true[ind_BC_meas:ind_IC_0, :])
        eqs_loss = Loss(res_i[:ind_inner], torch.zeros((ind_inner, 3), dtype=torch.float32).cuda())

        loss_batch = bcs_loss_1 + bcs_loss_2 + bcs_loss_3 + bcs_loss_4 + ics_loss_0 + eqs_loss
        loss_batch.backward()

        data_loss = Loss(out_var, out_true)
        log_loss.append([eqs_loss.item(), bcs_loss_1.item(), bcs_loss_2.item(), bcs_loss_3.item(), bcs_loss_4.item(),
                         ics_loss_0.item(), data_loss.item()])

        return loss_batch

    optimizer.step(closure)
    scheduler.step()

def inference(inn_var, model):

    with torch.no_grad():
        out_pred = model(inn_var)

    return out_pred


if __name__ == '__main__':

    times, nodes, field = read_data()
    INN, BCS, ICS = BCS_ICS(nodes)

    res_path = 'res\\cylinder_2d_t_test\\'
    isCreated = os.path.exists(res_path)

    if not isCreated:
        os.makedirs(res_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Nt, Nx, Ny, Nf = field.shape[0], field.shape[1], field.shape[2], field.shape[3]
    times = np.tile(times[:, None, None, :], (1, Nx, Ny, 1))
    nodes = np.tile(nodes[None, :, :, :], (Nt, 1, 1, 1))
    times = times.reshape(-1, 1)
    nodes = nodes.reshape(-1, 2)
    field = field.reshape(-1, Nf)
    input = np.concatenate((nodes, times), axis=-1)
    input_norm = data_norm(input, method='mean-std')
    field_norm = data_norm(field, method='mean-std')

    input_visual = input.reshape((Nt, Nx, Ny, 3))
    field_visual = field.reshape((Nt, Nx, Ny, Nf))

    # Training Data
    input = torch.tensor(input, dtype=torch.float32)
    field = torch.tensor(field, dtype=torch.float32)

    NumNodes = Nx * Ny
    BC_in = data_sampler(BCS[0], NumNodes, time=Nt)
    BC_out = data_sampler(BCS[1], NumNodes, time=Nt)
    BC_cyl = data_sampler(BCS[2], NumNodes, time=Nt)
    BC_meas = data_sampler(BCS[3], NumNodes, time=Nt)
    IC_cyl = data_sampler(ICS[0], NumNodes, time=0)
    IN_cyl = data_sampler(INN, NumNodes, time=Nt)
    BCs = [IN_cyl, BC_in, BC_out, BC_cyl,  BC_meas]
    ICs = [IC_cyl,]


    L1Loss = nn.L1Loss().cuda()
    HBLoss = nn.SmoothL1Loss().cuda()
    L2Loss = nn.MSELoss().cuda()

    Net_model = Net(planes=[3, 64, 64, 64, 64, 64, 64, 3], data_norm=(input_norm, field_norm)).to(device)
    Optimizer0 = torch.optim.SGD(Net_model.parameters(), lr=0.05, momentum=0.9, )
    Optimizer1 = torch.optim.Adam(Net_model.parameters(), lr=0.001, betas=(0.7, 0.9))
    Optimizer2 = torch.optim.LBFGS(Net_model.parameters(), lr=1, max_iter=100, history_size=50,)
    Boundary_epoch = [300000, 400000, 500000]
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
        train(input, BCs, ICs, field, Net_model, L2Loss, Optimizer1, Scheduler, log_loss)



        if iter > 0 and iter % 500 == 0:
            print('iter: {:6d}, lr: {:.1e}, cost: {:.2f}, dat_loss: {:.2e} \n'
                  'eqs_loss: {:.2e}, BCS_loss_in: {:.2e}, BCS_loss_out: {:.2e}, '
                  'BCS_loss_wall: {:.2e}, BCS_loss_meas: {:.2e}, ICS_loss_0: {:.2e}'.
                  format(iter, learning_rate, time.time() - star_time, log_loss[-1][-1],
                         log_loss[-1][0], log_loss[-1][1], log_loss[-1][2],
                         log_loss[-1][3], log_loss[-1][4], log_loss[-1][5]))

            plt.figure(1, figsize=(20, 10))
            plt.clf()
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'BCS_loss_in')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'BCS_loss_out')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'BCS_loss_wall')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'BCS_loss_meas')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 5], 'ICS_loss_0')
            plt.savefig(res_path + 'log_loss.svg')

            star_time = time.time()

        if iter > 0 and iter % 2000 == 0:
            input_visual_p = torch.tensor(input_visual[:100:10], dtype=torch.float32)
            field_visual_p = inference(input_visual_p.to(device), Net_model)
            field_visual_t = field_visual[:100:10]
            field_visual_p = field_visual_p.cpu().numpy()

            for t in range(field_visual_p.shape[0]):
                plt.figure(2, figsize=(30, 12))
                plt.clf()
                Visual.plot_fields_ms(field_visual_t[t], field_visual_p[t], input_visual_p[0, :, :, :2],
                                      cmin_max=[[-5, -4], [6, 4]])
                # plt.savefig(res_path + 'field_' + str(t) + '-' + str(iter) + '.jpg')
                plt.savefig(res_path + 'loca_' + str(t) + '.jpg')

                plt.figure(3, figsize=(30, 20))
                plt.clf()
                Visual.plot_fields_ms(field_visual_t[t], field_visual_p[t], input_visual_p[0, :, :, :2].numpy())
                plt.savefig(res_path + 'full_' + str(t) + '.jpg', dpi=200)

                torch.save({'epoch': iter, 'model': Net_model.state_dict(), }, res_path + 'latest_model.pth')