import torch
import torch.nn as nn
from torch.autograd import grad
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def gradients(y, x):
    """
    计算y对x的一阶导数，dydx
    :param y: torch tensor，网络输出，shape（...×N）
    :param x: torch tensor，网络输入，shape（...×M）
    :return dydx: torch tensor，网络输入，shape（...M×N）如果N=1，则最后一个维度被缩并
    """
    return torch.stack([grad([y[..., i].sum()], [x], retain_graph=True, create_graph=True)[0]
                            for i in range(y.size(-1))], dim=-1).squeeze(-1)



def jacobians(u, x):

    return torch.autograd.functional.jacobian(u, x, create_graph=True)[0]



class DeepModel_multi(nn.Module):
    def __init__(self, planes, active=nn.GELU()):
        """
        :param planes: list，[M,...,N],全连接神经网络的输入维度M，每个隐含层维度，输出维度N
        :param active: 激活函数
               与single区别，multi采用N个全连接层,每个全连接层输出维度为1
        """

        super(DeepModel_multi, self).__init__()
        self.planes = planes
        self.active = active

        self.layers = nn.ModuleList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1]))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1))
            self.layers.append(nn.Sequential(*layer))
        self.apply(initialize_weights)

    def forward(self, in_var):
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        return torch.cat(y, dim=-1)

    def loadmodel(self, File):

        try:
            checkpoint = torch.load(File)
            self.load_state_dict(checkpoint['model'])        # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at epoch " + str(start_epoch))
        except:
            print("load model failed！ start a new model.")

    def equation(self, **kwargs):
        return 0

class DeepModel_single(nn.Module):
    def __init__(self, planes, active=nn.GELU()):
        """
        :param planes: list，[M,...,N],全连接神经网络的输入维度，每个隐含层维度，输出维度
        :param active: 激活函数
                       与multi，single采用1个全连接层,该全连接层输出维度为N
        """
        super(DeepModel_single, self).__init__()
        self.planes = planes
        self.active = active

        self.layers = nn.ModuleList()
        for i in range(len(self.planes)-2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))

        self.layers = nn.Sequential(*self.layers)
        self.apply(initialize_weights)

    def forward(self, in_var):
        out_var = self.layers(in_var)
        return out_var

    def loadmodel(self, File):

        try:
            checkpoint = torch.load(File)
            self.load_state_dict(checkpoint['model'])        # 从字典中依次读取
            start_epoch = len(checkpoint['log_loss'])
            print("load start epoch at epoch " + str(start_epoch))
        except:
            print("load model failed！ start a new model.")


    def equation(self, **kwargs):
        return 0


def adaptive_weights(loss_list, model):
    max_grad_list = []
    avg_grad_list = []
    for i in range(len(loss_list)-1):
        avg_grad_list.append([])

    for name, param in model.named_parameters():
        if 'bias' not in name:
            max_grad_list.append(gradients(loss_list[0], param).abs().max().detach())
            for k, loss in enumerate(loss_list[1:]):
                avg_grad_list[k].append(gradients(loss, param).abs().mean().detach())

    avg_grad = torch.tensor(avg_grad_list).mean()
    max_grad = torch.tensor(max_grad_list).max()

    return max_grad / avg_grad


def causal_weights_loss(res):
    tol = 100.
    Nt = res.shape[0]
    M_t = torch.triu(torch.ones((Nt, Nt), device=res.device), diagonal=1).T
    L_t = torch.mean(res**2, dim=1)
    W_t = torch.exp(-tol*(M_t @ L_t.detach()))
    loss = torch.mean(W_t * L_t)
    return loss, W_t

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.xavier_uniform_(m.weight, gain=1)
            m.bias.data.zero_()