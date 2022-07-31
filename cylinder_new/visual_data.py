import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sbn
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
import matplotlib.cm as cm


class matplotlib_vision(object):

    def __init__(self, log_dir, input_name=('x'), field_name=('f',)):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        # sbn.set_style('ticks')
        # sbn.set()

        self.field_name = field_name
        self.input_name = input_name
        self._cbs = [None] * len(self.field_name) * 3

        gs = gridspec.GridSpec(1, 1)
        gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.5, hspace=0.7)
        gs_dict = {key: value for key, value in gs.__dict__.items() if key in gs._AllowedKeys}
        self.fig, self.axes = plt.subplots(len(self.field_name), 3, gridspec_kw=gs_dict, num=100, figsize=(30, 20))
        self.font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30}

    def plot_loss(self, x, y, label, title=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label)
        plt.semilogy()
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper right", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('loss value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)

    def plot_value(self, x, y, label, title=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)

        plt.plot(x, y, label=label)
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper right", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('pred value', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)
        # plt.pause(0.001)

    def plot_scatter(self, true, pred, axis=0, title=None):
        # sbn.set(color_codes=True)

        plt.scatter(np.arange(true.shape[0]), true, marker='*')
        plt.scatter(np.arange(true.shape[0]), pred, marker='.')

        plt.ylabel('target value', self.font)
        plt.xlabel('samples', self.font)
        plt.xticks(fontproperties='Times New Roman', size=25)
        plt.yticks(fontproperties='Times New Roman', size=25)
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)

    def plot_regression(self, true, pred, axis=0, title=None):
        # 所有功率预测误差与真实结果的回归直线
        # sbn.set(color_codes=True)

        max_value = max(true)  # math.ceil(max(true)/100)*100
        min_value = min(true)  # math.floor(min(true)/100)*100
        split_value = np.linspace(min_value, max_value, 11)

        split_dict = {}
        split_label = np.zeros(len(true), np.int)
        for i in range(len(split_value)):
            split_dict[i] = str(split_value[i])
            index = true >= split_value[i]
            split_label[index] = i + 1

        plt.scatter(true, pred, marker='.')

        plt.plot([min_value, max_value], [min_value, max_value], 'r-', linewidth=5.0)
        plt.fill_between([min_value, max_value], [0.95 * min_value, 0.95 * max_value],
                         [1.05 * min_value, 1.05 * max_value],
                         alpha=0.2, color='b')

        # plt.ylim((min_value, max_value))
        plt.xlim((min_value, max_value))
        plt.ylabel('pred value', self.font)
        plt.xlabel('real value', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.grid(True)  # 添加网格
        plt.title(title, self.font)
        # plt.ylim((-0.2, 0.2))
        # plt.pause(0.001)

    def plot_error(self, error, title=None):
        # sbn.set_color_codes()
        error = pd.DataFrame(error) * 100
        sbn.distplot(error, bins=20, norm_hist=True, rug=True, fit=stats.norm, kde=False,
                     rug_kws={"color": "g"}, fit_kws={"color": "r", "lw": 3}, hist_kws={"color": "b"})
        # plt.xlim([-1, 1])
        plt.xlabel("predicted relative error / %", self.font)
        plt.ylabel('distribution density', self.font)
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.grid(True)
        # plt.legend()
        plt.title(title, self.font)

    def plot_fields1d(self, curve, true, pred, name):
        plt.plot(curve, true)
        plt.plot(curve, pred)
        plt.xlabel('x coordinate', self.font)
        plt.ylabel(name, self.font)
        plt.yticks(fontproperties='Times New Roman', size=20)
        plt.xticks(fontproperties='Times New Roman', size=20)

    def plot_optimization(self, max_Ys, title=None):
        mean = max_Ys.mean(axis=1)  # 计算开盘价的5期平均移动
        std = max_Ys.std(axis=1)
        plt.plot(range(100), mean, 'r-', linewidth=5.0, label="mean_efficiency")  # 50条数据不能错

        plt.fill_between(range(100), mean - std, mean + std, alpha=0.5, color='b', label="confidence")
        plt.fill_between(range(100), max_Ys.min(axis=1), max_Ys.max(axis=1), alpha=0.2, color='g', label="min-max")

        plt.grid(True)
        plt.legend(loc="lower right", prop=self.font)
        plt.xlabel('iterations', self.font)
        plt.ylabel('efficiency / %', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.title(title, self.font)

    def plot_fields_2d(self, fields_true, fields_pred, fmin_max=None):

        plt.clf()

        Num_fields = fields_true.shape[-1]
        for fi in range(Num_fields):
            field_true, field_pred = fields_true[:, :, fi], fields_pred[:, :, fi]

            if fmin_max == None:
                fmin, fmax = fields_true.min(axis=(0, 1)), fields_true.max(axis=(0, 1))
            else:
                fmin, fmax = fmin_max[0], fmin_max[1]

            plt.subplot(3, Num_fields, 3 * fi + 1)
            plt.imshow(field_true, cmap='RdYlBu_r', aspect='auto', interpolation="spline16", )
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            plt.rcParams['font.family'] = 'Times New Roman'
            cb.set_label(self.fields_name[fi], rotation=0, fontdict=self.font, y=1.08)
            plt.grid(False)
            # plt.xlabel('foils', self.font)
            # plt.ylabel('time step', self.font)
            plt.yticks(fontproperties='Times New Roman', size=20)
            plt.xticks(fontproperties='Times New Roman', size=20)
            plt.title('True field $' + self.fields_name[fi] + '(t,x,y)$' + '$\mathregular{*10^3}$', fontsize=20)

            plt.subplot(3, Num_fields, 3 * fi + 2)
            plt.imshow(field_pred, cmap='RdYlBu_r', aspect='auto', interpolation="spline16", )
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            plt.rcParams['font.family'] = 'Times New Roman'
            cb.set_label(self.fields_name[fi], rotation=0, fontdict=self.font, y=1.08)
            plt.grid(False)
            # plt.xlabel('foils', self.font)
            # plt.ylabel('time step', self.font)
            plt.yticks(fontproperties='Times New Roman', size=20)
            plt.xticks(fontproperties='Times New Roman', size=20)
            plt.title('True field $' + self.fields_name[fi] + '(t,x,y)$' + '$\mathregular{*10^3}$', fontsize=20)

            plt.subplot(3, Num_fields, 3 * fi + 3)
            plt.imshow(field_pred - field_true, aspect='auto', cmap='coolwarm', interpolation="spline16", )
            plt.clim(vmin=-max(abs(fmin[fi]), abs(fmax[fi])), vmax=max(abs(fmin[fi]), abs(fmax[fi])))
            cb = plt.colorbar()
            plt.rcParams['font.family'] = 'Times New Roman'
            cb.set_label(self.fields_name[fi], rotation=0, fontdict=self.font, y=1.08)
            plt.grid(False)
            # plt.xlabel('foils', self.font)
            # plt.ylabel('time step', self.font)
            plt.yticks(fontproperties='Times New Roman', size=20)
            plt.xticks(fontproperties='Times New Roman', size=20)
            plt.title('True field $' + self.fields_name[fi] + '(t,x,y)$' + '$\mathregular{*10^3}$', fontsize=20)

    def plot_fields_tri(self, out_true, out_pred, coord, cell, cmin_max=None, fmin_max=None, cmap='jet', field_name=None):

        plt.clf()
        Num_fields = out_true.shape[-1]
        if fmin_max == None:
            fmin, fmax = out_true.min(axis=(0,)), out_true.max(axis=(0,))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if field_name == None:
            field_name = self.field_name

        x_pos = coord[:, 0]
        y_pos = coord[:, 1]
        ############################# Plotting ###############################
        for fi in range(Num_fields):
            plt.rcParams['font.size'] = 20
            triObj = tri.Triangulation(x_pos, y_pos, triangles=cell)  # 生成指定拓扑结构的三角形剖分.

            Num_levels = 20
            # plt.triplot(triObj, lw=0.5, color='white')

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########      Exact f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 0 * Num_fields + fi + 1)
            levels = np.arange(out_true.min(), out_true.max(), 0.05)
            plt.tricontourf(triObj, out_true[:, fi], Num_levels, cmap=cmap)
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            # cb.set_label('value', rotation=0, fontdict=self.font, y=1.08)
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('True field $' + field_name[fi] + '$' + '', fontsize=30)

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########     Learned f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 1 * Num_fields + fi + 1)
            # levels = np.arange(out_true.min(), out_true.max(), 0.05)
            plt.tricontourf(triObj, out_pred[:, fi], Num_levels, cmap=cmap)
            cb = plt.colorbar()
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('Pred field $' + field_name[fi] + '$' + '', fontsize=30)

            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            ########     Error f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 2 * Num_fields + fi + 1)
            err = out_pred[:, fi] - out_true[:, fi]
            plt.tricontourf(triObj, err, Num_levels, cmap='coolwarm')
            cb = plt.colorbar()
            plt.clim(vmin=-max(abs(fmin[fi]), abs(fmax[fi])), vmax=max(abs(fmin[fi]), abs(fmax[fi])))
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            plt.xlabel('$' + self.input_name[0] + '$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('field error$' + field_name[fi] + '$' + '', fontsize=30)

    def plot_fields_ms(self, out_true, out_pred, coord, cmin_max=None, fmin_max=None, field_name=None):

        plt.clf()
        Num_fields = out_true.shape[-1]
        if fmin_max == None:
            fmin, fmax = out_true.min(axis=(0, 1)), out_true.max(axis=(0, 1))
        else:
            fmin, fmax = fmin_max[0], fmin_max[1]

        if cmin_max == None:
            cmin, cmax = coord.min(axis=(0, 1)), coord.max(axis=(0, 1))
        else:
            cmin, cmax = cmin_max[0], cmin_max[1]

        if field_name == None:
            field_name = self.field_name

        x_pos = coord[:, :, 0]
        y_pos = coord[:, :, 1]
        ############################# Plotting ###############################
        for fi in range(Num_fields):
            plt.rcParams['font.size'] = 20

            ########      Exact f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 0 * Num_fields + fi + 1)
            f_true = out_true[:, :, fi]
            plt.pcolormesh(x_pos, y_pos, f_true, cmap='jet', shading='gouraud', antialiased=True, snap=True)
            # plt.contourf(x_pos, y_pos, f_true, cmap='jet',)
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.family'] = 'Times New Roman'
            # cb.set_label('value', rotation=0, fontdict=self.font, y=1.08)
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('True field $' + field_name[fi] + '$' + '', fontsize=30)


            ########     Learned f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 1 * Num_fields + fi + 1)
            f_pred = out_pred[:, :, fi]
            plt.pcolormesh(x_pos, y_pos, f_pred, cmap='jet', shading='gouraud', antialiased=True, snap=True)
            # plt.contourf(x_pos, y_pos, f_pred, cmap='jet',)
            cb = plt.colorbar()
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            plt.clim(vmin=fmin[fi], vmax=fmax[fi])
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            # plt.xlabel('$x$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('Pred field $' + field_name[fi] + '$' + '', fontsize=30)

            ########     Error f(t,x,y)     ###########
            plt.subplot(3, Num_fields, 2 * Num_fields + fi + 1)
            err = f_true - f_pred
            plt.pcolormesh(x_pos, y_pos, err, cmap='coolwarm', shading='gouraud', antialiased=True, snap=True)
            # plt.contourf(x_pos, y_pos, err, cmap='coolwarm', )
            if cmin_max is not None:
                plt.axis((cmin[0], cmax[0], cmin[1], cmax[1]))
            cb = plt.colorbar()
            plt.clim(vmin=-max(abs(fmin[fi]), abs(fmax[fi])), vmax=max(abs(fmin[fi]), abs(fmax[fi])))
            cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小
            plt.rcParams['font.size'] = 20
            plt.xlabel('$' + self.input_name[0] + '$', fontdict=self.font)
            plt.ylabel('$' + self.input_name[1] + '$', fontdict=self.font)
            plt.title('field error$' + field_name[fi] + '$' + '', fontsize=30)

    def plot_fields_am(self, out_true, out_pred, coord, p_id, fig):

        fmax = out_true.max(axis=(0, 1, 2))  # 云图标尺
        fmin = out_true.min(axis=(0, 1, 2))  # 云图标尺

        def anim_update(t_id):
            print('para:   ' + str(p_id) + ',   time:   ' + str(t_id))
            axes = self.plot_fields_ms(out_true[t_id], out_pred[t_id], coord[t_id], fmin_max=(fmin, fmax))
            return axes

        anim = FuncAnimation(fig, anim_update,
                             frames=np.arange(0, out_true.shape[0]).astype(np.int64), interval=200)

        anim.save(self.log_dir + "\\" + str(p_id) + ".gif", writer='pillow', dpi=300)

    def plot_dynamics(self, t_star, target_true, target_pred):
        ## plot F_D F_L
        min_value = target_true - (target_true.max() - target_true.min()) * 0.05
        max_value = target_true + (target_true.max() - target_true.min()) * 0.05

        plt.plot(t_star, target_true, 'b-', linewidth=5.0, label="true_value")
        plt.fill_between(t_star, min_value, max_value, alpha=0.3, color='b', label="confidence")
        plt.scatter(t_star, target_pred, c='r', marker=10.0, label="pred_value")
        plt.xlabel('$t$', fontdict=self.font)
        plt.ylabel('$value$', fontdict=self.font)
        plt.grid(True)
        plt.legend(prop=self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])

    def plot_pareto(self, objs):

        poly_coff = np.polyfit(objs[:, 0], objs[:, 1], deg=2)  # 一次多项式拟合，相当于线性拟合
        pareto_x = np.linspace(objs[:, 0].min(), objs[:, 0].max(), 20)
        pareto_y = np.polyval(poly_coff, pareto_x)

        plt.scatter(objs[:, 0], objs[:, 1], label="real optimizations")
        plt.plot(pareto_x, pareto_y, c='r', label="pareto fronts fit")
        plt.legend(prop=self.font)
        plt.grid(True)
        plt.xlabel('power / W', self.font)
        plt.ylabel('efficiency / %', self.font)
        plt.yticks(fontproperties='Times New Roman', size=self.font["size"])
        plt.xticks(fontproperties='Times New Roman', size=self.font["size"])
    #   delta = objs.max(axis=0) - objs.min(axis=0)
    #   plt.xlim(objs[:, 0].min() - delta[:, 0] * 0.1, objs[:, 0].max() + delta[:, 0] * 0.1)
    #   plt.ylim(objs[:, 1].min() - delta[:, 1] * 0.1, objs[:, 1].max() + delta[:, 1] * 0.1)
    #