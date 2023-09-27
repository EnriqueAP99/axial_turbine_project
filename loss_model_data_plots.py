
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt


def x_sp(x_list: list):
    return np.linspace(x_list[0], x_list[-1])


def f_sp(x_list: list, y_list: list, order: int):
    return InterpolatedUnivariateSpline(x_list, y_list, k=order)


def lineal_interpolation(x_target=None, x=None, series=None, y=None):
    if y is None:
        serie_x = [parameter for _, _, parameter in series]
        serie_y = [float(funcion(x)) for _, funcion, _ in series]
    else:
        serie_x, serie_y = x, y
    if serie_x[-1] < serie_x[0]:
        serie_x, serie_y = serie_x[::-1], serie_y[::-1]
    return InterpolatedUnivariateSpline(serie_x, serie_y, k=2)(x_target)


class Loss_Model_Data:
    """ Clase que contiene instancias necesarias para aplicar la correlación de Ainley and Mathieson, haciendo uso de
    interpolación por splines generados por la librería Scipy. Notar que una vez creado el objeto no se estará
    interpolando con cada llamada al módulo actual, sino que se llamará a una instancia del objeto que se inicializó
    durante la configuración del solver."""

    def __init__(self):

        self.crown_num = int()
        self.outlet_angle_before_mod = list()   # radianes
        self.Yp_preiter = list()
        self.Yp_iter_mode = None
        self.Y_t_preiter = list()
        self.Y_t_stator_iter_mode = float()
        self.limit_mssg = [False, False, False]

        x, y = [0, 0.02, 0.04, 0.061, 0.082, 0.10, 0.12], [0.92, 1.00, 1.10, 1.23, 1.38, 1.52, 1.69]
        self.ji_Te_te_s = [x_sp(x), f_sp(x, y, 3)]
        x = [0.43, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.995]
        y = [1.1, 1.08, 1.065, 1.05, 1.035, 1.018, 1.0, 0.98, 0.96, 0.94, 0.92, 0.9]
        self.alfa2rel_s_c = [x_sp(x), f_sp(x, y, 3)]
        x_a2m70, y_a2m70 = [-0.43, 0.09, 0.35], [35, 45, 42]
        x_a2m65, y_a2m65 = [-0.8, 0.16, 0.96], [13, 37, 20]
        x_a2m60, y_a2m60 = [-0.9, 0.23, 0.97], [7.5, 30, 20]
        x_a2m55, y_a2m55 = [-0.88, 0.33, 0.97], [7.5, 25, 20]
        x_a2m50, y_a2m50 = [-0.88, 0.6, 0.99], [7.5, 21, 20]
        x_a2m40, y_a2m40 = [-0.98, 0.2, 0.7], [5, 15, 17.5]
        x_a2m30, y_a2m30 = [-0.93, 0.03, 0.35], [4.9, 10, 11]
        is_list = [[x_a2m70, y_a2m70], [x_a2m65, y_a2m65], [x_a2m60, y_a2m60], [x_a2m55, y_a2m55],
                   [x_a2m50, y_a2m50], [x_a2m40, y_a2m40], [x_a2m30, y_a2m30]]
        self.is_b1a2_sc_075 = [[x_sp(X), f_sp(X, Y, 2)] for X, Y in is_list]
        for ii, v in enumerate([70, 65, 60, 55, 50, 40, 30]):
            self.is_b1a2_sc_075[ii] += [v]
        s_c, d_i_s_a2m60 = [0.4, 0.5, 0.6, 0.7, 0.77, 0.9, 1], [8.0, 6.8, 4.5, 1.7, -0.6, -8.3, -15]
        d_i_s_a2m50, d_i_s_a2m40 = [8.0,  6.8, 4.5, 1.7, -0.6, -6.6, -11.4], [8.0,  6.8, 4.5, 1.7, -0.6, -4.9, -6.6]
        self.d_i_s_s_c = [[x_sp(s_c), f_sp(s_c, ii, 3)] for ii in [d_i_s_a2m60, d_i_s_a2m50, d_i_s_a2m40]]
        for ii, v in enumerate([60, 50, 40]):
            self.d_i_s_s_c[ii] += [v]
        s_c = [0.3, 0.5, 0.7, 0.9, 1.1]
        s_c_m80 = [0.4, 0.5, 0.7, 0.84]
        s_c_m75 = [0.45, 0.5, 0.7, 0.95]
        yp_a2m80 = [0.064, 0.058, 0.06, 0.068]
        yp_a2m75 = [0.052, 0.05, 0.049, 0.058]
        yp_a2m70 = [0.069, 0.045, 0.037, 0.043, 0.057]
        yp_a2m65 = [0.068, 0.043, 0.032, 0.032, 0.045]
        yp_a2m60 = [0.066, 0.041, 0.028, 0.025, 0.035]
        yp_a2m50 = [0.0645, 0.039, 0.026, 0.022, 0.024]
        yp_a2m40 = [0.063, 0.037, 0.024, 0.019, 0.018]
        self.yp_s_c_b1kn = [[x_sp(s_c), f_sp(s_c, ii, 3)] for ii in [yp_a2m70, yp_a2m65, yp_a2m60,
                                                                     yp_a2m50, yp_a2m40]]
        self.yp_s_c_b1kn.insert(0, [x_sp(s_c_m75), f_sp(s_c_m75, yp_a2m75, 3)])
        self.yp_s_c_b1kn.insert(0, [x_sp(s_c_m80), f_sp(s_c_m80, yp_a2m80, 3)])
        for ii, v in enumerate([80, 75, 70, 65, 60, 50, 40]):
            self.yp_s_c_b1kn[ii] += [v]
        s_c = [0.3, 0.5, 0.7, 0.8, 1.0]
        s_c_m55 = [0.4, 0.5, 0.7, 0.8, 1.0]
        yp_a2m70 = [0.162, 0.134, 0.149, 0.162, 0.191]
        yp_a2m65 = [0.154, 0.119, 0.121, 0.129, 0.153]
        yp_a2m60 = [0.148, 0.106, 0.104, 0.109, 0.126]
        yp_a2m55 = [0.118, 0.098, 0.084, 0.087, 0.106]
        yp_a2m50 = [0.144, 0.093, 0.074, 0.075, 0.091]
        yp_a2m40 = [0.141, 0.088, 0.068, 0.067, 0.076]
        self.yp_s_c_b1kb2k = [[x_sp(s_c), f_sp(s_c, ii, 2)] for ii in [yp_a2m70, yp_a2m65, yp_a2m60,
                                                                       yp_a2m50, yp_a2m40]]
        self.yp_s_c_b1kb2k.insert(3, [x_sp(s_c_m55), f_sp(s_c_m55, yp_a2m55, 2)])
        for ii, v in enumerate([70, 65, 60, 55, 50, 40]):
            self.yp_s_c_b1kb2k[ii] += [v]
        i_f, yp_f = [-4.1, -3.0, -2.0, -1.0, 0.05, 1.0, 1.7], [6.4, 4.3, 2.75, 1.6, 1.0, 2.1, 6.1]
        self.yp_f_i_f = [x_sp(i_f), f_sp(i_f, yp_f, 2)]
        x, lambdav = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.0055, 0.0063, 0.0087, 0.0130, 0.0192, 0.0276]
        self.sec_losses = [x_sp(x), f_sp(x, lambdav, 3)]

        cos_m1, tau_2_ast = [35.5, 79], [30, 80]
        self.tau_2_ast = [x_sp(cos_m1), f_sp(cos_m1, tau_2_ast, 1)]
        s_e = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        k_m_1 = [1.0, 1.0, 1.2, 1.45, 1.8, 2.3, 3.0, 3.8, 5.0]
        k_m_09 = [1.0, 1.0, 1.1, 1.23, 1.38, 1.6, 1.85, 2.15, 2.5]
        k_m_08 = [1.0, 1.0, 1.03, 1.08, 1.14, 1.21, 1.29, 1.39, 1.5]
        s_e_07 = [0.0, 0.2, 0.8]
        k_m_07 = [1.0, 1.0, 1.17]

        serie_k_m_1 = [x_sp(s_e), f_sp(s_e, k_m_1, 3), 1.0]
        serie_k_m_09 = [x_sp(s_e), f_sp(s_e, k_m_09, 3), 0.9]
        serie_k_m_08 = [x_sp(s_e), f_sp(s_e, k_m_08, 3), 0.8]
        serie_k_m_07 = [x_sp(s_e_07), f_sp(s_e_07, k_m_07, 2), 0.7]

        self.km_series = [serie_k_m_07, serie_k_m_08, serie_k_m_09, serie_k_m_1]


if __name__ == '__main__':
    LMDobj = Loss_Model_Data()
    for item in [LMDobj.km_series, LMDobj.is_b1a2_sc_075, LMDobj.d_i_s_s_c, LMDobj.yp_s_c_b1kn, LMDobj.yp_s_c_b1kb2k]:
        for [i, j, k] in item:
            plt.plot(i, j(i))
        plt.grid()
        plt.show()

    for item in [LMDobj.ji_Te_te_s, LMDobj.alfa2rel_s_c, LMDobj.yp_f_i_f, LMDobj.sec_losses, LMDobj.tau_2_ast]:
        plt.plot(item[0], item[1](item[0]))
        plt.grid()
        plt.show()

# $\alpha_1^\prime$/$\alpha_2$ (when $s/c=0.75$)
# Stalling incidence $i_s$ (when $s/c=0.75$)
# $\alpha_2$ = 30$\degree$
