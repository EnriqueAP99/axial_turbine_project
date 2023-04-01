"""
En este módulo se caracteriza el modelo de pérdidas que se va a emplear en el módulo "axial_turb_solver.py", además se
calcula el número de Reynolds.
"""
from config_class import logger
from config_class import configuration_parameters
from math import cos, fabs, atan, tan, radians, degrees, pi
import gas_modeling
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
# https://medium.com/@hdezfloresmiguelangel/introducci%C3%B3n-a-la-interpolaci%C3%B3n-unidimensional-con-python-1127abe510a1
# http://hyperphysics.phy-astr.gsu.edu/hbasees/Kinetic/visgas.html
# https://www.youtube.com/watch?v=tTG5Re5G7B8&ab_channel=Petr%C3%B3leoyprogramaci%C3%B3nSMAE


def Reynolds(rho_2: float, C_2: float, T_2: float, s: float, H: float, alpha_2: float,
             productos: gas_modeling.mixpm):
    """ Se calcula el número de Reynolds usando el diámetro hidráulico y las propiedades del fluido a la salida del
    estátor.
            :param rho_2: Densidad a la salida del estátor (kg/m^3).
            :param C_2: Velocidad a la salida del estátor (m/s).
            :param T_2: Temperatura a la salida del estátor (K).
            :param s: Parámetro de paso geométrico del estátor (m).
            :param H: Altura de los álabes del estátor (m).
            :param alpha_2: Ángulo del fluido con la dirección axial a la salida del estátor (rads).
            :param productos: objeto que modela los productos como clase mixpm del módulo gas_modeling.
                    :return: Se devuelve el número de Reynolds."""
    D_h = 2 * s * H * cos(alpha_2) / (s * cos(alpha_2) + H)
    mu = productos.Wilke_visc_calculator(T_2)
    Re = rho_2 * C_2 * D_h / mu
    return Re


# Usar criterio de Zweifel para establecer un s/b para ejemplo que se aplique
# https://core.ac.uk/download/pdf/147259438.pdf
def Soderberg_correlation(blade: str, alfap_1: float, alfap_2: float, H: float, b: float):
    """ Función que aplica la correlación de Soderberg, algunas correcciones propias de la correlación se introducen
    en el módulo "axial_turb_solver.py".
            :param blade: Cadena de caracteres para diferenciar estátor ('est') y rótor ('rot').
            :param alfap_1: Ángulo que forma el borde de ataque del álabe en cuestión con la dirección axial (rads).
            :param alfap_2: Ángulo que forma el borde de salida del álabe en cuestión con la dirección axial (rads).
            :param H: Altura de los álabes de la corona que corresponda (m).
            :param b: Cuerda de los álabes de la corona que corresponda (m).
                    :return: Se devuelve únicamente el valor del coeficiente adimensional de pérdidas, ya que se
                            establece que la incidencia y la desviación del flujo son nulos."""
    xi = 0.0
    theta = alfap_1 + alfap_2
    epsilon = theta     # t_max_adim = 0.2
    if blade == 'est':
        xi = (1.04 + 0.06*(0.01*epsilon)**2)*(0.993+0.021*b/H)-1
    elif blade == 'rot':
        xi = (1.04 + 0.06*(0.01*epsilon)**2)*(0.975+0.075*b/H)-1
    return xi


class AM_loss_model:
    """ Clase que contiene instancias necesarias para aplicar la correlación de Ainley and Mathieson, haciendo uso de
    interpolación por splines generados por la librería Scipy. Notar que una vez creado el objeto no se estará
    interpolando con cada llamada al módulo actual, sino que se llamará a una instancia del objeto que se inicializó
    durante la configuración del solver."""
    def __init__(self, cfg: configuration_parameters):
        """ :param cfg: Argumento necesario para tener acceso a los parámetros geométricos que contiene dicho objeto."""
        self.tau2_ypmin = float()
        self.tau2_ypmin_seed = []
        self.Yp_min = float()
        self.crown_num = int()
        self.Yp_preiter = list()
        self.Y_t_preiter = list()
        self.Y_t_rotor_iter_mode = float()
        self.cfg = cfg

        def x_sp(x_list):
            return np.linspace(x_list[0], x_list[-1])

        def f_sp(x_list, y_list, order):
            return InterpolatedUnivariateSpline(x_list, y_list, k=order)

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
        for i, v in enumerate([70, 65, 60, 55, 50, 40, 30]):
            self.is_b1a2_sc_075[i] += [v]
        s_c, d_i_s_a2m40 = [0.4, 0.5, 0.6, 0.7, 0.77, 0.9, 1], [8.0, 6.8, 4.5, 1.7, -0.6, -8.3, -15]
        d_i_s_a2m50, d_i_s_a2m60 = [8.0,  6.8, 4.5, 1.7, -0.6, -6.6, -11.4], [8.0,  6.8, 4.5, 1.7, -0.6, -4.9, -6.6]
        self.d_i_s_s_c = [[x_sp(s_c), f_sp(s_c, i, 3)] for i in [d_i_s_a2m60, d_i_s_a2m50, d_i_s_a2m40]]
        for i, v in enumerate([60, 50, 40]):
            self.d_i_s_s_c[i] += [v]
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
        self.yp_s_c_b1kn = [[x_sp(s_c), f_sp(s_c, i, 3)] for i in [yp_a2m70, yp_a2m65, yp_a2m60,
                                                                   yp_a2m50, yp_a2m40]]
        self.yp_s_c_b1kn.insert(0, [x_sp(s_c_m75), f_sp(s_c_m75, yp_a2m75, 3)])
        self.yp_s_c_b1kn.insert(0, [x_sp(s_c_m80), f_sp(s_c_m80, yp_a2m80, 3)])
        for i, v in enumerate([80, 75, 70, 65, 60, 50, 40]):
            self.yp_s_c_b1kn[i] += [v]
        s_c = [0.3, 0.5, 0.7, 0.8, 1.0]
        s_c_m55 = [0.4, 0.5, 0.7, 0.8, 1.0]
        yp_a2m70 = [0.162, 0.134, 0.149, 0.162, 0.191]
        yp_a2m65 = [0.154, 0.119, 0.121, 0.129, 0.153]
        yp_a2m60 = [0.148, 0.106, 0.104, 0.109, 0.126]
        yp_a2m55 = [0.118, 0.098, 0.084, 0.087, 0.106]
        yp_a2m50 = [0.144, 0.093, 0.074, 0.075, 0.091]
        yp_a2m40 = [0.141, 0.088, 0.068, 0.067, 0.076]
        self.yp_s_c_b1kb2k = [[x_sp(s_c), f_sp(s_c, i, 2)] for i in [yp_a2m70, yp_a2m65, yp_a2m60,
                                                                     yp_a2m50, yp_a2m40]]
        self.yp_s_c_b1kb2k.insert(3, [x_sp(s_c_m55), f_sp(s_c_m55, yp_a2m55, 2)])
        for i, v in enumerate([70, 65, 60, 55, 50, 40]):
            self.yp_s_c_b1kb2k[i] += [v]
        i_f, yp_f = [-4.1, -3.0, -2.0, -1.0, 0.05, 1.0, 1.7], [6.4, 4.3, 2.75, 1.6, 1.0, 2.1, 6.1]
        self.yp_f_i_f = [x_sp(i_f), f_sp(i_f, yp_f, 2)]
        y_f, d_a2 = [1.0, 2.2], [0.0, 2.4]
        self.d_a2_yp_f = [x_sp(y_f), f_sp(y_f, d_a2, 1)]
        x, lambdav = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.0055, 0.0063, 0.0087, 0.0130, 0.0192, 0.0276]
        self.sec_losses = [x_sp(x), f_sp(x, lambdav, 3)]

    @staticmethod
    def interp_series_a2(ang, x, series):
        """ Método que permite evaluar valores no discretos de ángulos de salida del fluido en la aplicación de la
        correlación de "Ainley and Mathieson". Se hace uso de interpolación con splines que facilita la librería scipy.
                :param ang: Ángulo a evaluar (degrees).
                :param x: Coordenada de abscisas para la que se evalúan las funciones dadas para diferentes ángulos.
                :param series: Lista que contiene listas de 3 elementos: la serie de abscisas, una función spline
                              y el ángulo que le corresponde.
                        :return: Devuelve el resultado de evaluar, para el valor no discreto del ángulo requerido, el
                                valor de la función generada por splines a partir de los valores de las funciones
                                originales evaluadas en 'x' frente a los ángulos que corresponden a cada función
                                original. """
        serie_a2 = [a2 for _, _, a2 in series]
        serie_y = [float(funcion(x)) for _, funcion, _ in series]
        f_a2 = InterpolatedUnivariateSpline(list(reversed(serie_a2)), list(reversed(serie_y)), k=2)
        return f_a2(ang)

    def Yp_min_calculator(self, tau_1: float, tau_2_seed=None):
        """ Se determina el valor de Yp mínimo y el valor de ángulo de salida del flujo, que corresponde, respecto al
        álabe y respecto la dirección axial. Se almacenan como instacias el valor mínimo de Yp y el ángulo que
        corresponde de salida del flujo para que durante llamadas recursivas no sea necesario reevaluar estos valores
        constantemente.
                :param tau_2_seed: Valor que se obtuvo en este método en iteraciones previas.
                :param tau_1: alfap_1 más la incidencia del flujo (degrees).
                        :return: No se devuelve nada. """
        num, tol = self.crown_num, self.cfg.TOL
        alfap_1 = self.cfg.geom['alfap_i_est'][num//2] if num % 2 == 0 else self.cfg.geom['alfap_i_rot'][num//2]
        alfap_2 = self.cfg.geom['alfap_o_est'][num//2] if num % 2 == 0 else self.cfg.geom['alfap_o_rot'][num//2]
        alfap_1, alfap_2, d_tol = degrees(alfap_1), degrees(alfap_2), 180*tol/pi
        if tau_2_seed is None:
            tau_2n, tau_2nc = [alfap_2, (alfap_2 * (1 + d_tol))] if fabs(alfap_2) > d_tol else [d_tol, d_tol*(1+d_tol)]
        else:
            tau_2n, tau_2nc = (tau_2_seed * (1 - d_tol)), tau_2_seed
        f_1, cond, Yp_n = None, False, 0.0
        if tau_1 < alfap_1*(1 - d_tol):
            logger.warning('Incidencia negativa:    tau_1=%.2f°   ángulo_B.A.=%.2f°', tau_1, alfap_1)
        elif tau_1 - alfap_1 > 15:
            logger.warning('Incidencia alta:    tau_1=%.2f°   ángulo_B.A.=%.2f°', tau_1, alfap_1)
        else:
            logger.info('Incidencia apta:    tau_1=%.2f°   ángulo_B.A.=%.2f°', tau_1, alfap_1)
        if abs(tau_1 - alfap_1) > d_tol:
            while not cond:
                Yp_n = self.AM_loss_model_operations(tau_1, tau_2n)
                Yp_nc = self.AM_loss_model_operations(tau_1, tau_2nc)
                f_2 = f_1
                f_1 = (Yp_nc - Yp_n)/(tau_2nc - tau_2n)
                if f_2 is None:
                    f_2 = f_1
                if f_1 * f_2 <= 0.0:
                    cond = True
                else:
                    if f_1 < 0.0:
                        tau_2n += tau_2n * d_tol
                        tau_2nc += tau_2nc * d_tol
                    else:
                        tau_2n -= tau_2n * d_tol
                        tau_2nc -= tau_2nc * d_tol
                        if tau_2n < alfap_2*(1-d_tol):
                            tau_2n = alfap_2
                            break
        else:
            Yp_n, tau_2n = self.AM_loss_model_operations(tau_1, alfap_2, True), alfap_2
        logger.info('Se ha encontrado el mínimo ->    Pérdida primaria de presión adimensional mínima: %.4f    '
                    'Valor ideal de tau_2: %.3f°', Yp_n, tau_2n)
        self.Yp_min, self.tau2_ypmin = Yp_n, tau_2n
        return

    def AM_loss_model_operations(self, tau_1: float, tau_2: float, Yp_i0_req=False):
        """ Se determina el valor de Yp según la correlación de AM. En caso de incidencia negativa se aplica el valor
        absoluto, pese a que se subestimen las pérdidas reales se evitan pérdidas negativas que resultarían si no se
        corrige.
                :param Yp_i0_req: Se requiere Yp_i0.
                :param tau_1: alfap_1 más la incidencia del flujo (degrees).
                :param tau_2: alfap_2 más la deflexión del flujo (degrees).
                        :return: Se devuelve el valor de Yp calculado. """
        geom, num = self.cfg.geom, self.crown_num
        alfap_1 = degrees(geom['alfap_i_est'][num//2] if num % 2 == 0 else geom['alfap_i_rot'][num//2])
        s, b, t_max = geom['s'][num], geom['b'][num], geom['t_max'][num]
        a2_a2sc075 = self.alfa2rel_s_c[1](s / b)
        alpha_2_sc075 = tau_2 / a2_a2sc075
        is_sc075 = self.interp_series_a2(alpha_2_sc075, alfap_1 / alpha_2_sc075, self.is_b1a2_sc_075)
        delta_is = self.interp_series_a2(tau_2, s / b, self.d_i_s_s_c)
        i_s = delta_is + is_sc075
        i_is = fabs(tau_1 - alfap_1) / i_s
        yp_f = self.yp_f_i_f[1](i_is)
        Yp_i0_b1kn = self.interp_series_a2(tau_2, s / b, self.yp_s_c_b1kn)
        Yp_i0_b1kb2k = self.interp_series_a2(tau_2, s / b, self.yp_s_c_b1kb2k)
        t_max_b = t_max / b
        if t_max_b < 0.15:
            t_max_b = 0.15
        elif t_max_b > 0.25:
            t_max_b = 0.25
        Yp_i0 = (Yp_i0_b1kn + (((alfap_1 / tau_2) ** 2) * (Yp_i0_b1kb2k - Yp_i0_b1kn))) * ((t_max_b / 0.2) **
                                                                                           (alfap_1 / tau_2))
        Yp = Yp_i0*yp_f
        if not Yp_i0_req:
            return Yp
        else:
            return Yp_i0

    def Ainley_and_Mathieson_Loss_Model(self, num: int, tau_1: float, step_iter_mode=False,
                                        Y_total=0.0):
        """ Se coordinan el resto de instancias para que al llamar a este método externamente se aplique la
        correlación.
                :param step_iter_mode: True cuando se itera el método gen_step para la corrección por Reynolds.
                :param num: Numeración de cada corona de álabes, empezando por 0.
                :param tau_1: alfap_1 más la incidencia del flujo (degrees).
                :param Y_total: Parámetro de pérdidas del modelo de pérdidas AM.
                        :return: Se devuelve Y_total y tau_2, excepto cuando step_iter_mode, en dicho caso Y_total es
                                conocido y solo se devuelve tau_2, en ambos casos en radianes. """
        self.crown_num, geom = num, self.cfg.geom
        alfap_2 = degrees(geom['alfap_o_est'][num//2] if num % 2 == 0 else geom['alfap_o_rot'][num//2])
        A_1, s, K_i, t_max, H, r_r, r_c, b, t_e = [geom[i][num] for i in ['areas', 's', 'K', 't_max', 'H', 'r_r',
                                                                          'r_c', 'b', 't_e']]
        A_2 = geom['areas'][num+1]
        if not step_iter_mode:
            if len(self.tau2_ypmin_seed) == self.cfg.n_step*2:
                tau_2_seed = self.tau2_ypmin_seed[num]
                self.Yp_min_calculator(tau_1, tau_2_seed)
            else:
                self.Yp_min_calculator(tau_1)
                self.tau2_ypmin_seed.append(self.tau2_ypmin)
            tau_2 = alfap_2
            tau_diff = tau_2_iter = alfap_2
            tol = self.cfg.TOL
            while fabs(tau_diff / alfap_2) > tol:
                Yp = self.AM_loss_model_operations(tau_1, tau_2_iter)
                d_tau_2 = self.d_a2_yp_f[1](Yp / self.Yp_min)
                if d_tau_2 < 0:
                    d_tau_2 = 0.0
                tau_2 = d_tau_2 + self.tau2_ypmin
                tau_diff = tau_2 - tau_2_iter
                tau_2_iter = tau_2
            Yp = self.AM_loss_model_operations(tau_1, tau_2)
            d_tau_2 = self.d_a2_yp_f[1](Yp / self.Yp_min)
            tau_2 = d_tau_2 + self.tau2_ypmin
            if len(self.Yp_preiter) < self.crown_num + 1:
                self.Yp_preiter.append(Yp)
            else:
                self.Yp_preiter[-1] = Yp
            tau_2, tau_1 = radians(tau_2), radians(tau_1)
            x = ((A_2*cos(tau_2)/(A_1*cos(tau_1)))**2)/(1+(r_r/r_c))
            lambda_ = self.sec_losses[1](x)
            tau_m = atan((tan(tau_1)+tan(tau_2))/2)
            C_L = 2*(s/b)*(tan(tau_1) - tan(tau_2))*cos(tau_m)
            Z = ((C_L*b/s)**2)*((cos(tau_2)**2)/(cos(tau_m)**3))
            Ys = lambda_*Z
            Yk = 0.0
            if not K_i == 0.0:
                Yk = 0.5 * (K_i / H) * Z
            Y_total = (Yp + Ys + Yk) * self.ji_Te_te_s[1](t_e / s)
            if len(self.Y_t_preiter) < num + 1:
                self.Y_t_preiter.append(Y_total)
            else:
                self.Y_t_preiter[-1] = Y_total
            logger.info('Resulta ->    Pérdida total de presión adimensional: %.4f    Ángulo del B.S.: %.3f°    '
                        'Valor de tau_2: %.3f°', fabs(Y_total), alfap_2, degrees(tau_2))
            return fabs(Y_total), tau_2
        else:
            if not bool(num % 2):
                Yp = (Y_total / self.Y_t_preiter[num]) * self.Yp_preiter[num]
                d_tau_2 = self.d_a2_yp_f[1](Yp / self.Yp_min)
                tau_2 = d_tau_2 + self.tau2_ypmin
                tau_2 = radians(tau_2)
                self.Y_t_rotor_iter_mode = Y_total
                return tau_2
            else:
                # Y_total = self.Y_t_preiter[num]*self.Y_t_rotor_iter_mode/self.Y_t_preiter[num-1]
                # Yp = (Y_total / self.Y_t_preiter[num]) * self.Yp_preiter[num]
                Yp = (self.Y_t_rotor_iter_mode/self.Y_t_preiter[num-1]) * self.Yp_preiter[num]
                d_tau_2 = self.d_a2_yp_f[1](Yp / self.Yp_min)
                tau_2 = d_tau_2 + self.tau2_ypmin
                tau_2 = radians(tau_2)
                self.Y_t_rotor_iter_mode = Y_total
                return Y_total, tau_2
