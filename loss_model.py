"""
En este módulo se caracteriza el modelo de pérdidas que se va a emplear en el módulo "axial_turb_solver.py", además se
calcula el número de Reynolds.
"""
from config_class import *
from math import cos, atan, tan, radians, degrees, pi, acos, fabs, sqrt, log10, sin, asin
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
# https://medium.com/@hdezfloresmiguelangel/introducci%C3%B3n-a-la-interpolaci%C3%B3n-unidimensional-con-python-1127abe510a1
# http://hyperphysics.phy-astr.gsu.edu/hbasees/Kinetic/visgas.html
# https://www.youtube.com/watch?v=tTG5Re5G7B8&ab_channel=Petr%C3%B3leoyprogramaci%C3%B3nSMAE


def Reynolds(num: int, rho_2: float, C_2: float, T_2: float, config: config_class, productos: gas_model_to_solver):
    """ Se calcula el número de Reynolds usando el diámetro hidráulico y las propiedades del fluido a la salida del
    estátor.
            :param num: Número identificador de la corona que se evalúa.
            :param rho_2: Densidad a la salida del estátor (kg/m^3).
            :param C_2: Velocidad a la salida del estátor (m/s).
            :param T_2: Temperatura a la salida del estátor (K).
            :param config: Objeto con información necesaria para la ejecución.
            :param productos: objeto que modela los productos como clase mixpm del módulo gas_modeling.
                    :return: Se devuelve el número de Reynolds."""

    if config.loss_model == 'soderberg_correlation':
        alfa_key = 'alfap_i_est' if num % 2 == 0 else 'alfap_o_rot'
        s, H, alfa = config.geom['s'][num], config.geom['H'][num], config.geom[alfa_key][num//2]
        D_h = 2 * s * H * cos(alfa) / (s * cos(alfa) + H)
        c_len = D_h
    else:
        c_len = config.geom['b'][num]

    mu = productos.get_din_visc(T_2)
    Re = int((rho_2 * C_2 * c_len / mu).__round__(0))
    return Re


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
    return InterpolatedUnivariateSpline(serie_x, serie_y, k=2)(x_target)


class Ainley_and_Mathieson_Loss_Model:  # Ver paper: https://apps.dtic.mil/sti/pdfs/ADA950664.pdf
    """ Clase que contiene instancias necesarias para aplicar la correlación de Ainley and Mathieson, haciendo uso de
    interpolación por splines generados por la librería Scipy. Notar que una vez creado el objeto no se estará
    interpolando con cada llamada al módulo actual, sino que se llamará a una instancia del objeto que se inicializó
    durante la configuración del solver."""

    def __init__(self, cfg: config_class):
        """ :param cfg: Argumento necesario para tener acceso a los parámetros geométricos que contiene dicho objeto."""

        self.cfg = cfg
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
        for i, v in enumerate([70, 65, 60, 55, 50, 40, 30]):
            self.is_b1a2_sc_075[i] += [v]
        self.is_b1a2_sc_075 = self.is_b1a2_sc_075[::-1]
        s_c, d_i_s_a2m40 = [0.4, 0.5, 0.6, 0.7, 0.77, 0.9, 1], [8.0, 6.8, 4.5, 1.7, -0.6, -8.3, -15]
        d_i_s_a2m50, d_i_s_a2m60 = [8.0,  6.8, 4.5, 1.7, -0.6, -6.6, -11.4], [8.0,  6.8, 4.5, 1.7, -0.6, -4.9, -6.6]
        self.d_i_s_s_c = [[x_sp(s_c), f_sp(s_c, i, 3)] for i in [d_i_s_a2m60, d_i_s_a2m50, d_i_s_a2m40]]
        for i, v in enumerate([60, 50, 40]):
            self.d_i_s_s_c[i] += [v]
        self.d_i_s_s_c = self.d_i_s_s_c[::-1]
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
        self.yp_s_c_b1kn = self.yp_s_c_b1kn[::-1]
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
        self.yp_s_c_b1kb2k = self.yp_s_c_b1kb2k[::-1]
        i_f, yp_f = [-4.1, -3.0, -2.0, -1.0, 0.05, 1.0, 1.7], [6.4, 4.3, 2.75, 1.6, 1.0, 2.1, 6.1]
        self.yp_f_i_f = [x_sp(i_f), f_sp(i_f, yp_f, 2)]
        x, lambdav = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.0055, 0.0063, 0.0087, 0.0130, 0.0192, 0.0276]
        self.sec_losses = [x_sp(x), f_sp(x, lambdav, 3)]

        cos_m1, tau_2_ast = [35.5, 79], [30, 80]
        self.tau_2_ast = [x_sp(cos_m1), f_sp(cos_m1, tau_2_ast, 1)]

        for num, cuerda in enumerate(self.cfg.geom['b']):
            solidez = cuerda / self.cfg.geom['s'][num]
            if 1/solidez < 0.3 or 1/solidez > 1:
                record.warning('La solidez del %s del escalonamiento %s excede el rango de valores para los que '
                               'existe función. El valor es %.2f y los límites son [1.0, 2.5].',
                               'estátor' if bool((num+1) % 2) else 'rótor', (num//2) + 1, solidez)

        throat_distance = self.cfg.geom['o']
        pitch = self.cfg.geom['s']
        mean_radius_curvature = self.cfg.geom['e']
        for i in range(self.cfg.n_steps*2):
            if -1 < throat_distance[i] / pitch[i] < 1:
                x_value = degrees(acos(throat_distance[i] / pitch[i]))
            elif throat_distance[i] / pitch[i] > 1:
                x_value = 0
            else:
                x_value = 90
            if x_value < 35 or x_value > 80:
                record.warning('El valor de acos(o/s) en el escalonamiento número %s excede los límites de la '
                               'correlación. Valor: %.1f; Límites: %s', (i+2)//2, x_value, [35, 80])
            tau_2_ast_value = self.tau_2_ast[1](x_value)
            flow_angle = tau_2_ast_value + 4 * (pitch[i] / mean_radius_curvature[i])
            self.outlet_angle_before_mod.append(radians(flow_angle))

    def AM_diameter_def(self):
        def AM_mean_radius() -> tuple:
            """ Se establece el radio de referencia de cada escalonamiento (en el contexto del modelo AM). """

            heights = self.cfg.geom['H']
            mr = self.cfg.geom['Rm']
            ref_radius = list()
            for index in range(self.cfg.n_steps):
                A = (2*mr[index*2]) + heights[index*2]
                B = (2*mr[index*2]) - heights[index*2]
                C = (2*mr[index*2]) + heights[index*2+1]
                D = (2*mr[index*2 + 1]) - heights[index*2 + 1]
                E = (2*mr[index*2 + 1]) + heights[index*2 + 2]
                F = (2*mr[index*2 + 1]) - heights[index*2 + 2]
                diameter = (A + B + 2*C + 2*D + E + F)/8
                ref_radius.append(diameter/2)
                ref_radius.append(diameter/2)
            return tuple(ref_radius)

        need_to_adapt = False
        for i, rm1 in enumerate(self.cfg.geom['Rm'][::2]):
            rm2 = self.cfg.geom['Rm'][1::2][i]
            if not rm2*(1-self.cfg.relative_error) < rm1 < rm2*(1+self.cfg.relative_error):
                # Evaluar si los valores son iguales o no
                need_to_adapt = True
        if need_to_adapt:
            self.cfg.edit_geom('Rm', AM_mean_radius())  # Se reescribe la tupla anterior por la tupla adaptada.
            r_corregidos = [i for i in self.cfg.geom['Rm'][::2]]
            str_rm_logger = "El radio medio ha sido corregido acorde al diámetro de referencia del modelo AM: "
            for _ in r_corregidos:
                str_rm_logger += "  ...  %.3f m"
            record.info(str_rm_logger, *r_corregidos)
        return

    def calculating_incidence_stall_incidence_fraction(self, tau_1: float, tau_2: float):
        geom, num = self.cfg.geom, self.crown_num
        alfap_1 = degrees(geom['alfap_i_est'][num//2] if num % 2 == 0 else geom['alfap_i_rot'][num//2])
        s, b, t_max = geom['s'][num], geom['b'][num], geom['t_max'][num]

        a2_a2sc075 = self.alfa2rel_s_c[1](s / b)
        alpha_2_sc075 = tau_2 / a2_a2sc075
        if self.limit_mssg[0]:
            if alfap_1 / alpha_2_sc075 < -1.1 or alfap_1 / alpha_2_sc075 > 1:
                record.warning('La relación B.A. - Ángulo de salida (s/c=0.75) sobrepasa los límites válidos. '
                               'El valor es %.2f y los límites son [-1.1, 1].', alfap_1 / alpha_2_sc075)
                self.limit_mssg[0] = False

        is_sc075 = lineal_interpolation(x_target=alpha_2_sc075, x=alfap_1 / alpha_2_sc075, series=self.is_b1a2_sc_075)
        delta_is = lineal_interpolation(x_target=tau_2, x=s / b, series=self.d_i_s_s_c)
        i_s = delta_is + is_sc075
        i_is = (tau_1 - alfap_1) / i_s

        if self.limit_mssg[1]:
            if i_is > 1.7 or i_is < -4.1:
                record.warning('La relación entre la incidencia y la incidencia de desprendimiento sobrepasa los '
                               'límites de validez del ajuste. El valor es %.3f y los límites son [-4.1, 1.7]', i_is)
                self.limit_mssg[1] = False

        return i_is

    def AM_loss_model_operations(self, tau_1: float, tau_2: float) -> float:
        """ Se determina el valor de Yp según la correlación de AM.
                :param tau_1: alfap_1 más la incidencia del flujo (degrees).
                :param tau_2: alfap_2 más la deflexión del flujo (degrees).
                        :return: Se devuelve el valor de Yp calculado. """

        geom, num = self.cfg.geom, self.crown_num
        alfap_1 = degrees(geom['alfap_i_est'][num//2] if num % 2 == 0 else geom['alfap_i_rot'][num//2])
        s, b, t_max = geom['s'][num], geom['b'][num], geom['t_max'][num]

        i_is = self.calculating_incidence_stall_incidence_fraction(tau_1, tau_2)

        yp_f = self.yp_f_i_f[1](i_is)
        Yp_i0_b1kn = lineal_interpolation(x_target=tau_2, x=s / b, series=self.yp_s_c_b1kn)
        Yp_i0_b1kb2k = lineal_interpolation(x_target=tau_2, x=s / b, series=self.yp_s_c_b1kb2k)

        t_max_b = t_max / b

        if t_max_b < 0.15:
            if self.limit_mssg[2]:
                record.warning('La relación espesor-cuerda es demasiado baja, se ha corregido el valor.')
                self.limit_mssg[2] = False
            t_max_b = 0.15

        elif t_max_b > 0.25:
            if self.limit_mssg[2]:
                record.warning('La relación espesor-cuerda es demasiado alta, se ha corregido el valor.')
                self.limit_mssg[2] = False
            t_max_b = 0.25

        Yp_i0 = (Yp_i0_b1kn + (((alfap_1 / tau_2) ** 2) * (Yp_i0_b1kb2k - Yp_i0_b1kn))) * ((t_max_b / 0.2) **
                                                                                           (alfap_1 / tau_2))
        Yp = Yp_i0*yp_f

        return Yp

    def tau2_corrector(self, num: int, M_out: float):
        tau_in_key = 'alfap_i_est' if num % 2 == 0 else 'alfap_i_rot'
        k_parameter = self.cfg.geom['k'][num]
        height = self.cfg.geom['H'][num+1]
        throat_distance = self.cfg.geom['o'][num]
        pitch = self.cfg.geom['s'][num]

        def tau_mach_low():
            xk_h = self.cfg.geom['X']*k_parameter/height
            c_c = cos(self.cfg.geom[tau_in_key][num//2])/cos(self.outlet_angle_before_mod[num])
            t_t2 = tan(-self.outlet_angle_before_mod[num])
            t_ba = tan(self.cfg.geom[tau_in_key][num//2])
            return -atan(((1-(xk_h*c_c))*t_t2)+(xk_h*c_c*t_ba))  # tau_out (0.0 < M < 0.5) (rads)

        def tau_mach_unit():
            An0, An2 = self.cfg.geom['areas'][num], self.cfg.geom['areas'][num+1]
            Ak = pi * (height + 2*self.cfg.geom['Rm'][num]) * k_parameter
            throat_area = (((throat_distance/pitch) * ((5*An2)+An0) / 6)*(1-(k_parameter/height))) + Ak
            return acos(throat_area / An2)  # tau_out (M = 1.0)  (rads)

        if M_out < 0.5:
            return tau_mach_low()
        else:
            # (M_out - 0.5)/(tau_out - tau_mach_low()) = 0.5 /(tau_mach_unit() - tau_mach_low())
            # Se despeja tau_out
            return tau_mach_low() + (2*(M_out - 0.5)*(tau_mach_unit() - tau_mach_low()))

    def Ainley_and_Mathieson_Loss_Model(self, num: int, tau_1: float, tau_2: float, step_iter_mode=False,
                                        Y_total=0.0) -> None | float:
        """ Se coordinan el resto de instancias para que al llamar a este método externamente se aplique la
        correlación.
                :param step_iter_mode: True cuando se itera el método gen_step para la corrección por Reynolds.
                :param num: Numeración de cada corona de álabes, empezando por 0.
                :param tau_1: Ángulo de entrada del flujo (degrees).
                :param tau_2: Ángulo de salida del flujo (degrees).
                :param Y_total: Parámetro de pérdidas del modelo de pérdidas AM.
                        :return: Se devuelve Y_total y tau_2, excepto cuando step_iter_mode, en dicho caso Y_total es
                                conocido y solo se devuelve tau_2, en ambos casos en radianes. """

        self.crown_num, geom, relative_error = num, self.cfg.geom, self.cfg.relative_error

        lista_local = [geom[i][num] for i in ['areas', 's', 'k', 'H', 'r_h', 'r_t', 'b', 't_e']]
        A_1, s, K_i, H, r_h, r_t, b, t_e = lista_local
        clave_BA = 'alfap_i_est' if num % 2 == 0 else 'alfap_i_rot'
        A_1, A_2 = A_1 * cos(self.cfg.geom[clave_BA][num//2]), geom['areas'][num+1]*cos(radians(tau_2))

        if not step_iter_mode:

            Yp = self.AM_loss_model_operations(tau_1, tau_2)

            if len(self.Yp_preiter) < (self.crown_num + 1):
                self.Yp_preiter.append(Yp)
            else:
                self.Yp_preiter[num] = Yp

            tau_2, tau_1 = radians(tau_2), radians(tau_1)

            x = ((A_2*cos(tau_2)/(A_1*cos(tau_1)))**2)/(1+(r_h/r_t))
            lambda_ = self.sec_losses[1](x)
            tau_m = atan((tan(tau_1)+tan(tau_2))/2)
            C_L = 2*(s/b)*(tan(tau_1) - tan(tau_2))*cos(tau_m)
            Z = ((C_L*b/s)**2)*((cos(tau_2)**2)/(cos(tau_m)**3))
            Ys = lambda_*Z
            Yk = 0.0
            if not K_i > relative_error:
                Yk = 0.5 * (K_i / H) * Z

            Y_total = (Yp + Ys + Yk) * self.ji_Te_te_s[1](t_e / s)

            if len(self.Y_t_preiter) < num + 1:
                self.Y_t_preiter.append(Y_total)
            else:
                self.Y_t_preiter[num] = Y_total

            return Y_total

        else:

            if num % 2 == 0:  # Es estátor
                self.Y_t_stator_iter_mode = Y_total
                self.Yp_iter_mode = self.Yp_preiter[num] * self.Y_t_preiter[num] / Y_total
                return
            else:
                Y_total = self.Y_t_preiter[num] * self.Y_t_stator_iter_mode / self.Y_t_preiter[num - 1]
                self.Yp_iter_mode = self.Yp_preiter[num] * self.Y_t_preiter[num] / Y_total
                return Y_total


class Aungier_Loss_Model(Ainley_and_Mathieson_Loss_Model):
    def __init__(self, cfg: config_class):
        super().__init__(cfg)

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

        alfap_2_e, alfap_2_r = cfg.geom['alfap_o_est'], cfg.geom['alfap_o_rot']
        self.ap_2 = [degrees(alfap_2_e[i//2]) if i % 2 == 0 else degrees(alfap_2_r[i//2]) for i in range(cfg.n_steps*2)]
        alfap_1_e, alfap_1_r = cfg.geom['alfap_i_est'], cfg.geom['alfap_i_rot']
        self.ap_1 = [degrees(alfap_1_e[i//2]) if i % 2 == 0 else degrees(alfap_1_r[i//2]) for i in range(cfg.n_steps*2)]

    def Aungier_operations(self, num, Min, Mout, Re_c, tau_1, V_2x, U_2, V_1x, p_2, pr0_2, d2, d1):
        # El mach indicado debe ser el mach a la salida (Mb)

        t_max, t_e, b_z = self.cfg.geom['t_max'][num], self.cfg.geom['t_e'][num], self.cfg.geom['b_z'][num]
        h_j, o_j = self.cfg.geom['H'][num], self.cfg.geom['o'][num]
        s_j, e_j, b_j = self.cfg.geom['s'][num], self.cfg.geom['e'][num], self.cfg.geom['b'][num]

        if fabs(o_j/s_j) < 1:
            beta_g = degrees(asin(o_j/s_j))
        else:
            beta_g = degrees(asin((o_j/s_j)/fabs(o_j/s_j)))
        Y_TE = (t_e/((s_j*sin(radians(beta_g)))-t_e))**2

        rel_position = self.cfg.geom['gauge_adimensional_position'][num]
        F_AR = 1 - rel_position + (rel_position*(d2*V_2x/(d1*V_1x)))
        F_TE = 1 + (Y_TE * (pr0_2 - p_2) / pr0_2)
        o_s = F_TE*F_AR*o_j/s_j
        if fabs(o_s) < 1:
            beta_g = degrees(asin(o_s))
        else:
            beta_g = degrees(asin(o_s/fabs(o_s)))
        in_asin = o_s*(1+((1-o_s)*((beta_g/90)**2)))
        if fabs(in_asin) > 1:
            in_asin = fabs(in_asin)/in_asin
        delta_0 = (degrees(asin(in_asin))) - beta_g

        if Mout <= 0.5:
            pass
        else:
            X_delta = (2*Mout) - 1
            delta_0 = delta_0*(1 - (10*(X_delta**3)) + (15*(X_delta**4)) - (6*(X_delta**5)))

        taud_1, taud_2 = degrees(tau_1), 90 - (beta_g + delta_0)
        if fabs(taud_2) < self.cfg.relative_error:
            taud_2 = self.cfg.relative_error  # Límite para no dividir por cero luego
        tau_2 = radians(taud_2)

        i_is = self.calculating_incidence_stall_incidence_fraction(taud_1, taud_2)
        k_inc = self.yp_f_i_f[1](i_is)

        k_m = 1 if (s_j / e_j) < 0.105 or Mout < 0.6 else lineal_interpolation(Mout, s_j / e_j, self.km_series)

        Minmod = (Min + 0.566 - fabs(0.566 - Min))/2
        Moutmod = (Mout + 1 - fabs(Mout - 1))/2
        X_param = 2*Minmod / (Moutmod + Minmod + fabs(Moutmod - Minmod))
        k_1 = 1 - (0.625*(Moutmod - 0.2 + fabs(Moutmod - 0.2)))
        k_p = 1 - ((X_param**2)*(1-k_1))

        roughness_heigh = self.cfg.geom['roughness_ptv'][num]
        Re_r = int((100 * b_j / roughness_heigh).__round__(0))

        if Re_c < Re_r:
            if Re_c < 100_000:
                k_Re = sqrt(100_000/Re_c)
            elif Re_c > 500_000:
                k_Re = ((log10(500_000))/((log10(Re_c))**2.58))
            else:
                k_Re = 1
        else:
            if Re_r < 500_000:
                k_Re = 1+((-1+(((log10(50_000))/log10(Re_r))**2.58))*(1-(500_000/Re_c)))
            else:
                k_Re = ((log10(500_000))/((log10(Re_r))**2.58))

        Yp1 = lineal_interpolation(taud_2, s_j / b_j, self.yp_s_c_b1kn)
        Yp2 = lineal_interpolation(taud_2, s_j / b_j, self.yp_s_c_b1kb2k)
        ksi = self.ap_1[num]/taud_2

        Y_TE_002s = (0.02*s_j/((s_j*sin(radians(beta_g)))-(0.02*s_j)))**2
        delta_Y_TE = Y_TE_002s

        Yp = self.cfg.geom['design_factor'] * k_inc * k_m * k_p * k_Re * \
            (((Yp1 + ((ksi**2)*(Yp2-Yp1))) * ((5*t_max/b_j)**ksi))-delta_Y_TE)
        self.Yp_iter_mode = Yp

        CL = 2*(tan(tau_1) + tan(tau_2))*s_j/b_j

        apm = (pi/2) - atan((1/((tan((pi/2) - tau_1 - (1/tan((pi/2) - tau_2))))*2)))
        Z = ((CL*b_j/s_j)**2) * ((cos(tau_2))**2) / ((sin(apm))**3)

        F_AR = b_j/h_j if h_j/b_j >= 2 else (0.5*((2*b_j/h_j)**0.7))
        Y_s_pre = 0.0334 * F_AR * Z * (cos(tau_2)) / cos(radians(self.ap_1[num]))

        k_s = 1 - ((1-k_p)*((b_z/h_j)**2)/(1+((b_j/h_j)**2)))  # bm ¿medio?

        Ys = k_Re*k_s*sqrt((Y_s_pre**2)/(1+(7.5*(Y_s_pre**2))))

        delta_j = self.cfg.geom['delta'][num]
        Y_CL = 0.47*Z*(b_j/h_j)*((delta_j/b_j)**0.78)
        Y_lw = 0.0

        if num % 2 == 1:     # Aplica para rótor
            if 'lashing_wires' in self.cfg.geom and 'wire_diameter' in self.cfg.geom:
                D_lw = self.cfg.geom['wire_diameter']
                N_lw = self.cfg.geom['lashing_wires']
                Re_lw = Re_c*D_lw/b_j
                CD_lw = 1 if Re_lw <= 500_000 else 0.35
                Y_lw = N_lw*CD_lw*D_lw*(V_2x**2)/(h_j*(U_2**2))

        Y = Yp + Ys + Y_CL + Y_TE + Y_lw

        return Y, tau_2
