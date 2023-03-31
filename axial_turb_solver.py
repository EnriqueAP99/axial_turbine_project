"""
En este módulo se crea una clase que, con sus métodos y con herramientas de otros módulos, permite determinar
las condiciones de funcionamiento de la turbina axial que se defina.
"""
from time import time
from loss_model import *
from gas_modeling import *
from math import cos, sin, tan, fabs, sqrt, atan, asin, acos, log, degrees


def solver_timer(solver_method):
    """ Decorador que gestiona la función interna del método problem_solver de la clase solver_process, para
    conocer el tiempo de cálculo.
            :param solver_method: Función que se decora.
                    :return: Se devuelve wrapper_t."""
    def wrapper_t(*args):
        """ Se evalúa cuánto tiempo tarda en ejecutarse el solver.
                        :param args: Argumentos que requiera la función decorada.
                                    :returns: Se devuelve o se almacena el output de la función decorada, según el
                                             modo de funcionamiento que aplique. """
        t_1 = time()
        list_from_ps: list
        solver_method(*args)
        t_2 = (time() - t_1).__trunc__()
        m, s = divmod(t_2, 60)
        logger.info('Tiempo de cálculo: %s minutos y %s segundos.', m, s)
        return
    return wrapper_t


def Reynolds_correction(tol: float, loss_model: str):
    """ Decorador del decorador real, permite referenciar los parámetros necesarios para manipular la salida de la
    función decorada.
                :param tol: Error relativo máximo que se establece en los cálculos.
                :param loss_model: Cadena de caracteres identificadora del modelo de pérdidas establecido.
                            :return: Se devuelve el decorador real. """
    def Reynolds_corrector(step_inner_funct):
        """ Decorador real, gestiona la función interna del método gen_steps de la clase solver_process y posibilita
         aplicar la corrección por el número de Reynolds que en ocasiones requiere el modelo de pérdidas de Ainley and
         Mathieson.
                    :param step_inner_funct: Función que se decora.
                                :return: Se devuelve el wrapper_r. """
        def corrector(eta_TT, Re, xi_est, p_seed, rho_seed) -> list:
            """ Función que aplica la corrección de Reynolds cuando es llamada por wrapper_r. La solución se determina
                aplicando el teorema de Bolzano y Régula Falsi. Se va a usar como semilla de la siguiente iteración los
                valores que resultan de la anterior.
                    :param rho_seed: Lista con semillas para los dos valores de densidad a la salida de cada corona.
                    :param p_seed: Lista con semillas para los dos valores de presión a la salida de cada corona.
                    :param eta_TT: Rendimiento total a total del escalonamiento.
                    :param Re: Número de Reynolds.
                    :param xi_est: Coeficiente adimensional de pérdidas.
                            :return: Se devuelve la lista de variables que corresponda según el modo de
                                    funcionamiento que se defina."""
            eta_TT_obj = 1-(((1-eta_TT)/200_000)*Re)
            bolz_c, xi_e1, xi_e2, f1, f2, first, alt = 1.0, xi_est*0.9, xi_est, 0.0, 0.0, True, 0
            p_seed_1 = p_seed_2 = p_seed_c = p_seed
            rho_seed_1 = rho_seed_2 = rho_seed_c = rho_seed
            while bolz_c > 0:
                if not bool(alt % 2):
                    sif1 = step_inner_funct(True, False, xi_e1, p_seed_1, rho_seed_1)
                    f1, p_seed_1, rho_seed_1 = sif1[0] - eta_TT_obj, sif1[1], sif1[2]
                else:
                    sif2 = step_inner_funct(True, False, xi_e2, p_seed_2, rho_seed_2)
                    f2, p_seed_2, rho_seed_2 = sif2[0] - eta_TT_obj, sif2[1], sif2[2]
                bolz_c = f1*f2
                if bolz_c < 0:
                    pass
                else:
                    if not bool(alt % 2):
                        xi_e2 = 1.1*xi_e2
                    else:
                        xi_e1 = 0.9*xi_e1
            xi_ec = float(xi_e2 - f2*(xi_e2-xi_e1)/(f2-f1))
            sifc = step_inner_funct(True, False, xi_ec, p_seed_c, rho_seed_c)
            fc, p_seed_c, rho_seed_c = sifc[0] - eta_TT_obj, sifc[1], sifc[2]
            while fabs(fc/eta_TT_obj) > tol:
                if not first:
                    sif1 = step_inner_funct(True, False, xi_e1, p_seed_1, rho_seed_1)
                    f1, p_seed_1, rho_seed_1 = sif1[0] - eta_TT_obj, sif1[1], sif1[2]
                    sif2 = step_inner_funct(True, False, xi_e2, p_seed_2, rho_seed_2)
                    f2, p_seed_2, rho_seed_2 = sif2[0] - eta_TT_obj, sif2[1], sif2[2]
                    xi_ec = xi_e2 - (f2*(xi_e2-xi_e1)/(f2-f1))
                    sifc = step_inner_funct(True, False, xi_ec, p_seed_c, rho_seed_c)
                    fc, p_seed_c, rho_seed_c = sifc[0] - eta_TT_obj, sifc[1], sifc[2]
                else:
                    first = False
                if fc*f2 < 0:
                    xi_e1, p_seed_1, rho_seed_1 = xi_ec, p_seed_c, rho_seed_c
                elif fc*f1 < 0:
                    xi_e2, p_seed_2, rho_seed_2 = xi_ec, p_seed_c, rho_seed_c
            _, _, _, ll_1 = step_inner_funct(True, True, xi_ec, p_seed_c, rho_seed_c)
            return ll_1

        def wrapper_r():
            """ Función que evalúa el modelo de pérdidas que se ha definido y, si es el de Ainley and Mathieson, aplica
            una corrección en caso de que el número de Reynolds sea inferior a 50.000.
                            :return: Se devuelve la lista de variables que se procesan, según el modo que se defina. """
            if loss_model == 'ainley_and_mathieson':
                Re, eta_TT, xi_est, p_seed, rho_seed, ll_1 = step_inner_funct()
                logger.info('Reynolds: %d', Re)
                if Re < 50_000:
                    ll_1 = corrector(eta_TT, Re, xi_est, p_seed, rho_seed)
            else:
                ll_1 = step_inner_funct()
            return ll_1
        return wrapper_r
    return Reynolds_corrector


class solver_process:
    """ Clase que define un objeto que agrupa y coordina al conjunto de procesos/métodos/atributos que conforman
    el procedimiento resolutivo propuesto para determinar unas condiciones de funcionamiento de una turbina axial."""
    def __init__(self, config: configuration_parameters):
        """ :param config: Objeto que agrupa lo relativo a la configuración establecida para la ejecución del solver."""
        self.vmmr = []  # Almacena ciertas variables, para facilitar la comunicación de sus valores
        self.cfg = config
        self.rho_seed = None   # Para aligerar los cálculos para variaciones pequeñas de las variables de entrada
        self.p_seed = None
        # prd: Atributo que caracteriza y modela las propiedades de los productos de la combustión
        self.prd = mixpm(config.thermo_mode)
        self.prd.setmix(config.C_atoms, config.H_atoms, config.N)
        self.AM_object = None
        if config.loss_model == 'ainley_and_mathieson':
            self.AM_object = AM_loss_model(config)

    def problem_solver(self, T_in: float, p_in: float, n: float, C_inx=None, m_dot=None):
        """Esta función inicia la resolución del problema definido por la geometría configurada y las variables
        requeridas como argumento.
                :param m_dot:
                :param T_in: Temperatura a la entrada de la turbina (K).
                :param p_in: Presión a la entrada de la turbina (Pa).
                :param C_inx: Velocidad a la entrada de la turbina (Supuesta completamente axial) (m/s).
                :param n: Velocidad de giro (rpm).
                        :return: Si fast_mode se devuelven los valores a la salida de temperatura (K), presión (Pa),
                                velocidad (m/s) y ángulo del flujo con la dirección axial (degrees), en caso contrario
                                no se devuelve nada."""
        ps_list = []

        @solver_timer
        def inner_solver() -> None:
            nonlocal m_dot, C_inx, ps_list
            tol = self.cfg.TOL
            logger.info('El error relativo establecido es: %s', tol)
            rho_in = self.prd.get_props_by_Tpd({'T': T_in, 'p': p_in}, 'd')
            if m_dot is None:
                m_dot = rho_in * self.cfg.geom['areas'][0] * C_inx
            elif C_inx is None:
                C_inx = m_dot / (rho_in * self.cfg.geom['areas'][0])
            else:
                logger.critical('Se debe establecer uno de los parámetros opcionales "Cinx" ó "m_dot".')
                sys.exit()
            s_in = self.prd.get_props_by_Tpd({'T': T_in, 'p': p_in}, 's')
            h_in = self.T_to_h_comparator(T_in, key_prop2='p', value_prop2=p_in)
            h_0in = h_in + ((10 ** (-3)) * (C_inx ** 2) / 2)
            ps_list = [T_in, p_in, rho_in, s_in, h_in, h_0in, C_inx]
            for i in range(self.cfg.n_step):
                list_i = ps_list[i-1] if i > 0 and not self.cfg.fast_mode else ps_list
                args = i, list_i[0], list_i[1], list_i[6], list_i[3], list_i[4], list_i[5], m_dot, n, list_i[2]
                if self.cfg.fast_mode:
                    ps_list = self.step_block(*args)
                else:
                    if i == 0:
                        ps_list = [self.step_block(*args)]
                    else:
                        ps_list += [self.step_block(*args)]
            if not self.cfg.fast_mode:  # Los subindices A y B indican, resp., los pts. inicio y fin de la turbina.
                p_B, s_B, h_B, h_0B = [ps_list[-1][1]] + ps_list[-1][3:6]
                h_in = self.T_to_h_comparator(T=T_in, key_prop2='p', value_prop2=p_in)
                h_0A = h_in + ((10 ** (-3)) * (C_inx ** 2) / 2)
                DELTA_h = h_B - h_in
                w_total = h_0A - h_0B
                P_total = w_total * m_dot
                s_A = self.prd.get_props_by_Tpd({'T': T_in, 'p': p_in}, 's')
                p_0B, T_0B = self.Zero_pt_calculator(p_B, s_B, h_0B)
                T_0Bss = self.prd.get_props_with_hs({'s': s_A, 'p': p_0B}, {'T': T_0B * 0.9}, tol)
                h_0Bss = self.T_to_h_comparator(T_0Bss, key_prop2='p', value_prop2=p_0B)
                Y_maq = h_0B - h_0Bss
                w_ss_total = h_0A - h_0Bss
                eta_maq = w_total / w_ss_total
                p_0A, T_0A = self.Zero_pt_calculator(p_in, s_A, h_0A)
                eta_p = log(1 - (eta_maq * (1 - (T_0Bss / T_0A))), 10) / log(T_0Bss / T_0A, 10)
                r_turbina = p_0A / p_0B
                ps_list += [[DELTA_h, w_total, P_total, s_A, s_B, p_0B, T_0B, T_0Bss, h_0Bss, Y_maq, w_ss_total,
                             eta_maq, p_0A, T_0A, eta_p, r_turbina, m_dot]]
            self.vmmr = ps_list
            return

        inner_solver()
        if self.cfg.fast_mode:
            return ps_list[0], ps_list[1], ps_list[6], degrees(ps_list[7])
        else:
            return

    def step_block(self, count: int, T_1: float, p_1: float, C_1: float, s_1: float, h_1: float, h_01: float,
                   m_dot: float, n: float, rho_1: float) -> list:
        """ Método que se llama de manera recurrente por cada escalonamiento definido, para determinar las propiedades
        del fluido en los puntos de interés del mismo.

        Punto 1: Entrada del escalonamiento

        Punto 2: Punto intermedio del escalonamiento

        Punto 3: Salida del escalonamiento

                :param count: Numeración del escalonamiento comenzando por 0.
                :param T_1: Temperatura a la entrada del escalonamiento (K).
                :param p_1: Presión a la entrada del escalonamiento (Pa).
                :param C_1: Velocidad absoluta a la entrada del escalonamiento (m/s).
                :param s_1: Entropía a la entrada del escalonamiento (kJ/K).
                :param h_1: Entalpía a la entrada del escalonamiento (kJ/Kg).
                :param h_01: Entalpía total a la entrada del escalonamiento (kJ/Kg).
                :param m_dot: Flujo másico a través de la turbina (kg/s).
                :param n: Velocidad de giro tangencial (rpm).
                :param rho_1: Densidad a la entrada (kg/m^3).
                        :return: Se devuelve una lista de valores de variables diferente según el modo establecido. """
        @Reynolds_correction(self.cfg.TOL, self.cfg.loss_model)
        def inner_funct(iter_mode=False, iter_end=False, xi_est=0.0, p_seed=None, rho_seed=None):
            """ Esta función interna se crea para poder comunicar al decorador instancias de la clase.
                    :param rho_seed: Densidad que se emplea como valor semilla en bucle_while_x2 (kg/m^3).
                    :param p_seed: Presión que se emplea como valor semilla en el cálculo de propiedades.
                    :param xi_est: Valor opcional que permite al decorador aplicar recursividad para efectuar la
                                  corrección por dependencia de Reynolds.
                    :param iter_mode: Permite diferenciar si se está aplicando recursividad para la corrección por
                                 dependencia de Reynolds y así conocer las instrucciones más convenientes.
                    :param iter_end: Permite omitir cálculos innecesarios durante la corrección por dependencia de
                                    Reynolds.
                            :return: Se devuelven diferentes variables que se requieren según la situación."""
            if p_seed is None and rho_seed is None:
                if self.rho_seed is not None and len(self.p_seed) == self.cfg.n_step:
                    p_seed, rho_seed = self.p_seed[count], self.rho_seed[count]
                else:
                    p_seed, rho_seed = [p_1*0.95, p_1*0.9], [rho_1*0.9, rho_1*0.9]
                    if count == 0:
                        self.p_seed = []
                        self.rho_seed = []
            A_tpl = (self.cfg.geom['areas'][count * 2], self.cfg.geom['areas'][count * 2 + 1],
                     self.cfg.geom['areas'][count * 2 + 2])
            Re, eta_TT = 0.0, 0.0
            C_1x = m_dot / (rho_1 * A_tpl[0])
            logger.info('La velocidad axial establecida a la entrada del escalonamiento %s es: %.2f m/s', count+1, C_1x)
            if C_1x > C_1:  # Evitar math domain error producido por error de cálculo cuando entrada completamente axial
                C_1x = C_1
            alfa_1 = acos(C_1x / C_1)
            h_02 = h_01
            if not iter_mode:
                p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2, xi_est, Re = \
                    self.blade_outlet_calculator('est', count, A_tpl[1], alfa_1, h_02, m_dot, s_1, p_seed[0],
                                                 rho_seed[0], step_iter_mode=iter_mode)
            else:
                p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2, Re = \
                    self.blade_outlet_calculator('est', count, A_tpl[1], alfa_1, h_02, m_dot, s_1, p_seed[0],
                                                 rho_seed[0], step_iter_mode=iter_mode, xi=xi_est)
            s_2 = self.prd.get_props_by_Tpd({'p': p_2, 'T': T_2}, 's')
            # En las líneas que siguen se determinan los triángulos de velocidades
            U = self.cfg.geom['Rm'][(count*2)+1]*n*2*pi/60
            C_2u = C_2 * sin(alfa_2)
            omega_2u = C_2u - U
            omega_2x = C_2x
            beta_2 = atan(omega_2u / C_2x)
            omega_2 = C_2x / cos(beta_2)
            h_r3 = h_r2 = h_2 + ((10 ** (-3)) * (omega_2 ** 2) / 2)
            p_3, h_3, T_3, omega_3, rho_3, h_3s, T_3s, C_3x, M_3, beta_3, xi_rot = \
                self.blade_outlet_calculator('rot', count, A_tpl[2], beta_2, h_r3, m_dot, s_2, p_seed[1], rho_seed[1],
                                             U=U, step_iter_mode=iter_mode, Re=Re)
            omega_3u = omega_3 * sin(beta_3)
            omega_3x = omega_3 * cos(beta_3)
            C_3u = omega_3u - U
            C_3 = sqrt(C_3u ** 2 + C_3x ** 2)
            alfa_3 = asin(C_3u / C_3)
            s_3 = self.prd.get_props_by_Tpd({'T': T_3, 'p': p_3}, 's')
            h_03 = h_3 + (10 ** (-3)) * ((C_3 ** 2) / 2)
            local_list_1 = [T_3, p_3, rho_3, s_3, h_3, h_03, C_3, alfa_3]
            if h_02 - h_03 < 0:
                logger.error('No se extrae energía del fluido.')
            if len(self.p_seed) < self.cfg.n_step:
                if not iter_mode:
                    self.p_seed.append([p_2, p_3].copy())
                    self.rho_seed.append([rho_2, rho_3].copy())
                elif iter_end:
                    self.p_seed[-1] = [p_2, p_3].copy()
                    self.rho_seed[-1] = [rho_2, rho_3].copy()
            # Se determina en estas líneas el rendimiento total a total para aplicar la corrección:
            if self.cfg.loss_model == 'ainley_and_mathieson' and self.cfg.fast_mode:
                tol = self.cfg.TOL
                w_esc = h_02 - h_03
                p_03, T_03 = self.Zero_pt_calculator(p_3, s_3, h_03)
                T_03ss = self.prd.get_props_with_hs({'s': s_1, 'p': p_03}, {'T': T_03}, tol)
                h_03ss = self.T_to_h_comparator(T_03ss, key_prop2='p', value_prop2=p_03)
                Y_esc = h_03 - h_03ss
                eta_TT = w_esc / (w_esc + Y_esc)
            if not self.cfg.fast_mode and (iter_end or not iter_mode):
                tol = self.cfg.TOL
                Y_est = xi_est * ((10 ** (-3)) * (C_2 ** 2) / 2)
                Y_rot = xi_rot * ((10 ** (-3)) * (omega_3 ** 2) / 2)
                w_esc = h_02 - h_03
                p_03, T_03 = self.Zero_pt_calculator(p_3, s_3, h_03)
                T_03ss = self.prd.get_props_with_hs({'s': s_1, 'p': p_03}, {'T': T_03}, self.cfg.TOL)
                h_03ss = self.T_to_h_comparator(T_03ss, key_prop2='p', value_prop2=p_03)
                Y_esc = h_03 - h_03ss
                eta_TT = w_esc / (w_esc + Y_esc)
                C_1u = sqrt(C_1 ** 2 - C_1x ** 2)
                a_1 = self.prd.get_a(T_1, p_1)
                M_1 = C_1 / a_1
                a_3 = self.prd.get_a(T_3, p_3)
                M_3 = C_3 / a_3
                Pot_esc = w_esc * m_dot
                GR = (h_2 - h_3) / w_esc
                PSI = w_esc / U ** 2
                PHI_2 = C_2x / U
                PHI_3 = C_3x / U
                C_2s = sqrt((h_02 - h_2s) * 2 / (10 ** (-3)))
                omega_3s = sqrt((h_r3 - h_3s) * 2 / (10 ** (-3)))
                speed_loss_coeff_E = C_2 / C_2s
                speed_loss_coeff_R = omega_3 / omega_3s
                DELTA_h_esc = h_3 - h_1
                p_02, T_02 = self.Zero_pt_calculator(p_2, s_2, h_02)
                T_02s = self.prd.get_props_with_hs({'s': s_1, 'p': p_02}, {'T': T_02 * 0.9}, tol)
                h_02s = self.T_to_h_comparator(T_02s, key_prop2='p', value_prop2=p_02)
                T_3ss = self.prd.get_props_with_hs({'p': p_3, 's': s_1}, {'T': T_3 * 0.95}, tol)
                h_3ss = self.T_to_h_comparator(T_3ss, key_prop2='p', value_prop2=p_3)
                T_03s = self.prd.get_props_with_hs({'p': p_03, 's': s_2}, {'T': T_1 * 0.9}, tol)
                h_03s = self.T_to_h_comparator(T_03s, key_prop2='p', value_prop2=p_03)
                Y_esc = h_03 - h_03ss
                w_ss_esc = w_esc + Y_esc
                T_03 = self.h_to_T_comparator(h_03, key_prop2='p', value_prop2=p_03)
                T_02 = self.h_to_T_comparator(h_02, key_prop2='p', value_prop2=p_02)
                w_s_esc = w_esc + h_03 - h_03s
                eta_TE = w_esc / (w_esc + Y_esc + ((10 ** (-3)) * (C_3 ** 2) / 2))
                # Aunque resulta más relevante la relación isentrópica, es posible calcular el rendimiento politrópico
                # Se emplean para ello los valores de remanso
                p_01, T_01 = self.Zero_pt_calculator(p_1, s_1, h_01)
                eta_p_esc = log(1 - (eta_TT * (1 - (T_03ss / T_01))), 10) / log(T_03ss / T_01, 10)
                r_esc = p_01 / p_03
                local_list_1 += [C_3x, C_3u, T_03, p_03, h_r3, h_3s, T_3s, omega_3, omega_3x, omega_3u, beta_3,
                                 M_3, PHI_3, T_2, p_2, rho_2, s_2, h_2, h_02, C_2, C_2x, C_2u, alfa_2, T_02, p_02, h_r2,
                                 h_2s, T_2s, omega_2, omega_2x, omega_2u, beta_2, M_2, PHI_2, T_1, p_1, rho_1, s_1, h_1,
                                 h_01, C_1, C_1x, C_1u, alfa_1, T_01, p_01, M_1, Y_est, xi_est, Y_rot, xi_rot, w_esc,
                                 Pot_esc, eta_p_esc, r_esc, GR, PSI, speed_loss_coeff_E, speed_loss_coeff_R,
                                 DELTA_h_esc, eta_TT, eta_TE, Y_esc, w_s_esc, w_ss_esc, C_2s, T_02s, h_02s, omega_3s,
                                 T_3ss, T_03s, T_03ss, h_3ss, h_03s, h_03ss, U]
                # Ordenar y eleminar las que se pueden calcular mediante otras sin usar este módulo
                # eta_TT se debe enviar siempre que iter_mode para evaluar la condición del decorador
                if not iter_mode and self.cfg.loss_model == 'ainley_and_mathieson':
                    return Re, eta_TT, xi_est, [p_2, p_3], [rho_2, rho_3], local_list_1.copy()
                elif self.cfg.loss_model == 'ainley_and_mathieson':
                    return eta_TT, [p_2, p_3], [rho_2, rho_3], local_list_1.copy()
                else:
                    return local_list_1.copy()
            else:
                if not iter_mode and self.cfg.loss_model == 'ainley_and_mathieson':
                    return Re, eta_TT, xi_est, [p_2, p_3], [rho_2, rho_3], local_list_1
                elif self.cfg.loss_model == 'ainley_and_mathieson':
                    return eta_TT, [p_2, p_3], [rho_2, rho_3], local_list_1
                else:
                    return local_list_1
        ll_1 = inner_funct()
        return ll_1

    def blade_outlet_calculator(self, blade: str, counter: int, area_b: float, tau_a: float,
                                h_tb: float, m_dot: float, s_a: float, p_seed: float, rho_seed: float, U=None,
                                step_iter_mode=False, xi=0.0, Re=0.0):
        """ Se hace un cálculo iterativo para conocer las propiedades a la salida del estátor y del rótor (según el
        caso, estátor / rótor, se proporciona el valor de la entalpía total / rotalpía y del ángulo que forma la
        velocidad absoluta / relativa del fluido con la dirección del eje de la turbina axial, respectivamente).

        Punto a: Sección al principio del álabe

        Punto b: Sección al final del álabe

            :param Re: Número de Reynolds, se recibe cuando se calcula el rótor.
            :param xi: Coeficiente adimensional de pérdidas de la corona en cuestión.
            :param U: Velocidad tangencial de giro del rótor, argumento opcional (m/s).
            :param step_iter_mode: Permite diferenciar si se está aplicando recursividad para la corrección por
                               dependencia de Reynolds y así conocer las instrucciones más convenientes.
            :param tau_a: Ángulo del movimiento absoluto/relativo de entrada del fluido con la dirección axial
                         (rads).
            :param counter: Contador de cada escalonamiento, comienza por 0.
            :param blade: Diferenciador del tipo de álabe, puede ser 'est' o 'rot'.
            :param area_b: El área de la sección de paso al final de la corona (m^2).
            :param h_tb: Entalpía total / rotalpía a la salida del álabe (kJ/kg).
            :param m_dot: Flujo másico (kg/s).
            :param s_a: Entropía en a (kJ/kgK).
            :param rho_seed: Densidad que se emplea como semilla, debe aproximar la densidad en b (kg/m^3).
            :param p_seed: Presión que se emplea como semilla, debe aproximar la presión en b (Pa).
                    :return: Se devuelven las variables que contienen las propiedades a la salida que se han
                    determinado (p_b, h_b, T_b, U_b, rho_b, h_bs, T_bs, C_bx). """
        print('\n')
        rho_b = rho_seed
        M_b = p_b = h_b = U_b = h_bs = T_bs = C_bx = T_b = tau_b = Y_total = 0.0
        rho_bp, rel_diff, tol, geom = rho_b, 1.0, self.cfg.TOL, self.cfg.geom
        num = counter*2 + (0 if blade == 'est' else 1)
        if self.cfg.loss_model == 'soderberg_correlation':
            s, H, b = geom['s'][num], geom['H'][num], geom['b'][num]
            args = [blade, geom['alfap_i_est'][counter], geom['alfap_i_rot'][counter], H, b]
            xi = geom['A_rel'][num]*Soderberg_correlation(*args)
            tau_b = geom['alfap_o_est'][counter] if blade == 'est' else geom['alfap_o_rot'][counter]
        elif self.cfg.loss_model == 'ainley_and_mathieson':
            if not step_iter_mode:
                Y_total, tau_b = self.AM_object.Ainley_and_Mathieson_Loss_Model(num, tol, degrees(tau_a), False)
            else:
                tau_b = geom['alfap_o_est'][counter] if blade == 'est' else geom['alfap_o_rot'][counter]
        # p: iteración previa .... b: estado que se quiere conocer, a la salida del álabe
        while fabs(rel_diff) > tol:
            C_bx = m_dot / (area_b * rho_b)  # C_bx: velocidad axial a la salida
            diff_tau_b = tau_b
            while fabs(diff_tau_b/tau_b) > tol:
                U_b = C_bx / cos(tau_b)  # U_b: vel. absoluta o relativa ... tau: alfa o beta ... según el caso
                h_b = h_tb-((10**(-3))*(U_b**2)/2)
                # Se aplica conservación de la entalpía total/rotalpía ... según el caso
                # (no se puede determinar otra variable con h2s y s1 con PyroMat)
                p_b = self.prd.get_props_with_hs({'d': rho_b, 'h': h_b}, {'p': p_seed}, tol)
                T_b = self.h_to_T_comparator(h_b, key_prop2='d', value_prop2=rho_b)
                a_b, _, _, gamma_b = self.prd.get_a(T_b, p_b, extra=True)
                T_bs = self.prd.get_props_with_hs({'p': p_b, 's': s_a}, {'T': T_b}, tol)
                h_bs = self.T_to_h_comparator(T_bs, key_prop2='p', value_prop2=p_b)
                if blade == 'est':
                    M_b = U_b/a_b
                    if not step_iter_mode:
                        s, H = geom['s'][num], geom['H'][num]
                        Re = Reynolds(rho_b, U_b, T_b, s, H, geom['alfap_o_est'][counter], self.prd)
                elif blade == 'rot':
                    C_b = sqrt((U-(C_bx/tan(tau_b)))**2 + (C_bx**2))
                    M_b = C_b/a_b
                if self.cfg.loss_model == 'soderberg_correlation':
                    xi = ((1E5 / Re) ** 0.25) * xi
                    diff_tau_b = 0.0
                elif self.cfg.loss_model == 'ainley_and_mathieson' and not step_iter_mode:
                    xi = Y_total/(1 + (0.5*gamma_b*(M_b**2)))
                    diff_tau_b = 0.0
                else:
                    Y_total = xi*(1 + (0.5*gamma_b*(M_b**2)))
                    args = [num, tol, degrees(tau_a), True, Y_total]
                    tau_b_n = self.AM_object.Ainley_and_Mathieson_Loss_Model(*args)
                    diff_tau_b = tau_b - tau_b_n
                    tau_b = tau_b_n
            h_b = ((10 ** (-3)) * xi * (U_b ** 2) / 2) + h_bs
            rho_b = self.prd.get_props_with_hs({'p': p_b, 'h': h_b}, {'d': rho_b}, tol)
            rel_diff = (rho_bp - rho_b) / rho_b
            rho_bp = rho_b
            logger.info('Densidad (kg/m^3): %.10f    Error relativo: %.10f', rho_b, rel_diff)
        return_vars = [p_b, h_b, T_b, U_b, rho_b, h_bs, T_bs, C_bx, M_b, tau_b]
        if blade == 'est':
            if not step_iter_mode:
                return *return_vars, xi, Re
            else:
                return *return_vars, Re
        else:
            return *return_vars, xi

    def Zero_pt_calculator(self, p_x: float, s_x: float, h_0x: float):
        """ Este método es para determinar presiones y temperaturas de remanso.

                :param p_x: Presión en la sección x (Pa).
                :param s_x: Entropía en la sección x (kJ/kgK).
                :param h_0x: Entalpía total en la sección x (kJ/kg).
                        :return: Devuelve la presión total (Pa) y la temperatura total (K) en una sección x. """
        p_0x, T_0x, end, tol = p_x * 1.1, 0.0, False, self.cfg.TOL
        while not end:
            T_0x = self.h_to_T_comparator(h_0x, key_prop2='p', value_prop2=p_0x)
            if self.prd.mode == "ig":
                p_0x = self.prd.get_props_with_hs({'T': T_0x, 's': s_x}, {'p': p_0x}, tol)
                end = True
            else:
                p_0x_iter = self.prd.get_props_with_hs({'T': T_0x, 's': s_x}, {'p': p_0x}, tol)
                if fabs(p_0x - p_0x_iter) / p_0x < tol:
                    end = True
                p_0x = p_0x_iter
        return p_0x, T_0x

    def h_to_T_comparator(self, h: float, **kwargs):
        """ Este método actúa de una manera u otra según si se emplea el modelo de gas ideal o no, para evitar el
        posible error en que se indiquen dos propiedades y se dé el caso en que se emplee el modelo de gas ideal y
        la primera propiedad indicada no sea la temperatura (ver módulo 'gas_modeling.py'). No se acepta entropía como
        entrada.
                :param h: Entalpía indicada (kJ/kg).
                :param kwargs: Diccionario con el caracter identificador de la segunda propiedad indicada
                              (key_prop2) y con el valor de la misma (value_prop2).
                        :return: Se devuelve la temperatura calculada (K). """
        key_prop2 = kwargs.get('key_prop2', '')
        value_prop2 = kwargs.get('value_prop2', 0)
        if self.prd.mode == "ig":
            T = self.prd.get_props_with_hs({'h': h}, {'T': 1600}, self.cfg.TOL)
        else:
            T = self.prd.get_props_with_hs({'h': h, key_prop2: value_prop2}, {'T': 1600}, self.cfg.TOL)
        return T

    def T_to_h_comparator(self, T: float, **kwargs):
        """ Este método actúa de una manera u otra según si se emplea el modelo de gas ideal o no, para evitar el
        posible error en que se indiquen dos propiedades y se dé el caso en que se emplee el modelo de gas ideal y la
        primera propiedad indicada no sea la temperatura (ver módulo 'gas_modeling.py').

        No se acepta entropía como entrada.
                :param T: Temperatura indicada (K).
                :param kwargs: Diccionario con el caracter identificador de la segunda propiedad indicada (key_prop2) y
                              con el valor de la misma (value_prop2).
                        :return: Se devuelve la entalpía calculada (kJ/kg). """
        key_prop2 = kwargs.get('key_prop2', '')
        value_prop2 = kwargs.get('value_prop2', 0.0)
        if self.prd.mode == "ig":
            h = self.prd.get_props_by_Tpd({'T': T}, 'h')
        else:
            h = self.prd.get_props_by_Tpd({'T': T, key_prop2: value_prop2}, 'h')
        return h


if __name__ == '__main__':
    fast_mode = True
    settings = configuration_parameters(rel_error=1E-6, number_steps=2, thermo_mode="ig",
                                        loss_model_id='ainley_and_mathieson', C_atoms=12, H_atoms=23.5, N=4,
                                        fast_mode=fast_mode)
    # alfap_1, theta_e, betap_2, theta_r, cuerda, R_average, alturas
    settings.set_geometry([0, 20], [70, 90], [50, 10], [120, 30], 0.03, 0.3, H=[0.011, 0.018, 0.028, 0.0320, 0.0400],
                          A_rel=0.75, t_max=0.008, r_r=0.003, r_c=0.002, t_e=0.004, K=0.0)
    solver = solver_process(settings)
    # T_in, P_in, C_xin, U
    if fast_mode:
        T_salida, p_salida, C_salida, alfa_salida = solver.problem_solver(1800, 1_200_000, 6_500, C_inx=200)
        print(' T_out', T_salida, '\n', 'P_out', p_salida, '\n', 'C_out', C_salida, '\n', 'alfa_out', alfa_salida)
    else:
        solver.problem_solver(1800, 1_200_000, 6_500, C_inx=200)
