"""
En este módulo se crea una clase que, con sus métodos y con herramientas de otros módulos, permite determinar
las condiciones de funcionamiento de la turbina axial que se defina.
"""
from time import time
from loss_model import *
from config_class import gas_model_to_solver
import sys
from math import cos, sin, fabs, sqrt, atan, asin, acos, log, degrees, pi


def solver_timer(solver_method):
    """ Decorador que gestiona la función interna del método problem_solver de la clase solver, para
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
        registro.info('Tiempo de cálculo: %s minutos y %s segundos.', m, s)
        return
    return wrapper_t


def Reynolds_correction(corrector_tol: float, loss_model: str,
                        step_corrector_memory: None | list[float, list[float, float]]):
    """ Decorador del decorador real, permite referenciar los parámetros necesarios para manipular la salida de la
    función decorada.
                :param corrector_tol: Error relativo máximo que se permite en el rendimiento corregido.
                :param loss_model: Cadena de caracteres identificadora del modelo de pérdidas establecido.
                :param step_corrector_memory: Lista con las variables xi_ec y rho_seed_c de la ejecución previa.
                            :return: Se devuelve el decorador real. """

    def Reynolds_corrector(step_inner_funct):
        """ Decorador real, gestiona la función interna del método gen_steps de la clase solver y posibilita
         aplicar la corrección por el número de Reynolds que en ocasiones requiere el modelo de pérdidas de Ainley and
         Mathieson.
                    :param step_inner_funct: Función que se decora.
                                :return: Se devuelve el wrapper_r. """

        def corrector(eta_TT: float, Re: int, xi_est: float, rho_seed: float) -> list:
            """ Función que aplica la corrección de Reynolds cuando es llamada por wrapper_r. La solución se determina
                aplicando el teorema de Bolzano y Régula Falsi. Se va a usar como semilla de la siguiente iteración los
                valores que resultan de la anterior.
                    :param rho_seed: Lista con semillas para los dos valores de densidad a la salida de cada corona.
                    :param eta_TT: Rendimiento total a total del escalonamiento.
                    :param Re: Número de Reynolds.
                    :param xi_est: Coeficiente adimensional de pérdidas en el estátor.
                            :return: Se devuelve la lista de variables que corresponda según el modo de
                                    funcionamiento que se defina."""

            # Almacenar solución y comprobar si existe antes de nada y emplearla como semilla.

            eta_TT_obj = 1 - (((1-eta_TT) / (200_000**(-1/5))) * (Re**(-1/5)))
            if step_corrector_memory is not None:
                xi_e1 = step_corrector_memory[0]*(1 - corrector_tol)
                xi_e2 = step_corrector_memory[0]*(1 + corrector_tol)
                rho_seed_1 = rho_seed_2 = rho_seed_c = step_corrector_memory[1]
            else:
                xi_e0 = xi_est * (Re**(-1/5)) / (200_000**(-1/5))
                xi_e1 = xi_e0 * 0.99
                xi_e2 = xi_e0 * 1.01
                rho_seed_1 = rho_seed_2 = rho_seed_c = rho_seed
            bolz_c, f1, f2, ff = 1.0, None, None, 1.0

            while bolz_c > 0:
                registro.info('Buscando el rango que garantice encontrar la solución.')

                if f1 is None and f2 is None:
                    sif1 = step_inner_funct(True, False, xi_e1, rho_seed_1)
                    f1, rho_seed_1 = sif1[0] - eta_TT_obj, sif1[1]
                    sif2 = step_inner_funct(True, False, xi_e2, rho_seed_2)
                    f2, rho_seed_2 = sif2[0] - eta_TT_obj, sif2[1]
                elif ff > 0:
                    sif1 = step_inner_funct(True, False, xi_e1, rho_seed_1)
                    f1, rho_seed_1 = sif1[0] - eta_TT_obj, sif1[1]
                else:
                    sif2 = step_inner_funct(True, False, xi_e2, rho_seed_2)
                    f2, rho_seed_2 = sif2[0] - eta_TT_obj, sif2[1]

                ff = (f2 - f1)/(xi_e2 - xi_e1)
                bolz_c = f1*f2

                if bolz_c < 0:
                    pass
                else:
                    if ff > 0:
                        xi_e1 = 0.95*xi_e1
                    else:
                        xi_e2 = 1.05*xi_e2

            registro.info('Corrección iniciada.')

            rel_error_eta_TT = 1.0
            xi_ec = None

            while fabs(rel_error_eta_TT) > corrector_tol:
                if xi_ec is None:
                    xi_ec = float(xi_e2 - f2*(xi_e2-xi_e1)/(f2-f1))
                    sifc = step_inner_funct(True, False, xi_ec, rho_seed_c)
                    fc, rho_seed_c = sifc[0] - eta_TT_obj, sifc[1]
                else:
                    sif1 = step_inner_funct(True, False, xi_e1, rho_seed_1)
                    f1, rho_seed_1 = sif1[0] - eta_TT_obj, sif1[1]
                    sif2 = step_inner_funct(True, False, xi_e2, rho_seed_2)
                    f2, rho_seed_2 = sif2[0] - eta_TT_obj, sif2[1]
                    xi_ec = xi_e2 - (f2*(xi_e2-xi_e1)/(f2-f1))
                    sifc = step_inner_funct(True, False, xi_ec, rho_seed_c)
                    fc, rho_seed_c = sifc[0] - eta_TT_obj, sifc[1]
                rel_error_eta_TT = fc/eta_TT_obj
                registro.info('Corrección en proceso  ...  eta_TT: %.5f  ...  Error: %.5f',
                              sifc[0], rel_error_eta_TT)
                if fc*f2 < 0:
                    xi_e1, rho_seed_1 = xi_ec, rho_seed_c
                elif fc*f1 < 0:
                    xi_e2, rho_seed_2 = xi_ec, rho_seed_c

            _, _, ll_1 = step_inner_funct(True, True, xi_ec, rho_seed_c)
            registro.info('Corrección finalizada.')

            return ll_1

        def wrapper_r():
            """ Función que evalúa el modelo de pérdidas que se ha definido y, si es el de Ainley and Mathieson, aplica
            una corrección basándose en el número de Reynolds.
                            :return: Se devuelve la lista de variables que se procesan, según el modo que se defina. """

            if loss_model == 'ainley_and_mathieson':
                Re, eta_TT, xi_est, rho_seed, ll_1 = step_inner_funct()
                ll_1 = corrector(eta_TT, Re, xi_est, rho_seed)
            else:
                ll_1 = step_inner_funct()

            return ll_1
        return wrapper_r
    return Reynolds_corrector


class solver_object:
    """ Clase que define un objeto que agrupa y coordina al conjunto de métodos/atributos que conforman
    el procedimiento resolutivo propuesto para determinar unas condiciones de funcionamiento de una turbina axial."""

    def __init__(self, config: config_parameters, productos: gas_model_to_solver):
        """ :param config: Objeto que agrupa lo relativo a la configuración establecida para la ejecución del solver."""

        self.vmmr = []  # Almacena ciertas variables, para facilitar la comunicación de sus valores
        self.cfg = config  # Objeto que contiene los parámetros de interés para la ejecución del solver.
        self.rho_seed = None   # Para aligerar los cálculos para variaciones pequeñas de las variables de entrada
        self.prd = productos  # Modela el comportamiento termodinámico de los productos de la combustión
        self.AM_object = None
        self.inputs_props = None  # Se emplea como valor de referencia en las primeras semillas.
        self.Re_corrector_counter = 0  # Contador del número de llamadas efectuadas durante la corrección por Re.
        self.step_counter = 0  # Número empleado para iterar por cada escalonamiento comenzando por 0.
        self.corrector_seed = None
        # corrector_seed: Lista de listas que contienen datos concluidos por el corrector en cada escalonamiento
        #                 para aprovecharlos en cálculos consecutivos.

        if config.loss_model == 'ainley_and_mathieson':
            self.AM_object = AM_loss_model(config)

    def problem_solver(self, T_in: float, p_in: float, n: float,
                       C_inx=None, m_dot=None) -> None | tuple[float, float, float, float]:
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

        # En esta lista se almacenan variables que se requieren según el modo de funconamiento.
        ps_list: list[float] = []

        self.inputs_props = (T_in, p_in)

        @solver_timer
        def inner_solver() -> None:

            nonlocal m_dot, C_inx, ps_list
            tol = self.cfg.TOL
            registro.debug('El error relativo establecido en el solver es: %s', tol)

            rho_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='d')

            if m_dot is None:
                m_dot = rho_in * self.cfg.geom['areas'][0] * C_inx
            elif C_inx is None:
                C_inx = m_dot / (rho_in * self.cfg.geom['areas'][0])
            else:
                registro.critical('Se debe establecer uno de los parámetros opcionales "Cinx" ó "m_dot".')
                sys.exit()

            s_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='s')
            h_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='h')
            h_0in = h_in + ((10 ** (-3)) * (C_inx ** 2) / 2)

            ps_list = [T_in, p_in, rho_in, s_in, h_in, h_0in, C_inx]
            for i in range(self.cfg.n_steps):
                list_i = ps_list[i-1] if i > 0 and not self.cfg.fast_mode else ps_list
                args = list_i[0], list_i[1], list_i[6], list_i[3], list_i[4], list_i[5], m_dot, n, list_i[2]
                self.step_counter = 0
                self.Re_corrector_counter = 0
                if self.cfg.fast_mode:
                    ps_list = self.step_block(*args)
                else:
                    if i == 0:
                        ps_list = [self.step_block(*args)]
                    else:
                        ps_list += [self.step_block(*args)]

                self.step_counter += 1

            if not self.cfg.fast_mode:  # Los subindices A y B indican, resp., los pts. inicio y fin de la turbina.
                p_B, s_B, h_B, h_0B = [ps_list[-1][1]] + ps_list[-1][3:6]
                h_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='h')
                h_0A = h_in + ((10 ** (-3)) * (C_inx ** 2) / 2)
                DELTA_h = h_B - h_in
                w_total = h_0A - h_0B
                P_total = w_total * m_dot
                s_A = self.prd.get_prop({'T': T_in, 'p': p_in}, 's')
                p_0B, T_0B = self.Zero_pt_calculator(p_B, s_B, h_0B)
                T_0Bss = self.prd.get_prop(known_props={'s': s_A, 'p': p_0B}, req_prop={'T': T_0B * 0.9})
                h_0Bss = self.prd.get_prop(known_props={'T': T_0Bss, 'p': p_0B}, req_prop='h')
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

    def step_block(self, T_1: float, p_1: float, C_1: float, s_1: float, h_1: float, h_01: float,
                   m_dot: float, n: float, rho_1: float) -> list:
        """ Método que se llama de manera recurrente por cada escalonamiento definido, para determinar las propiedades
        del fluido en los puntos de interés del mismo.

        Punto 1: Entrada del escalonamiento

        Punto 2: Punto intermedio del escalonamiento

        Punto 3: Salida del escalonamiento

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

        if self.corrector_seed is None:
            self.corrector_seed = []
            corrector_memory = None
        else:
            if len(self.corrector_seed) == self.cfg.n_steps:
                corrector_memory = self.corrector_seed[self.step_counter]
            else:
                corrector_memory = None

        @Reynolds_correction(self.cfg.ETA_TOL, self.cfg.loss_model, corrector_memory)
        def inner_funct(iter_mode=False, iter_end=False, xi_est=0.0, rho_seed=None):
            """ Esta función interna se crea para poder comunicar al decorador instancias de la clase.
                    :param rho_seed: Densidad que se emplea como valor semilla en bucle_while_x2 (kg/m^3).
                    :param xi_est: Valor opcional que permite al decorador aplicar recursividad para efectuar la
                                  corrección por dependencia de Reynolds.
                    :param iter_mode: Permite diferenciar si se está aplicando recursividad para la corrección por
                                 dependencia de Reynolds y así conocer las instrucciones más convenientes.
                    :param iter_end: Permite omitir cálculos innecesarios durante la corrección por dependencia de
                                    Reynolds.
                            :return: Se devuelven diferentes variables que se requieren según la situación."""

            count = self.step_counter

            if rho_seed is None:
                if self.rho_seed is not None and len(self.rho_seed) == self.cfg.n_steps:
                    rho_seed = self.rho_seed[count]
                else:
                    rho_seed = [rho_1*0.8, rho_1*0.9]
                    if count == 0:
                        self.rho_seed = []

            A_tpl = (self.cfg.geom['areas'][count * 2],
                     self.cfg.geom['areas'][count * 2 + 1],
                     self.cfg.geom['areas'][count * 2 + 2])

            eta_TT = Re_in = Re_out = Re = None

            if not iter_mode:
                s, H = self.cfg.geom['s'][count*2], self.cfg.geom['H'][count*2]
                Re_in = Reynolds(rho_1, C_1, T_1, s, H, self.cfg.geom['alfap_o_est'][count], self.prd)
            else:
                self.Re_corrector_counter += 1

            registro.info('Iter mode: %s  ...  Llamadas: %s', iter_mode, self.Re_corrector_counter)

            if count > 0:
                C_1x = m_dot / (rho_1 * A_tpl[0])
                alfa_1 = acos(C_1x / C_1)
            else:
                C_1x = C_1
                alfa_1 = 0.0
            registro.debug('La velocidad axial establecida a la entrada del escalonamiento %s es: %.2f m/s',
                           count + 1, C_1x)

            registro.info('Se va a calcular la salida del estátor del escalonamiento número %d',
                          count + 1)

            h_02 = h_01

            if not iter_mode:
                args = ['est', A_tpl[1], alfa_1, h_02, m_dot, s_1, rho_seed[0]]
                outputs = self.blade_outlet_calculator(*args, step_iter_mode=iter_mode, Re_in=Re_in)
                p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2, xi_est, Re_in = outputs
            else:
                args = ['est', A_tpl[1], alfa_1, h_02, m_dot, s_1, rho_seed[0]]
                outputs = self.blade_outlet_calculator(*args, step_iter_mode=iter_mode, xi=xi_est)
                p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2 = outputs

            s_2 = self.prd.get_prop(known_props={'p': p_2, 'T': T_2}, req_prop='s')

            # En las líneas que siguen se determinan los triángulos de velocidades
            U = self.cfg.geom['Rm'][(count * 2) + 1] * n * 2 * pi / 60
            registro.debug('La velocidad tangencial de giro en el radio medio es: %.2f m/s', U)
            C_2u = C_2 * sin(alfa_2)
            registro.debug('La velocidad tangencial del flujo a la entrada del rótor es: %.2f m/s', C_2u)
            omega_2u = C_2u - U
            omega_2x = C_2x
            beta_2 = atan(omega_2u / C_2x)
            omega_2 = C_2x / cos(beta_2)

            registro.info(' Se va a calcular la salida del rótor del escalonamiento número %d.     ', count+1)
            h_r3 = h_r2 = h_2 + ((10 ** (-3)) * (omega_2 ** 2) / 2)
            args = ['rot', A_tpl[2], beta_2, h_r3, m_dot, s_2, rho_seed[1]]
            outputs = self.blade_outlet_calculator(*args, step_iter_mode=iter_mode)
            if not iter_mode:
                p_3, h_3, T_3, omega_3, rho_3, h_3s, T_3s, C_3x, M_3r, beta_3, xi_rot, Re_out = outputs
            else:
                p_3, h_3, T_3, omega_3, rho_3, h_3s, T_3s, C_3x, M_3r, beta_3, xi_rot = outputs

            omega_3u = omega_3 * sin(beta_3)
            omega_3x = omega_3 * cos(beta_3)
            C_3u = omega_3u - U
            C_3 = sqrt(C_3u ** 2 + C_3x ** 2)
            alfa_3 = asin(C_3u / C_3)

            s_3 = self.prd.get_prop(known_props={'T': T_3, 'p': p_3}, req_prop='s')
            h_03 = h_3 + (10 ** (-3)) * ((C_3 ** 2) / 2)
            local_list_1 = [T_3, p_3, rho_3, s_3, h_3, h_03, C_3, alfa_3]

            if h_02 - h_03 < 0:
                registro.error('No se extrae energía del fluido.')

            # Media aritmética de Re a la entrada y a la salida recomendada por AM para la corrección.
            if not iter_mode:
                Re = (Re_in + Re_out)/2
                registro.debug('Reynolds a la entrada: %d, Reynolds a la salida: %d, Reynolds promedio del '
                               'escalonamiento: %d', Re_in, Re_out, Re)
                if Re < 50_000:
                    registro.warning('El número de Reynolds es demasiado bajo.')

            if len(self.rho_seed) < self.cfg.n_steps:
                if not iter_mode:
                    self.rho_seed.append([rho_2, rho_3].copy())
                elif iter_end:
                    self.rho_seed[-1] = [rho_2, rho_3].copy()

            # Se determina en estas líneas el rendimiento total a total para que sea posible aplicar la corrección:
            if self.cfg.loss_model == 'ainley_and_mathieson' and (self.cfg.fast_mode or iter_mode):
                w_esc = h_02 - h_03
                p_03, T_03 = self.Zero_pt_calculator(p_x=p_3, s_x=s_3, h_0x=h_03)
                T_03ss = self.prd.get_prop(known_props={'s': s_1, 'p': p_03}, req_prop={'T': T_03})
                h_03ss = self.prd.get_prop(known_props={'T': T_03ss, 'p': p_03}, req_prop='h')
                Y_esc = h_03 - h_03ss
                eta_TT = w_esc / (w_esc + Y_esc)

            if iter_end:
                self.corrector_seed.append([xi_est, [rho_2, rho_3]])

            if not self.cfg.fast_mode and (iter_end or not iter_mode):
                Y_est = xi_est * ((10 ** (-3)) * (C_2 ** 2) / 2)
                Y_rot = xi_rot * ((10 ** (-3)) * (omega_3 ** 2) / 2)
                w_esc = h_02 - h_03
                p_03, T_03 = self.Zero_pt_calculator(p_x=p_3, s_x=s_3, h_0x=h_03)
                T_03ss = self.prd.get_prop(known_props={'s': s_1, 'p': p_03}, req_prop={'T': T_03})
                h_03ss = self.prd.get_prop(known_props={'T': T_03ss, 'p': p_03}, req_prop='h')
                Y_esc = h_03 - h_03ss
                eta_TT = w_esc / (w_esc + Y_esc)
                C_1u = sqrt(C_1 ** 2 - C_1x ** 2)
                a_1 = self.prd.get_sound_speed(T=T_1, p=p_1)
                M_1 = C_1 / a_1
                a_3 = self.prd.get_sound_speed(T=T_3, p=p_3)
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
                T_02s = self.prd.get_prop(known_props={'s': s_1, 'p': p_02}, req_prop={'T': T_02 * 0.9})
                h_02s = self.prd.get_prop(known_props={'T': T_02s, 'p': p_02}, req_prop='h')
                T_3ss = self.prd.get_prop(known_props={'p': p_3, 's': s_1}, req_prop={'T': T_3 * 0.95})
                h_3ss = self.prd.get_prop(known_props={'T': T_3ss, 'p': p_3}, req_prop='h')
                T_03s = self.prd.get_prop(known_props={'p': p_03, 's': s_2}, req_prop={'T': T_1 * 0.9})
                h_03s = self.prd.get_prop(known_props={'T': T_03s, 'p': p_03}, req_prop='h')
                Y_esc = h_03 - h_03ss
                w_ss_esc = w_esc + Y_esc
                w_s_esc = w_esc + h_03 - h_03s
                eta_TE = w_esc / (w_esc + Y_esc + ((10 ** (-3)) * (C_3 ** 2) / 2))
                # Aunque resulta más relevante la relación isentrópica, es posible calcular el rendimiento politrópico
                # Se emplean para ello los valores de remanso
                p_01, T_01 = self.Zero_pt_calculator(p_x=p_1, s_x=s_1, h_0x=h_01)
                eta_p_esc = log(1 - (eta_TT * (1 - (T_03ss / T_01))), 10) / log(T_03ss / T_01, 10)
                r_esc = p_01 / p_03
                local_list_1 += [C_3x, C_3u, T_03, p_03, h_r3, h_3s, T_3s, omega_3, omega_3x, omega_3u, beta_3, M_3r,
                                 M_3, PHI_3, T_2, p_2, rho_2, s_2, h_2, h_02, C_2, C_2x, C_2u, alfa_2, T_02, p_02, h_r2,
                                 h_2s, T_2s, omega_2, omega_2x, omega_2u, beta_2, M_2, PHI_2, T_1, p_1, rho_1, s_1, h_1,
                                 h_01, C_1, C_1x, C_1u, alfa_1, T_01, p_01, M_1, Y_est, xi_est, Y_rot, xi_rot, w_esc,
                                 Pot_esc, eta_p_esc, r_esc, GR, PSI, speed_loss_coeff_E, speed_loss_coeff_R,
                                 DELTA_h_esc, eta_TT, eta_TE, Y_esc, w_s_esc, w_ss_esc, C_2s, T_02s, h_02s, omega_3s,
                                 T_3ss, T_03s, T_03ss, h_3ss, h_03s, h_03ss, U]

                # eta_TT se debe enviar siempre que iter_mode para evaluar la condición del decorador
                if not iter_mode and self.cfg.loss_model == 'ainley_and_mathieson':
                    return Re, eta_TT, xi_est, [rho_2, rho_3].copy(), local_list_1.copy()
                elif self.cfg.loss_model == 'ainley_and_mathieson':
                    return eta_TT, [rho_2, rho_3].copy(), local_list_1.copy()
                else:
                    return local_list_1.copy()

            else:
                if not iter_mode and self.cfg.loss_model == 'ainley_and_mathieson':
                    return Re, eta_TT, xi_est, [rho_2, rho_3].copy(), local_list_1
                elif self.cfg.loss_model == 'ainley_and_mathieson':
                    return eta_TT, [rho_2, rho_3].copy(), local_list_1
                else:
                    return local_list_1

        ll_1 = inner_funct()
        return ll_1

    def blade_outlet_calculator(self, blade: str, area_b: float, tau_a: float, h_tb: float, m_dot: float,
                                s_a: float, rho_outer_seed: float, step_iter_mode=False, xi=0.0, Re_in=0.0):
        """ Se hace un cálculo iterativo para conocer las propiedades a la salida del estátor y del rótor (según el
        caso, estátor / rótor, se proporciona el valor de la entalpía total / rotalpía y del ángulo que forma la
        velocidad absoluta / relativa del fluido con la dirección del eje de la turbina axial, respectivamente).

        Punto a: Sección al principio del álabe

        Punto b: Sección al final del álabe

            :param Re_in: Valor del número de Reynolds a la entrada del estátor.
            :param xi: Coeficiente adimensional de pérdidas de la corona en cuestión.
            :param step_iter_mode: Permite diferenciar si se está aplicando recursividad para la corrección por
                               dependencia de Reynolds y así conocer las instrucciones más convenientes.
            :param tau_a: Ángulo del movimiento absoluto/relativo de entrada del fluido con la dirección axial
                         (rads).
            :param blade: Diferenciador del tipo de álabe, puede ser 'est' o 'rot'.
            :param area_b: El área de la sección de paso al final de la corona (m^2).
            :param h_tb: Entalpía total / rotalpía a la salida del álabe (kJ/kg).
            :param m_dot: Flujo másico (kg/s).
            :param s_a: Entropía en a (kJ/kgK).
            :param rho_outer_seed: Densidad que se emplea como semilla, debe aproximar la densidad en b (kg/m^3).
                    :return: Se devuelven las variables que contienen las propiedades a la salida que se han
                    determinado (p_b, h_b, T_b, U_b, rho_b, h_bs, T_bs, C_bx). """

        if self.cfg.loss_model == 'ainley_and_mathieson':
            self.AM_object.limit_mssg = [True, True, True]

        rho_b = rho_bp = rho_outer_seed
        M_b = h_b = U_b = h_bs = C_bx = tau_b = xi_preiter = Y_total = Re = None
        rel_diff, tol, geom = 1.0, self.cfg.TOL, self.cfg.geom

        T_b, T_bs, p_b = self.inputs_props[0]*0.95, self.inputs_props[0]*0.9, self.inputs_props[1]*0.95

        counter = self.step_counter
        num = counter*2 + (0 if blade == 'est' else 1)

        if self.cfg.loss_model == 'soderberg_correlation':
            s, H, b = geom['s'][num], geom['H'][num], geom['b'][num]
            args = [blade, geom['alfap_i_est'][counter], geom['alfap_i_rot'][counter], H, b, geom['A_rel'][num]]
            xi_preiter = Soderberg_correlation(*args)
            tau_b = geom['alfap_o_est'][counter] if blade == 'est' else geom['alfap_o_rot'][counter]

        elif self.cfg.loss_model == 'ainley_and_mathieson':
            # tau_2 debe ser corregida si Mb < 0.5, ahora se asigna el valor calculado inicialmente en cualquier caso.
            # El valor de xi se conoce únicamente cuando iter_mode y estátor, pero no el de tau_2 (Y_total no default).
            # El resto de veces depende xi depende de si se modifica o no el valor de tau_2.
            tau_b = self.AM_object.outlet_angle_before_mod[num]

        # p: iteración previa .... b: estado que se quiere conocer, a la salida del álabe
        while fabs(rel_diff) > tol:

            C_bx = m_dot / (area_b * rho_b)  # C_bx: velocidad axial a la salida
            tau_b_n = None

            while tau_b_n is None or fabs(tau_b_n - tau_b) > tol:

                if tau_b_n is None:
                    tau_b_n = tau_b
                else:
                    tau_b = tau_b_n

                U_b = C_bx / cos(tau_b)  # U_b: vel. absoluta o relativa ... tau: alfa o beta ... según el caso
                h_b = h_tb-(0.001*(U_b*U_b)/2)
                # Se aplica conservación de la entalpía total/rotalpía ... según el caso
                # (no se puede determinar otra variable con h2s y s1 con PyroMat)

                p_b = self.prd.get_prop(known_props={'d': rho_b, 'h': h_b}, req_prop={'p': p_b})
                T_b = self.prd.get_prop(known_props={'h': h_b, 'd': rho_b}, req_prop={'T': T_b})
                a_b = self.prd.get_sound_speed(T_b, p_b)
                gamma_b = self.prd.get_gamma(T_b, p_b)
                T_bs = self.prd.get_prop(known_props={'p': p_b, 's': s_a}, req_prop={'T': T_bs})
                h_bs = self.prd.get_prop(known_props={'T': T_bs, 'p': p_b}, req_prop='h')

                M_b = U_b/a_b   # Mach del flujo absoluto o relativo según si estátor o rótor (respectivamente)

                if not step_iter_mode:
                    if blade == 'est':
                        Re = Re_in
                    else:
                        s, H = geom['s'][num], geom['H'][num]
                        Re = Reynolds(rho_b, U_b, T_b, s, H, geom['alfap_o_est'][counter], self.prd)

                if self.cfg.loss_model == 'soderberg_correlation':
                    xi = soder_Re_mod(xi_preiter, Re)  # Una variable aparte para no repetir la corrección.

                elif self.cfg.loss_model == 'ainley_and_mathieson':
                    tau_b_n = self.AM_object.tau2_corrector(num, M_b)
                    # Se ejecuta el primer bloque a excepción de si es estátor y step_iter_mode.
                    if not step_iter_mode or blade == 'rot':
                        args = [num, degrees(tau_a), degrees(tau_b), step_iter_mode]
                        Y_total = self.AM_object.Ainley_and_Mathieson_Loss_Model(*args)
                        xi = Y_total/(1 + (0.5*gamma_b*(M_b*M_b)))
                    else:
                        Y_total = xi*(1 + (0.5*gamma_b*(M_b*M_b)))
                        args = [num, degrees(tau_a), degrees(tau_b), True, Y_total]
                        self.AM_object.Ainley_and_Mathieson_Loss_Model(*args)

            h_b = (0.001 * xi * (U_b * U_b) / 2) + h_bs
            rho_b = self.prd.get_prop({'p': p_b, 'h': h_b}, {'d': rho_b})

            rel_diff = (rho_bp - rho_b) / rho_b
            rho_bp = rho_b

            registro.debug('Densidad (kg/m^3): %.12f  ...  Error relativo: %.12f', rho_b, rel_diff)

        alfap_1 = degrees(geom['alfap_i_est'][num//2] if num % 2 == 0 else geom['alfap_i_rot'][num//2])
        alfap_2 = degrees(geom['alfap_o_est'][num//2] if num % 2 == 0 else geom['alfap_o_rot'][num//2])

        if M_b > 0.5:
            registro.warning('Mach %sa la salida superior a 0.5 ... Valor: %.2f',
                             '' if num % 2 == 0 else 'relativo ', M_b)
        else:
            registro.debug('Valor del número de Mach %sa la salida: %.2f',
                           '' if num % 2 == 0 else 'relativo ', M_b)

        if step_iter_mode:
            registro.debug('La relación entre el valor actual y el inicial de las pérdidas adimensionales de presión '
                           'en ambas coronas del escalonamiento %s es: %.3f\n  ...   ',
                           1 + (num//2), Y_total / self.AM_object.Y_t_preiter[num])

        registro.debug('Incidencia: %.2f°  ...  tau_in: %.2f°  ...  Ángulo del B.A.: %.2f°',
                       degrees(tau_a) - alfap_1, degrees(tau_a), alfap_1)
        registro.debug('Desviación: %.2f°  ...  tau_out: %.2f°  ...  Ángulo del B.S.: %.2f°',
                       degrees(tau_b) - alfap_2, degrees(tau_b), alfap_2)
        registro.debug('Pérdidas:  ...  Y_total: %.4f  ...  Yp: %.4f   ',
                       Y_total, self.AM_object.Yp_preiter[num] if not step_iter_mode else self.AM_object.Yp_iter_mode)

        return_vars = [p_b, h_b, T_b, U_b, rho_b, h_bs, T_bs, C_bx, M_b, tau_b]

        if blade == 'est':
            if not step_iter_mode:
                return *return_vars, xi, Re
            else:
                return return_vars
        else:
            if not step_iter_mode:
                return *return_vars, xi, Re
            else:
                return *return_vars, xi

    def Zero_pt_calculator(self, p_x: float, s_x: float, h_0x: float):
        """ Este método es para determinar presiones y temperaturas de remanso.

                :param p_x: Presión en la sección x (Pa).
                :param s_x: Entropía en la sección x (kJ/kgK).
                :param h_0x: Entalpía total en la sección x (kJ/kg).
                        :return: Devuelve la presión total (Pa) y la temperatura total (K) en una sección x. """

        p_0x, T_0x, end, tol = p_x * 1.1, self.inputs_props[0]*1.1, False, self.cfg.TOL

        while not end:
            T_0x = self.prd.get_prop(known_props={'h': h_0x, 'p': p_0x}, req_prop={'T': T_0x})
            if self.cfg.ideal_gas:
                p_0x = self.prd.get_prop(known_props={'T': T_0x, 's': s_x}, req_prop={'p': p_0x})
                end = True
            else:
                p_0x_iter = self.prd.get_prop(known_props={'T': T_0x, 's': s_x}, req_prop={'p': p_0x})
                if fabs(p_0x - p_0x_iter) / p_0x < tol:
                    end = True
                p_0x = p_0x_iter

        return p_0x, T_0x


def main():
    fast_mode = False
    settings = config_parameters(TOL=1E-6, ETA_TOL=1E-3, n_steps=1, ideal_gas=True, fast_mode=True,
                                 loss_model='ainley_and_mathieson')

    # Geometría procedente de: https://apps.dtic.mil/sti/pdfs/ADA950664.pdf
    Rm = 0.1429
    heights = [0.0445 for _ in range(3)]
    areas = [0.0399 for _ in range(3)]
    chord = [0.0338, 0.0241]
    t_max = [0.2*chord[0], 0.15*chord[1]]
    pitch = [0.0249, 0.0196]
    t_e = [0.01*s for s in pitch]
    blade_opening = [0.01090, 0.01354]
    e_param = [0.0893, 0.01135]
    tip_clearance = [0.0, 0.0008]

    settings.set_geometry(B_A_est=0, theta_est=70, B_A_rot=36, theta_rot=100, areas=areas,
                          cuerda=chord, radio_medio=Rm, e=e_param, o=blade_opening, s=pitch,
                          H=heights, t_max=t_max, r_r=0.002, r_c=0.001, t_e=t_e,
                          k=tip_clearance, holgura_radial=False)

    gas_model = gas_model_to_solver(thermo_mode="ig", relative_error=1E-6)
    solver = solver_object(settings, gas_model)

    if fast_mode:
        output = solver.problem_solver(T_in=1100, p_in=275_790, n=14_427.32, C_inx=130)
        T_salida, p_salida, C_salida, alfa_salida = output
        print(' T_out =', T_salida, '\n', 'P_out =', p_salida,
              '\n', 'C_out =', C_salida, '\n', 'alfa_out =', alfa_salida)

    else:
        solver.problem_solver(T_in=1100, p_in=275_790, n=14_427.32, C_inx=130)


if __name__ == '__main__':
    main()
