"""
En este módulo se crea una clase que, con sus métodos y con herramientas de otros módulos, permite determinar
las condiciones de funcionamiento de la turbina axial que se defina.
"""

from time import time
from math import degrees

from loss_model import *


def solver_decorator(cfg: config_parameters, p_out: np.float64 | None, C_inx: np.float64 | None):
    """ Decorador externo del método problem solver con la finalidad de definir los argumentos necesarios para permitir
    determinar las condiciones de funcionamiento dada la presión a la salida.
            :param cfg: Objeto que contiene la configuración establecida.
            :param p_out: Presión a la salida de la turbina (Pa).
            :param C_inx: Estimación de la velocidad a la entrada que se debe recibir cuando se fija la presión a
                          la salida."""

    def solver_inner_decorator(solver_method):
        """ Decorador interno que gestiona la función interna del método problem_solver de la clase solver.
                            :param solver_method: Función que se decora. """

        def iterate_ps():
            nonlocal C_inx
            ps_list = None

            def read_ps_list():
                if cfg.fast_mode:
                    new_p_out = copy.deepcopy(ps_list)[1]
                else:
                    new_p_out = copy.deepcopy(ps_list)[-2][1]
                return new_p_out

            iter_count = 0
            bolz = check = False
            # Puntos a, b tal que C_inx_b > C_inx_a
            pre_C_inx_a = pre_C_inx_b = C_inx
            delta = cfg.relative_jump
            C_inx_a = C_inx*(1 - (2*delta))
            C_inx_b = C_inx*(1 + (2*delta))
            p_out_iter_b = p_out_iter_a = None
            f_a = f_b = None

            registro.info('Se va a buscar un intervalo que contenga la solución.')
            while not bolz:
                try:
                    if not check:  # Primera vuelta
                        ps_list = solver_method(C_inx_a)
                        p_out_iter_a = read_ps_list()
                        ps_list = solver_method(C_inx_b)
                        p_out_iter_b = read_ps_list()
                        check = True
                    else:
                        ps_list = solver_method(C_inx)
                        p_out_iter = read_ps_list()
                        # Se omite el rango que no contiene la solución.
                        if C_inx_b > pre_C_inx_b*(1+cfg.STEP_DEC_TOL):  # Se proviene del nivel de avance de b
                            p_out_iter_a = pre_p_out_iter_b = p_out_iter_b
                            p_out_iter_b = p_out_iter
                            C_inx_a = pre_C_inx_b
                            f_a = f_b
                            f_b = (p_out_iter_b - pre_p_out_iter_b)/(C_inx_b - pre_C_inx_b)
                            if f_a is None:
                                f_a = f_b
                        else:  # Se proviene del nivel de retroceso de a
                            p_out_iter_b = pre_p_out_iter_a = p_out_iter_a
                            p_out_iter_a = p_out_iter
                            C_inx_b = pre_C_inx_a
                            f_b = f_a
                            f_a = (p_out_iter_a - pre_p_out_iter_a)/(C_inx_a - pre_C_inx_a)

                        # Se evalúa si el nuevo rango contiene la solución.
                        if (p_out_iter_b-p_out)*(p_out_iter_a-p_out) <= 0:
                            bolz = True

                    # Se almacenan los valores antiguos para repetir el proceso
                    pre_C_inx_a, pre_C_inx_b = C_inx_a, C_inx_b

                    registro.info('En proceso de búsqueda de la solución ... Rango actual: [%.2f, %.2f]',
                                  float(p_out_iter_a), float(p_out_iter_b))

                # Se capturan posibles excepciones (ver tendencia p_out vs m_dot)
                except NonConvergenceError:
                    registro.warning('Se ha capturado una excepción.')
                    delta /= 1.9
                    C_inx_a, C_inx_b = pre_C_inx_a, pre_C_inx_b

                except GasLibraryAdaptedException:
                    registro.warning('Se ha capturado una excepción.')
                    delta /= 1.75
                    C_inx_a, C_inx_b = pre_C_inx_a, pre_C_inx_b

                # Comprobación de que localmente creciente/decreciente en ambos extremos:
                # (f_a > 0 & f_b > 0) or (f_a < 0 & f_b < 0)
                if f_a is not None and f_a*f_b > 0 and not bolz:

                    signo_a, signo_b = f_a/np.fabs(f_a), f_b/np.fabs(f_b)

                    if p_out_iter_b*(1+(cfg.STEP_DEC_TOL*signo_b))*signo_b < p_out*signo_b:
                        # Nivel de avance en 'b'
                        C_inx_b = C_inx_b*(1 + delta)
                        C_inx = C_inx_b

                    elif p_out_iter_a*(1+(cfg.STEP_DEC_TOL*signo_a))*signo_a > p_out*signo_a:
                        # Nivel de retroceso en 'a'
                        C_inx_a = C_inx_a * (1 - delta)
                        C_inx = C_inx_a

                    else:
                        registro.info('Se ha localizado la solución.')

                elif f_a is None and f_b is None and not bolz:  # Primera vuelta
                    C_inx_b = C_inx_b*(1 + delta)
                    C_inx_a = C_inx_a*(1 - delta)
                    C_inx = C_inx_b

                elif bolz:
                    pass

                else:
                    registro.critical('Se debe elegir un valor de velocidad a la entrada diferente o modificar '
                                      'el salto relativo.')
                    sys.exit()

            rel_error = 1

            while np.fabs(rel_error) > cfg.SOLVER_DEC_TOL:  # Se emplea régula falsi
                iter_count += 1
                f_a = p_out_iter_a-p_out
                f_b = p_out_iter_b-p_out
                diff_value = (f_b * (C_inx_b - C_inx_a) / (f_b - f_a))
                C_inx = C_inx_b - diff_value
                try:
                    ps_list = solver_method(C_inx)
                    p_out_iter = read_ps_list()
                    f_c = p_out_iter - p_out
                    rel_error = f_c / p_out
                    if f_c * f_b <= 0:
                        C_inx_a = C_inx
                        p_out_iter_a = p_out_iter
                    elif f_c * f_a <= 0:
                        C_inx_b = C_inx
                        p_out_iter_b = p_out_iter
                except NonConvergenceError:
                    C_inx_b *= (1 + cfg.SOLVER_DEC_TOL)

                registro.info('Error de presión a la salida: %.10f  ...  Rango actual (Pa): [%.7f, %.7f]',
                              float(rel_error), float(p_out_iter_a), float(p_out_iter_b))

                if iter_count > cfg.iter_limit:
                    registro.critical('No converge.')

            return

        def wrapper_s():
            """ Se evalúa cuánto tiempo se tarda en determinar la condición de funcionamiento.
                                    :returns: Se devuelve o se almacena el output de la función decorada, según el
                                                modo de funcionamiento que aplique. """

            t_1 = time()

            if p_out is None:
                _ = solver_method()
            else:
                iterate_ps()
            t_2 = (time() - t_1).__round__(0)

            m, s = divmod(t_2, 60)
            registro.info('Tiempo de cálculo: %s minutos y %s segundos.', int(m.__round__(0)), int(s.__round__(0)))

            return

        return wrapper_s

    return solver_inner_decorator


def step_decorator(cfg: config_parameters, step_corrector_memory):
    """ Decorador del decorador real, permite referenciar los parámetros necesarios para manipular la salida de la
    función decorada.
                :param cfg: Objeto que contiene datos sobre la configuración que se ha establecido.
                :param step_corrector_memory: Lista con las variables xi_ec y rho_seed_c de la ejecución previa.
                            :return: Se devuelve el decorador real. """

    def Reynolds_corrector(step_inner_function):
        """ Decorador real, gestiona la función interna del método gen_steps de la clase solver y posibilita
         aplicar la corrección por el número de Reynolds.
                    :param step_inner_function: Función que se decora.
                                :return: Se devuelve el wrapper_r. """

        corrector_tol = cfg.STEP_DEC_TOL
        sif_output = None

        def get_sif_output(iter_mode: bool = None, iter_end: bool = None, xi=None, rho_seed: list = None, Re=None):
            """ Se hace deepcopy de la salida y se devuelve. """
            nonlocal sif_output
            sif_output = step_inner_function(iter_mode, iter_end, xi, rho_seed, Re)
            return copy.deepcopy(sif_output)

        def AM_corrector(eta_TT, Re: int, xi_est, rho_seed) -> list:
            """ Función que aplica la corrección de Reynolds cuando es llamada por wrapper_r. La solución se determina
                aplicando el teorema de Bolzano y Régula Falsi. Se va a usar como semilla de la siguiente iteración los
                valores que resultan de la anterior.
                    :param rho_seed: Lista con semillas para los dos valores de densidad a la salida de cada corona.
                    :param eta_TT: Rendimiento total a total del escalonamiento.
                    :param Re: Número de Reynolds.
                    :param xi_est: Coeficiente adimensional de pérdidas en el estátor.
                            :return: Se devuelve la lista de variables que corresponda según el modo de
                                    funcionamiento que se defina."""

            eta_TT_obj = 1 - (((1-eta_TT) / (200_000**(-1/5))) * (Re**(-1/5)))

            f1 = f2 = None
            bolz_c, ff = 1.0, 1.0

            if step_corrector_memory is not None:
                xi_e1 = step_corrector_memory[0]*(1 - corrector_tol)
                xi_e2 = step_corrector_memory[0]*(1 + corrector_tol)
                rho_seed_1 = rho_seed_2 = rho_seed_c = step_corrector_memory[1]
            else:
                xi_e0 = xi_est * (Re**(-1/5)) / (200_000**(-1/5))
                xi_e1 = xi_e0 * 0.999
                xi_e2 = xi_e0 * 1.001
                rho_seed_1 = rho_seed_2 = rho_seed_c = rho_seed

            while bolz_c > 0:
                registro.info('Buscando el rango que garantice encontrar la solución.')

                if (f1 is None and f2 is None) or step_corrector_memory is not None:
                    sif1 = get_sif_output(True, False, xi_e1, rho_seed_1)
                    f1, rho_seed_1 = sif1[1] - eta_TT_obj, sif1[3]
                    sif2 = get_sif_output(True, False, xi_e2, rho_seed_2)
                    f2, rho_seed_2 = sif2[1] - eta_TT_obj, sif2[3]
                elif ff > 0:
                    sif1 = get_sif_output(True, False, xi_e1, rho_seed_1)
                    f1, rho_seed_1 = sif1[1] - eta_TT_obj, sif1[3]
                else:
                    sif2 = get_sif_output(True, False, xi_e2, rho_seed_2)
                    f2, rho_seed_2 = sif2[1] - eta_TT_obj, sif2[3]

                ff = (f2 - f1)/(xi_e2 - xi_e1)
                bolz_c = f1*f2

                if bolz_c < 0:
                    pass
                else:
                    if step_corrector_memory is not None:
                        xi_e1 = (1-cfg.relative_jump) * xi_e1
                        xi_e2 = (1+cfg.relative_jump) * xi_e2
                    elif ff > 0:
                        xi_e1 = (1-cfg.relative_jump) * xi_e1
                    else:
                        xi_e2 = (1+cfg.relative_jump) * xi_e2

            registro.info('Corrección iniciada.')

            rel_error_eta_TT = 1.0
            xi_ec = None

            while np.fabs(rel_error_eta_TT) > corrector_tol:
                if xi_ec is None:
                    xi_ec = xi_e2 - (f2*(xi_e2-xi_e1)/(f2-f1))
                    sifc = get_sif_output(True, False, xi_ec, rho_seed_c)
                    fc, rho_seed_c = sifc[1] - eta_TT_obj, sifc[3]
                else:
                    sif1 = get_sif_output(True, False, xi_e1, rho_seed_1)
                    f1, rho_seed_1 = sif1[1] - eta_TT_obj, sif1[3]
                    sif2 = get_sif_output(True, False, xi_e2, rho_seed_2)
                    f2, rho_seed_2 = sif2[1] - eta_TT_obj, sif2[3]
                    xi_ec = xi_e2 - (f2*(xi_e2-xi_e1)/(f2-f1))
                    sifc = get_sif_output(True, False, xi_ec, rho_seed_c)
                    fc, rho_seed_c = sifc[1] - eta_TT_obj, sifc[3]
                rel_error_eta_TT = fc/eta_TT_obj
                registro.info('Corrección en proceso  ...  eta_TT: %.4f  ...  Error: %.4f',
                              float(sifc[1]), float(rel_error_eta_TT))
                if fc*f2 < 0:
                    xi_e1, rho_seed_1 = xi_ec, rho_seed_c
                elif fc*f1 < 0:
                    xi_e2, rho_seed_2 = xi_ec, rho_seed_c

            _, _, _, _, ll_1 = get_sif_output(True, True, xi_ec, rho_seed_c)
            registro.info('Corrección finalizada.')

            return ll_1

        def AU_corrector():
            if step_corrector_memory is None:
                Re, rho_seed, _ = get_sif_output()
            else:
                Re, rho_seed = step_corrector_memory[0], step_corrector_memory[1]
            rel_error = None
            iter_counter = 0
            while rel_error is None or rel_error > corrector_tol:
                iter_counter += 1
                if iter_counter > cfg.iter_limit:
                    registro.critical('No converge.')
                    raise NonConvergenceError
                Re_n = Re
                Re, rho_seed, _ = get_sif_output(True, False, None, rho_seed, Re_n)
                rel_error = np.fabs(Re_n-Re) / Re
            _, _, ll_1 = get_sif_output(True, True, None, rho_seed, Re)
            return ll_1

        def wrapper_r():
            """ Función que evalúa el modelo de pérdidas que se ha definido y, si es el de Ainley and Mathieson, aplica
            una corrección basándose en el número de Reynolds.
                            :return: Se devuelve la lista de variables que se procesan, según el modo que se defina. """

            if cfg.loss_model == 'Ainley_and_Mathieson':
                Re, eta_TT, xi_est, rho_seed, _ = get_sif_output()
                ll_1 = AM_corrector(eta_TT, Re, xi_est, rho_seed)
            else:
                ll_1 = AU_corrector()

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
        self.rho_seed_list = None  # Para aligerar los cálculos para variaciones pequeñas de las variables de entrada
        self.prd = productos  # Modela el comportamiento termodinámico de los productos de la combustión
        self.AM_object = None
        self.AUNGIER_object = None
        self.ref_values = None  # Valores de referencia en las primeras semillas dentro de "blade_outlet_calculator".
        self.first_seeds_boc = None
        self.step_iter_mode = False
        self.step_iter_end = False
        self.Re_corrector_counter = 0  # Contador del número de llamadas efectuadas durante la corrección por Re.
        self.step_counter = 0  # Número empleado para iterar por cada escalonamiento comenzando por 0.
        self.corrector_seed = None
        # corrector_seed: Lista de listas que contienen datos concluidos por el corrector en cada escalonamiento
        #                 para aprovecharlos en cálculos consecutivos.

        if config.loss_model == 'Ainley_and_Mathieson':
            for key in ['e', 'o', 't_max', 'r_r', 'r_c', 't_e', 'k']:
                if key not in self.cfg.geom:
                    registro.critical('Para emplear el modelo de pérdidas se debe introducir "%s"', key)
                    sys.exit()
            self.AM_object = Ainley_and_Mathieson_Loss_Model(config)
            self.AM_object.AM_diameter_def()

        if config.loss_model == 'Aungier':
            for key in ['e', 'o', 't_max', 'r_r', 'r_c', 't_e', 'k', 'roughness_ptv', 'b_z', 'delta']:
                if key not in self.cfg.geom:
                    registro.critical('Para emplear el modelo de pérdidas se debe introducir "%s".', key)
                    sys.exit()
            self.AUNGIER_object = Aungier_Loss_Model(config)

    def problem_solver(self, T_inlet, p_inlet, rpm, C_inlet=None, mass_flow=None,
                       p_outlet=None, C_inlet_ref=None) -> None | tuple:
        """Esta función inicia la resolución del problema definido por la geometría configurada y las variables
        requeridas como argumento.
                :param rpm: Velocidad de giro (rpm).
                :param mass_flow: Flujo másico que atraviesa la turbina (kg/s).
                :param T_inlet: Temperatura a la entrada de la turbina (K).
                :param p_inlet: Presión a la entrada de la turbina (Pa).
                :param p_outlet: Presión a la salida de la turbina (Pa).
                :param C_inlet: Velocidad a la entrada de la turbina (Supuesta completamente axial) (m/s).
                :param C_inlet_ref: Valor de referencia que debe aproximarse a C_inx.
                        :return: Si fast_mode se devuelven los valores a la salida de temperatura (K), presión (Pa),
                                velocidad (m/s) y ángulo del flujo con la dirección axial (degrees), en caso contrario
                                no se devuelve nada."""

        # En esta lista se almacenan variables que se requieren según el modo de funconamiento.
        ps_list = []

        tol = self.cfg.TOL
        registro.debug('El error relativo establecido en el solver es: %s', tol)
        p_out = C_inx = m_dot = T_in = p_in = n = C_inx_ref = None

        T_in, p_in, n = np.float64(T_inlet), np.float64(p_inlet), np.float64(rpm)
        if mass_flow is not None:
            m_dot = np.float64(mass_flow)
        elif C_inlet is not None:
            C_inx = np.float64(C_inlet)
        elif p_outlet is not None:
            p_out = np.float64(p_outlet)
            if C_inlet_ref is not None:
                C_inx_ref = np.float64(C_inlet_ref)
            else:
                registro.critical('Debe indicarse un valor de referencia de velocidad a la entrada si se desea fijar '
                                  'la presión a la salida de la turbina.')
                sys.exit()  # Quizas convenga cambiar esto por raise y algún error para datos insuficientes
        else:
            registro.critical('Se debe establecer uno de los parámetros opcionales "p_outlet", "Cinlet" ó "mass_flow".')
            sys.exit()

        if self.corrector_seed is not None:
            self.cfg.edit_relative_jump(self.cfg.SOLVER_DEC_TOL)

        rho_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='d')
        if m_dot is not None:
            C_inx = m_dot / (rho_in * self.cfg.geom['areas'][0])

        @solver_decorator(self.cfg, p_out, np.float64(C_inx_ref))
        def inner_solver(var_C_inx=None):
            nonlocal m_dot, C_inx, ps_list

            self.step_iter_mode = self.step_iter_end = False
            self.step_counter = 0
            self.Re_corrector_counter = 0

            if var_C_inx is not None:
                C_inx = var_C_inx
            if m_dot is None:
                m_dot = rho_in * self.cfg.geom['areas'][0] * C_inx

            s_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='s')
            h_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='h')
            h_0in = h_in + (0.001 * (C_inx**2) / 2)

            ps_list = [T_in, p_in, rho_in, s_in, h_in, h_0in, C_inx]
            for i in range(self.cfg.n_steps):
                list_i = ps_list[i-1] if i > 0 and not self.cfg.fast_mode else ps_list
                args = list_i[0], list_i[1], list_i[6], list_i[3], list_i[4], list_i[5], m_dot, n, list_i[2]

                if self.cfg.fast_mode:
                    ps_list = self.step_block(*args)
                else:
                    if i == 0:
                        ps_list = [copy.deepcopy(self.step_block(*args))]
                    else:
                        ps_list += [copy.deepcopy(self.step_block(*args))]

                self.Re_corrector_counter = 0
                self.step_counter += 1

            if not self.cfg.fast_mode:  # Los subindices A y B indican, resp., los pts. inicio y fin de la turbina.
                p_B, s_B, h_B, h_0B = [ps_list[-1][1]] + ps_list[-1][3:6]
                h_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='h')
                h_0A = h_0in
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
                eta_p = np.log10(1 - (eta_maq * (1 - (T_0Bss / T_0A)))) / np.log10(T_0Bss / T_0A)
                r_turbina = p_0A / p_0B
                ps_list += [[DELTA_h, w_total, P_total, s_A, s_B, p_0B, T_0B, T_0Bss, h_0Bss, Y_maq, w_ss_total,
                             eta_maq, p_0A, T_0A, eta_p, r_turbina, m_dot]]

            self.vmmr = ps_list
            return ps_list

        inner_solver()

        if self.cfg.fast_mode:
            return ps_list[0], ps_list[1], ps_list[6], np.degrees(ps_list[7])
        else:
            return

    def step_block(self, T_1, p_1, C_1, s_1, h_1, h_01, m_dot, n, rho_1) -> list:
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

        @step_decorator(self.cfg, corrector_memory)
        def inner_funct(iter_mode=False, iter_end=False, xi_est=None, rho_seed=None, Re_out=None):
            """ Esta función interna se crea para poder comunicar al decorador instancias de la clase.
                                :param rho_seed: Densidad que se emplea como valor semilla en blade_outlet_calculator
                                                 (kg/m^3).
                    :param xi_est: Valor opcional que permite al decorador aplicar recursividad para efectuar la
                                  corrección por dependencia de Reynolds.
                    :param iter_mode: Permite diferenciar si se está aplicando recursividad para la corrección por
                                 dependencia de Reynolds y así conocer las instrucciones más convenientes.
                    :param iter_end: Permite omitir cálculos innecesarios durante la corrección por dependencia de
                                    Reynolds.
                    :param Re_out: Lista de listas que contiene los valores de Reynolds a la salida de cada corona de
                                   cada escalonamiento.
                            :return: Se devuelven diferentes variables que se requieren según la situación."""

            step_count = self.step_counter
            self.step_iter_mode = iter_mode
            self.step_iter_end = iter_end

            if rho_seed is None:
                if self.rho_seed_list is not None and len(self.rho_seed_list) >= step_count+1:
                    rho_seed = self.rho_seed_list[step_count]
                else:
                    rho_seed = [rho_1*0.80, rho_1*0.75]
                    if step_count == 0:
                        self.rho_seed_list = []

            A_tpl = (self.cfg.geom['areas'][step_count * 2],
                     self.cfg.geom['areas'][(step_count * 2) + 1],
                     self.cfg.geom['areas'][(step_count * 2) + 2])

            eta_TT = Re_12 = Re_23 = Re = None

            if not iter_mode and not iter_end:
                pass
            else:
                self.Re_corrector_counter += 1
            registro.info('Modo de repetición: %s  ...  Llamadas: %s', iter_mode, self.Re_corrector_counter)

            if step_count > 0:
                C_1x = m_dot / (rho_1 * A_tpl[0])
                alfa_1 = np.arccos(C_1x / C_1)
            else:
                C_1x = C_1
                alfa_1 = 0.0
            registro.debug('La velocidad axial establecida a la entrada del escalonamiento %s es: %.2f m/s',
                           step_count + 1, float(C_1x))

            registro.info('Se va a calcular la salida del estátor del escalonamiento número %d',
                          step_count + 1)

            h_02 = h_01
            a_1 = self.prd.get_sound_speed(T=T_1, p=p_1)
            M_1 = C_1/a_1
            self.ref_values = (T_1, p_1 * 0.95)

            args = ['est', A_tpl[1], alfa_1, h_02, m_dot, s_1, rho_seed[0], M_1, rho_1, C_1, C_1x, T_1]
            if (not iter_mode and not iter_end) or self.cfg.loss_model == 'Aungier':
                outputs = self.blade_outlet_calculator(*args)
                p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2, xi_est, Re_12 = outputs
            else:
                outputs = self.blade_outlet_calculator(*args, xi=xi_est)
                p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2 = outputs

            s_2 = self.prd.get_prop(known_props={'p': p_2, 'T': T_2}, req_prop='s')

            # En las líneas que siguen se determinan los triángulos de velocidades
            U = self.cfg.geom['Rm'][(step_count * 2) + 1] * n * 2 * np.pi / 60
            registro.debug('La velocidad tangencial de giro en el radio medio es: %.2f m/s', float(U))
            C_2u = C_2 * np.sin(alfa_2)
            registro.debug('La velocidad tangencial del flujo a la entrada del rótor es: %.2f m/s', float(C_2u))
            omega_2u = C_2u - U
            omega_2x = C_2x
            beta_2 = np.arctan(omega_2u / C_2x)
            omega_2 = C_2x / np.cos(beta_2)

            registro.info(' Se va a calcular la salida del rótor del escalonamiento número %d.     ', step_count+1)
            h_r3 = h_r2 = h_2 + (0.001 * (omega_2 ** 2) / 2)
            self.ref_values = (T_2, p_2 * 0.95)

            outputs = self.blade_outlet_calculator(
                blade='rot', area_b=A_tpl[2], tau_a=beta_2, h_tb=h_r3,
                m_dot=m_dot, s_a=s_2, rho_outer_seed=rho_seed[1], M_a=M_2,
                rho_a=rho_2, C_a=C_2, C_ax=C_2x, T_a=T_2, Re_out=Re_out
            )

            if (not iter_mode and not iter_end) or self.cfg.loss_model == 'Aungier':
                p_3, h_3, T_3, omega_3, rho_3, h_3s, T_3s, C_3x, M_3r, beta_3, xi_rot, Re_23 = outputs
            else:
                p_3, h_3, T_3, omega_3, rho_3, h_3s, T_3s, C_3x, M_3r, beta_3, xi_rot = outputs

            omega_3u = omega_3 * np.sin(beta_3)
            omega_3x = omega_3 * np.cos(beta_3)
            C_3u = omega_3u - U
            C_3 = np.sqrt((C_3u**2) + (C_3x**2))
            alfa_3 = np.arcsin(C_3u / C_3)

            s_3 = self.prd.get_prop(known_props={'T': T_3, 'p': p_3}, req_prop='s')
            h_03 = h_3 + (0.001 * (C_3**2) / 2)
            local_list_1 = [T_3, p_3, rho_3, s_3, h_3, h_03, C_3, alfa_3]

            if h_02 - h_03 < 0:
                registro.error('No se extrae energía del fluido.')

            if self.cfg.loss_model == 'Ainley_and_Mathieson':
                # Media aritmética de Re a la entrada y a la salida recomendada por AM para la corrección.
                if not iter_mode and not iter_end:
                    Re = (Re_12 + Re_23)/2
                    registro.debug('Reynolds a la entrada: %d, Reynolds a la salida: %d, Reynolds promedio del '
                                   'escalonamiento: %d', Re_12, Re_23, Re)
                    if Re < 50_000:
                        registro.warning('El número de Reynolds es demasiado bajo.')

            if not iter_mode and not iter_end:
                if len(self.rho_seed_list) < self.cfg.n_steps:
                    self.rho_seed_list.append(copy.deepcopy([rho_2, rho_3]))
                else:
                    self.rho_seed_list[step_count] = copy.deepcopy([rho_2, rho_3])

            # Se determina en estas líneas el rendimiento total a total para que sea posible aplicar la corrección:
            if self.cfg.loss_model == 'Ainley_and_Mathieson' and (self.cfg.fast_mode or iter_mode):
                w_esc = h_02 - h_03
                p_03, T_03 = self.Zero_pt_calculator(p_x=p_3, s_x=s_3, h_0x=h_03)
                T_03ss = self.prd.get_prop(known_props={'s': s_1, 'p': p_03}, req_prop={'T': T_03})
                h_03ss = self.prd.get_prop(known_props={'T': T_03ss, 'p': p_03}, req_prop='h')
                Y_esc = h_03 - h_03ss
                eta_TT = w_esc / (w_esc + Y_esc)

            if iter_end:
                if len(self.corrector_seed) < self.cfg.n_steps:
                    if self.cfg.loss_model == 'Ainley_and_Mathieson':
                        self.corrector_seed.append(copy.deepcopy([xi_est, [rho_2, rho_3]]))
                    else:
                        self.corrector_seed.append(copy.deepcopy([Re_23, [rho_2, rho_3]]))
                else:
                    if self.cfg.loss_model == 'Ainley_and_Mathieson':
                        self.corrector_seed[step_count] = copy.deepcopy([xi_est, [rho_2, rho_3]])
                    else:
                        self.corrector_seed[step_count] = copy.deepcopy([Re_23, [rho_2, rho_3]])

            if not self.cfg.fast_mode and (iter_end or not iter_mode):
                Y_est = xi_est * (0.001 * (C_2**2) / 2)
                Y_rot = xi_rot * (0.001 * (omega_3**2) / 2)
                w_esc = h_02 - h_03
                p_03, T_03 = self.Zero_pt_calculator(p_x=p_3, s_x=s_3, h_0x=h_03)
                T_03ss = self.prd.get_prop(known_props={'s': s_1, 'p': p_03}, req_prop={'T': T_03})
                h_03ss = self.prd.get_prop(known_props={'T': T_03ss, 'p': p_03}, req_prop='h')
                Y_esc = h_03 - h_03ss
                eta_TT = w_esc / (w_esc + Y_esc)
                C_1u = np.sqrt((C_1**2) - (C_1x**2))
                a_1 = self.prd.get_sound_speed(T=T_1, p=p_1)
                M_1 = C_1 / a_1
                a_3 = self.prd.get_sound_speed(T=T_3, p=p_3)
                M_3 = C_3 / a_3
                Pot_esc = w_esc * m_dot
                GR = (h_2 - h_3) / w_esc
                PSI = w_esc / (U**2)
                PHI_2 = C_2x / U
                PHI_3 = C_3x / U
                C_2s = np.sqrt((h_02 - h_2s) * 2000)
                omega_3s = np.sqrt((h_r3 - h_3s) * 2000)
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
                eta_TE = w_esc / (w_esc + Y_esc + (0.001 * (C_3**2) / 2))
                # Aunque resulta más relevante la relación isentrópica, es posible calcular el rendimiento politrópico
                # Se emplean para ello los valores de remanso
                p_01, T_01 = self.Zero_pt_calculator(p_x=p_1, s_x=s_1, h_0x=h_01)
                eta_p_esc = np.log10(1 - (eta_TT * (1 - (T_03ss / T_01)))) / np.log10(T_03ss / T_01)
                r_esc = p_01 / p_03
                local_list_1 += [C_3x, C_3u, T_03, p_03, h_r3, h_3s, T_3s, omega_3, omega_3x, omega_3u, beta_3, M_3r,
                                 M_3, PHI_3, T_2, p_2, rho_2, s_2, h_2, h_02, C_2, C_2x, C_2u, alfa_2, T_02, p_02, h_r2,
                                 h_2s, T_2s, omega_2, omega_2x, omega_2u, beta_2, M_2, PHI_2, T_1, p_1, rho_1, s_1, h_1,
                                 h_01, C_1, C_1x, C_1u, alfa_1, T_01, p_01, M_1, Y_est, xi_est, Y_rot, xi_rot, w_esc,
                                 Pot_esc, eta_p_esc, r_esc, GR, PSI, speed_loss_coeff_E, speed_loss_coeff_R,
                                 DELTA_h_esc, eta_TT, eta_TE, Y_esc, w_s_esc, w_ss_esc, C_2s, T_02s, h_02s, omega_3s,
                                 T_3ss, T_03s, T_03ss, h_3ss, h_03s, h_03ss, U]

                if self.cfg.loss_model == 'Ainley_and_Mathieson':
                    return Re, eta_TT, xi_est, [rho_2, rho_3], local_list_1
                else:
                    return Re_23, [rho_2, rho_3], local_list_1

            else:
                if self.cfg.loss_model == 'Ainley_and_Mathieson':
                    return Re, eta_TT, xi_est, [rho_2, rho_3], local_list_1
                else:
                    return Re_23, [rho_2, rho_3], local_list_1

        ll_1 = inner_funct()
        return ll_1

    def blade_outlet_calculator(self, blade: str, area_b, tau_a, h_tb, m_dot, s_a, rho_outer_seed,
                                M_a, rho_a, C_a, C_ax, T_a, xi=None, Re_in=None, Re_out=None):
        """ Se hace un cálculo iterativo para conocer las propiedades a la salida del estátor y del rótor (según el
        caso, estátor / rótor, se proporciona el valor de la entalpía total / rotalpía y del ángulo que forma la
        velocidad absoluta / relativa del fluido con la dirección del eje de la turbina axial, respectivamente).

        Punto a: Sección al principio del álabe

        Punto b: Sección al final del álabe

            :param Re_out: Lista que se recibe como argumento en inner_funct en step_block.
            :param rho_a: Desidad a la entrada (kg/m^3).
            :param C_a: Velocidad absoluta a la entrada (m/s).
            :param C_ax: Velocidad axial a la entrada (m/s).
            :param T_a: Temperatura a la entrada (K).
            :param Re_in: Valor del número de Reynolds a la entrada del estátor (-).
            :param xi: Coeficiente adimensional de pérdidas de la corona en cuestión (-).
            :param tau_a: Ángulo del movimiento absoluto/relativo de entrada del fluido con la dirección axial
                         (rads).
            :param blade: Diferenciador del tipo de álabe, puede ser 'est' o 'rot'.
            :param area_b: El área de la sección de paso al final de la corona (m^2).
            :param M_a: Mach a la entrada (-).
            :param h_tb: Entalpía total / rotalpía a la salida del álabe (kJ/kg).
            :param m_dot: Flujo másico (kg/s).
            :param s_a: Entropía en a (kJ/kgK).
            :param rho_outer_seed: Densidad que se emplea como semilla, debe aproximar la densidad en b (kg/m^3).
                    :return: Se devuelven las variables que contienen las propiedades a la salida que se han
                    determinado (p_b, h_b, T_b, U_b, rho_b, h_bs, T_bs, C_bx). """

        if self.cfg.loss_model == 'Ainley_and_Mathieson':
            self.AM_object.limit_mssg = [True, True, True]

        rho_b = rho_bp = rho_outer_seed
        M_b = h_b = U_b = h_bs = C_bx = Y_total = Re = pr0_b = Tr0_b = None
        rel_diff, tol, geom = 1.0, self.cfg.TOL, self.cfg.geom

        if self.first_seeds_boc is None:
            T_b, T_bs, p_b = self.ref_values[0] * 0.9, self.ref_values[0], self.ref_values[1]
        else:
            T_b, T_bs, p_b = self.first_seeds_boc

        counter = self.step_counter
        iter_count = 0
        num = counter*2 + (0 if blade == 'est' else 1)

        alfap_1 = geom[f'alfap_i_{blade}'][counter]
        alfap_2 = geom[f'alfap_o_{blade}'][counter]

        if self.cfg.loss_model == 'Ainley_and_Mathieson':
            # tau_2 debe ser corregida si Mb < 0.5, ahora se asigna el valor calculado inicialmente en cualquier caso.
            # El valor de xi se conoce únicamente cuando iter_mode y estátor, pero no el de tau_2 (Y_total no default).
            # El resto de veces depende xi depende de si se modifica o no el valor de tau_2.
            tau_b = self.AM_object.outlet_angle_before_mod[num]
            Re_in = Reynolds(counter*2, rho_a, C_a, T_a, self.cfg, self.prd)

        else:
            tau_b = alfap_2

        # p: iteración previa .... b: estado que se quiere conocer, a la salida del álabe
        while np.fabs(rel_diff) > tol:
            iter_count += 1

            if iter_count > self.cfg.iter_limit:
                registro.error('Iteración aboratada, no se cumple el criterio de convergencia.')
                raise NonConvergenceError

            C_bx = m_dot / (area_b * rho_b)  # C_bx: velocidad axial a la salida
            tau_b_n = None

            while tau_b_n is None or np.fabs((tau_b_n - tau_b)/tau_b) > tol:

                if tau_b_n is None:
                    tau_b_n = tau_b
                else:
                    tau_b = tau_b_n

                U_b = C_bx / np.cos(tau_b)  # U_b: vel. absoluta o relativa ... tau: alfa o beta ... según el caso
                h_b = h_tb-(0.001*(U_b**2)/2)
                # Se aplica conservación de la entalpía total/rotalpía ... según el caso
                # (no se puede determinar otra variable con h2s y s1 con PyroMat)

                p_b = self.prd.get_prop(known_props={'d': rho_b, 'h': h_b}, req_prop={'p': p_b})
                T_b = self.prd.get_prop(known_props={'h': h_b, 'd': rho_b}, req_prop={'T': T_b})
                a_b = self.prd.get_sound_speed(T_b, p_b)
                gamma_b = self.prd.get_gamma(T_b, p_b)
                T_bs = self.prd.get_prop(known_props={'p': p_b, 's': s_a}, req_prop={'T': T_bs})
                h_bs = self.prd.get_prop(known_props={'T': T_bs, 'p': p_b}, req_prop='h')

                M_b = U_b/a_b   # Mout del flujo absoluto o relativo según si estátor o rótor (respectivamente)

                if self.cfg.loss_model == 'Ainley_and_Mathieson':
                    if not self.step_iter_mode and not self.step_iter_end:
                        if blade == 'est':
                            Re = Re_in
                        else:
                            Re = Reynolds(num, rho_b, U_b, T_b, self.cfg, self.prd)

                if self.cfg.loss_model == 'Ainley_and_Mathieson':
                    tau_b_n = self.AM_object.tau2_corrector(num, M_b)
                    # Se ejecuta el primer bloque a excepción de si es estátor y step_iter_mode.
                    if (not self.step_iter_mode and not self.step_iter_end) or blade == 'rot':
                        args = [num, np.degrees(tau_a), np.degrees(tau_b), self.step_iter_mode or self.step_iter_end]
                        Y_total = self.AM_object.Ainley_and_Mathieson_Loss_Model(*args)
                        xi = Y_total / (1 + (0.5*gamma_b*(M_b**2)))
                    else:
                        Y_total = xi * (1 + (0.5*gamma_b*(M_b**2)))
                        args = [num, np.degrees(tau_a), np.degrees(tau_b), True, Y_total]
                        self.AM_object.Ainley_and_Mathieson_Loss_Model(*args)

                elif self.cfg.loss_model == 'Aungier':
                    s_b = self.prd.get_prop(known_props={'T': T_b, 'p': p_b}, req_prop='s')
                    pr0_b, Tr0_b = self.Zero_pt_calculator(p_b, s_b, h_tb, p_0x=pr0_b, T_0x=Tr0_b)
                    Y_total, tau_b_n = self.AUNGIER_object.Aungier_operations(
                        num=num, Min=M_a, Mout=M_b, Re_c=Re_out, tau_1=tau_a, V_2x=C_bx,
                        V_1x=C_ax, p_2=p_b, pr0_2=pr0_b, d2=rho_b, d1=rho_a, U_2=U_b
                    )
                    xi = Y_total / (1 + (0.5*gamma_b*(M_b**2)))
                    if not self.step_iter_mode:
                        tau_b = tau_b_n

            h_b = (0.001 * xi * (U_b**2) / 2) + h_bs
            rho_b = self.prd.get_prop({'p': p_b, 'h': h_b}, {'d': rho_b})
            rel_diff = (rho_b - rho_bp) / rho_b

            rho_bp = rho_b

            registro.debug('Densidad (kg/m^3): %.12f  ...  Error relativo: %.12f', float(rho_b), float(rel_diff))

        if M_b > 0.5:
            registro.warning('Mout %sa la salida superior a 0.5 ... Valor: %.2f',
                             '' if num % 2 == 0 else 'relativo ', float(M_b))
        else:
            registro.debug('Valor del número de Mach %sa la salida: %.2f',
                           '' if num % 2 == 0 else 'relativo ', float(M_b))

        if (self.step_iter_mode or self.step_iter_end) and self.cfg.loss_model == 'Ainley_and_Mathieson':
            registro.debug('La relación entre el valor actual y el inicial de las pérdidas adimensionales de presión '
                           'en ambas coronas del escalonamiento %s es: %.3f\n  ...   ',
                           1 + (num//2), float(Y_total / self.AM_object.Y_t_preiter[num]))

        registro.debug('Incidencia: %.2f°  ...  tau_in: %.2f°  ...  Ángulo del B.A.: %.2f°',
                       degrees(float(tau_a)) - degrees(float(alfap_1)),
                       degrees(float(tau_a)), degrees(float(alfap_1)))
        registro.debug('Desviación: %.2f°  ...  tau_out: %.2f°  ...  Ángulo del B.S.: %.2f°',
                       degrees(float(tau_b)) - degrees(float(alfap_2)),
                       degrees(float(tau_b)), degrees(float(alfap_2)))

        if self.cfg.loss_model == 'Ainley_and_Mathieson':
            if not self.step_iter_mode and not self.step_iter_end:
                Yp = self.AM_object.Yp_preiter[num]
            else:
                Yp = self.AM_object.Yp_iter_mode
        else:
            Yp = self.AUNGIER_object.Yp_iter_mode
        registro.debug('Pérdidas:  ...  Y_total: %.4f  ...  Yp: %.4f   ', float(Y_total), float(Yp))

        if self.cfg.loss_model == 'Aungier':
            if blade == 'rot':
                Re = Reynolds(num, rho_b, U_b, T_b, self.cfg, self.prd)

        self.first_seeds_boc = copy.deepcopy([T_b, T_bs, p_b])
        return_vars = [p_b, h_b, T_b, U_b, rho_b, h_bs, T_bs, C_bx, M_b, tau_b]

        if blade == 'est':
            if (not self.step_iter_mode and not self.step_iter_end) or self.cfg.loss_model == 'Aungier':
                return *return_vars, xi, Re
            else:
                return return_vars
        else:
            if (not self.step_iter_mode and not self.step_iter_end) or self.cfg.loss_model == 'Aungier':
                return *return_vars, xi, Re
            else:
                return *return_vars, xi

    def Zero_pt_calculator(self, p_x, s_x, h_0x, p_0x=None, T_0x=None):
        """ Este método es para determinar presiones y temperaturas de remanso.
                        :param p_0x: Estimación inicial del valor de presión de remanso (Pa).
                :param T_0x: Estimación inicial del valor de temperatura de remanso (K).
                :param p_x: Presión en la sección x (Pa).
                :param s_x: Entropía en la sección x (kJ/kgK).
                :param h_0x: Entalpía total en la sección x (kJ/kg).
                        :return: Devuelve la presión total (Pa) y la temperatura total (K) en una sección x. """

        if p_0x is None and T_0x is None:
            p_0x, T_0x = p_x * 1.1, self.ref_values[0] * 1.1
        end, tol = False, self.cfg.TOL

        while not end:
            T_0x = self.prd.get_prop(known_props={'h': h_0x, 'p': p_0x}, req_prop={'T': T_0x})
            if self.cfg.ideal_gas:
                p_0x = self.prd.get_prop(known_props={'T': T_0x, 's': s_x}, req_prop={'p': p_0x})
                end = True
            else:
                p_0x_iter = self.prd.get_prop(known_props={'T': T_0x, 's': s_x}, req_prop={'p': p_0x})
                if np.fabs(p_0x - p_0x_iter) / p_0x < tol:
                    end = True
                p_0x = p_0x_iter

        return p_0x, T_0x


def main():
    fast_mode = False
    settings = config_parameters(TOL=1E-12, STEP_DEC_TOL=1E-11, SOLVER_DEC_TOL=1E-10,
                                 relative_jump=0.01, iter_limit=1200, ideal_gas=True,
                                 n_steps=1, fast_mode=fast_mode, loss_model='Aungier')

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
    tip_clearance = [0.0004, 0.0008]
    # 'wire_diameter' 'lashing_wires'
    chord_proj_z = [0.9*b for b in chord]
    blade_roughness_peak_to_valley = [0.00001 for _ in chord]

    settings.set_geometry(B_A_est=0, theta_est=70, B_A_rot=36, theta_rot=100, cuerda=chord,
                          radio_medio=Rm, s=pitch, H=heights, b_z=chord_proj_z, areas=areas,
                          t_max=t_max, r_r=0.002, r_c=0.001, t_e=t_e, k=tip_clearance,
                          delta=tip_clearance, roughness_ptv=blade_roughness_peak_to_valley,
                          e=e_param, o=blade_opening, holgura_radial=False)

    gas_model = gas_model_to_solver(thermo_mode="ig", relative_error=1E-12)
    solver = solver_object(settings, gas_model)

    if fast_mode:
        output = solver.problem_solver(T_inlet=1100, p_inlet=375_790, rpm=20_000, p_outlet=250_000, C_inlet_ref=140)
        T_salida, p_salida, C_salida, alfa_salida = output
        print(' T_out =', T_salida, '\n', 'P_out =', p_salida,
              '\n', 'C_out =', C_salida, '\n', 'alfa_out =', alfa_salida)

    else:
        solver.problem_solver(T_inlet=1500, p_inlet=375_790, rpm=27_000,  p_outlet=200_000, C_inlet_ref=150)


if __name__ == '__main__':
    main()
