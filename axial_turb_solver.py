"""
This module creates a class that allows to determine the operating conditions of the axial turbine to be defined, given
the temperature and pressure at the entrance and another variable such as inlet speed, mass flow or pressure at the
outlet.
"""

import copy

from math import log
from time import time

import numpy

from loss_model import *


def solver_decorator(solver, p_out: float | None, C_inx_estimated: float | None,
                     small_deviations_data: list | None):
    """
    Outer decorator of solver's main method in order to define the necessary arguments to handle it and allow
    to determine the operating conditions given the pressure at the outlet.

    Args:
        solver: This is an object for a resolutional purpose.
        p_out: Pressure at the turbine outlet (Pa).
        C_inx_estimated: Estimated inlet velocity to be received when the outlet pressure is set (m/s).
        small_deviations_data: When this variable is not None, it consists on a list of velocity values evaluated,
        a numpy array with outlet pressures resulting and arrays with partial derivatives at each velocity evaluated.
    """
    cfg = solver.cfg

    def solver_inner_decorator(inner_funtion_from_problem_solver):
        """ Inner decorator that manages the internal function of the problem_solver method from solver_class.

        Args:
            inner_funtion_from_problem_solver: Funtion to be decorated.
        """

        def corrector_for_small_deviations():
            C_in_eval_list, p_out_vref, dpout_dTin, dpout_dpin, \
                dpout_dn, T_in, p_in, n_rev = copy.deepcopy(small_deviations_data)
            Tin_ref, pin_ref, nrev_ref = cfg.T_nominal, cfg.p_nominal, cfg.n_rpm_nominal
            p_out_vref += ((T_in-Tin_ref)*dpout_dTin)+((p_in-pin_ref)*dpout_dpin)+((n_rev-nrev_ref)*dpout_dn)
            p_out_vref, C_in_eval_list = p_out_vref[::-1], C_in_eval_list[::-1]
            if isinstance(p_out_vref, numpy.ndarray):
                p_out_vref = p_out_vref.tolist()
            if isinstance(C_in_eval_list, numpy.ndarray):
                C_in_eval_list = C_in_eval_list.tolist()
            i = 0
            while i < len(p_out_vref)-1:
                if p_out_vref[i+1] < p_out_vref[i]:
                    del p_out_vref[i+1]
                    del C_in_eval_list[i+1]
                    i = 0
                else:
                    i += 1
            C_in_resulting = lineal_interpolation(x_target=p_out, x=p_out_vref, y=C_in_eval_list, order=1)
            return C_in_resulting

        def iterate_ps():
            """ Function to be used whenever the spline attribute of the solver is not set. """
            C_inx = C_inx_estimated
            ps_list = None

            def read_ps_list():
                if cfg.chain_mode:
                    return ps_list.copy[1]
                else:
                    return copy.deepcopy(ps_list)[-1][1]
            iter_count = 0
            from_b = from_a = False
            start = True
            # Points "a" and "b" such that C_inx_b > C_inx_a.
            delta = cfg.jump
            C_inx_a = C_inx - 0.5
            C_inx_b = C_inx
            pre_C_inx_a = pre_C_inx_b = C_inx
            p_out_iter_b = p_out_iter_a = None
            ps_list_a = ps_list_b = None

            def C_in_algorithm():
                nonlocal C_inx_a, C_inx_b, from_a, from_b, C_inx, pre_C_inx_a, pre_C_inx_b
                if p_out_iter_b > p_out:
                    # Here goes the level to increase velocity at point "b".
                    C_inx_a = C_inx_b
                    C_inx_b += delta
                    from_b = True  # To indicate where does the process flow come from
                    C_inx = C_inx_b
                elif p_out_iter_a < p_out:
                    # Here goes the level to decrease velocity at point "a".
                    C_inx_b = C_inx_a
                    C_inx_a -= delta
                    from_a = True
                    C_inx = C_inx_a
                return

            def first_iter_exception_task():
                solver.seed_reset()
                raise OuterLoopConvergenceError('Try another seed value.')

            def post_exception_tasks():
                record.warning('An exception was caught.')
                nonlocal C_inx_a, C_inx_b, delta
                # Returning to previous values.
                C_inx_a, C_inx_b = pre_C_inx_a, pre_C_inx_b
                # Then, reducing the relative jump.
                delta /= 2.5
                solver.seed_reset()
                return

            solver_relative_error = cfg.relative_error
            #  It must be noted that, when problem has convergence, the numerical error of all methods being
            #  used is smaller the more iterations are performed for the same evaluation.

            record.info('Searching for a range containing the solution.')
            # The search begins.
            while True:
                iter_count += 1
                try:
                    if start:
                        ps_list_a = inner_funtion_from_problem_solver(C_inx_a, False)
                        ps_list_b = inner_funtion_from_problem_solver(C_inx_b, False)
                    else:
                        ps_list = inner_funtion_from_problem_solver(C_inx, False)
                except InnerLoopConvergenceError:
                    if start:
                        first_iter_exception_task()
                    else:
                        post_exception_tasks()
                except GasLibraryAdaptedException:
                    if start:
                        first_iter_exception_task()
                    else:
                        post_exception_tasks()
                else:
                    # Saving previous values before doing changes, these are required in case of exceptions.
                    pre_C_inx_a, pre_C_inx_b = C_inx_a, C_inx_b
                    if start:
                        ps_list = ps_list_a
                        p_out_iter_a = read_ps_list()
                        ps_list = ps_list_b
                        p_out_iter_b = read_ps_list()
                        start = False
                    else:
                        p_out_iter = read_ps_list()
                        if from_b:
                            # Process flow comes from the block that increases velocity of point "b".
                            p_out_iter_a = p_out_iter_b
                            p_out_iter_b = p_out_iter
                            from_b = False
                        elif from_a:
                            # Process flow comes from the block that decreases velocity of point "a".
                            p_out_iter_b = p_out_iter_a
                            p_out_iter_a = p_out_iter
                            from_a = False
                        else:
                            solver.seed_reset()
                            record.error('Something went wrong.')
                            raise OuterLoopConvergenceError()

                    # It is evaluated whether the new range contains the solution.
                    if (p_out_iter_b-p_out)*(p_out_iter_a-p_out) < 0:
                        record.info('Operation point has been located.')
                        start = True
                        break

                finally:
                    if not start:
                        C_in_algorithm()
                        record.info('Current range: [%.2f, %.2f]  ...  Target value: %.2f',
                                    p_out_iter_a, p_out_iter_b, p_out)

                if iter_count > cfg.iter_limit_OL:
                    solver.seed_reset()
                    record.error('Search took too many evaluations, check inlet velocity seed value or iter limit for '
                                 'outer loops.')
                    raise OuterLoopConvergenceError()

            rel_error = None
            p_out_iter = None
            f_a = f_b = None
            diff_value = None
            iter_count = 0

            def update_C_inx():
                nonlocal diff_value, C_inx
                diff_value = (p_out_iter_b-p_out) * (C_inx_b - C_inx_a) / (p_out_iter_b - p_out_iter_a)
                if diff_value > 0.75 * (C_inx_b - C_inx_a):
                    diff_value = 0.75 * (C_inx_b - C_inx_a)
                elif diff_value < 0.25 * (C_inx_b - C_inx_a):
                    diff_value = 0.25 * (C_inx_b - C_inx_a)
                C_inx = C_inx_b - diff_value
            non_progression_counter = 0
            while rel_error is None or rel_error >= cfg.relative_error:  # Applying Regula Falsi
                iter_count += 1
                if rel_error is None:
                    rel_error = solver_relative_error
                    f_a = p_out_iter_a-p_out
                    f_b = p_out_iter_b-p_out
                    update_C_inx()
                try:
                    ps_list = inner_funtion_from_problem_solver(C_inx, False)
                except InnerLoopConvergenceError:
                    # This event will most likely only happen when limits are not high enough.
                    C_inx_b, C_inx_a = pre_C_inx_b, pre_C_inx_a
                    solver.seed_reset()
                except GasLibraryAdaptedException:
                    C_inx_b, C_inx_a = pre_C_inx_b, pre_C_inx_a
                    solver.seed_reset()
                else:
                    pre_C_inx_b, pre_C_inx_a = C_inx_b, C_inx_a  # Old values are stored
                    p_out_iter = read_ps_list()
                    f_c = p_out_iter - p_out
                    pre_rel_error = rel_error
                    if f_b * f_c <= 0:
                        f_c = f_a = p_out_iter - p_out
                        C_inx_a = C_inx
                        p_out_iter_a = p_out_iter
                        rel_error = fabs(f_c) / p_out
                        update_C_inx()
                    elif f_a * f_c <= 0:
                        f_c = f_b = p_out_iter - p_out
                        C_inx_b = C_inx
                        p_out_iter_b = p_out_iter
                        rel_error = fabs(f_c) / p_out
                        update_C_inx()
                    # This is a patch for an observed behaviour from the Aungier loss model:
                    if fabs(pre_rel_error - rel_error)/rel_error <= 1e-3:
                        iter_count -= 1
                        non_progression_counter += 1
                        if non_progression_counter > cfg.iter_limit_OL:
                            if rel_error < 5*1e-5:
                                record.warning('Relative error kept the same value too much time and it is low '
                                               'enough, work point is admited.')
                                break
                            else:
                                record.error('Relative error kept the same value too much time and it is not low '
                                             'enough, work point is skipped.')
                                raise OuterLoopConvergenceError()

                record.info('Error de presión a la salida: %.10f  ...  Valor actual: %.2f Pa ...  '
                            'Valor objetivo: %.2f Pa', rel_error, p_out_iter, p_out)

                if iter_count > cfg.iter_limit_OL:
                    if rel_error > 5*1e-5:
                        solver.seed_reset()
                        record.error('Recursive calculation does not reach convergence when fixing the desired outlet '
                                     'pressure.')
                        raise OuterLoopConvergenceError()
                    else:
                        break

            if not cfg.chain_mode:
                ps_list = inner_funtion_from_problem_solver(C_inx, True)
            return

        def wrapper_s():
            """ Se evalúa cuánto tiempo se tarda en determinar la condición de funcionamiento, que se obtendrá de
            manera directa o indirecta según corresponda."""

            t_1 = time()
            if p_out is None and small_deviations_data is None:
                _ = inner_funtion_from_problem_solver(None, True)
            else:
                if small_deviations_data is None:
                    iterate_ps()
                else:
                    C_in = corrector_for_small_deviations()
                    _ = inner_funtion_from_problem_solver(C_in, True)
            t_2 = (time() - t_1).__round__(0)

            m, s = divmod(t_2, 60)
            record.info('Tiempo de cálculo del punto de funcionamiento: %s minutos y %s segundos.', int(m), int(s))

            return

        return wrapper_s

    return solver_inner_decorator


def step_decorator(solver, step_corrector_memory):
    """ Decorador del decorador real, permite referenciar los parámetros necesarios para manipular la salida de la
    función decorada.
                :param solver: This is an object for a resolutional purpose.
                :param step_corrector_memory: Lista con las variables de la ejecución previa.
                            :return: Se devuelve el decorador real. """
    cfg = solver.cfg

    def Reynolds_corrector(step_inner_function):
        """ Decorador real, gestiona la función interna del método gen_steps de la clase solver y posibilita
         aplicar la corrección por el número de Reynolds.
                    :param step_inner_function: Función que se decora.
                                :return: Se devuelve el wrapper_r. """

        relative_error = cfg.relative_error

        def get_sif_output(iter_mode: bool = False, iter_end: bool = False, xi=None, rho_seed: list = None, Re=None):
            """ Se recibe la copia del output generado y se devuelve. """
            return step_inner_function(iter_mode, iter_end, xi, rho_seed, Re)

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

            nonlocal relative_error
            target_efficiency = 1 - (((1 - eta_TT) / (200_000 ** (-1 / 5))) * (Re ** (-1 / 5)))
            f1 = f2 = fpu = None
            bolz_c = None

            xi_e0 = xi_est * (Re ** (-1 / 5)) / (200_000 ** (-1 / 5))
            xi_e1 = xi_e2 = xi_e0
            rho_seed_1 = rho_seed_2 = rho_seed_c = rho_seed
            iter_counter = 0

            while bolz_c is None or bolz_c > 0:
                iter_counter += 1
                record.info('Buscando el rango que garantice encontrar la solución.')
                if bolz_c is None:
                    sif1 = sif2 = get_sif_output(True, False, xi_e1, rho_seed_1)
                    f1, rho_seed_1 = sif1[1] - target_efficiency, sif1[3]
                    f2, rho_seed_2 = sif2[1] - target_efficiency, sif2[3]
                else:
                    if fpu is None or fabs(f1-f2) < relative_error*fabs(f1):
                        fpu = -1
                    else:
                        fp = (f2 - f1) / (xi_e2 - xi_e1)
                        fpu = fp/fabs(fp)
                    if f1*fpu > 0:
                        f2, rho_seed_2, xi_e2 = f1, rho_seed_1, xi_e1
                        xi_e1 *= 0.98
                        sif1 = get_sif_output(True, False, xi_e1, rho_seed_1)
                        f1, rho_seed_1 = sif1[1] - target_efficiency, sif1[3]
                    elif f2*fpu < 0:
                        f1, rho_seed_1, xi_e1 = f2, rho_seed_2, xi_e2
                        xi_e2 *= 1.02
                        sif2 = get_sif_output(True, False, xi_e2, rho_seed_2)
                        f2, rho_seed_2 = sif2[1] - target_efficiency, sif2[3]
                bolz_c = f1 * f2
                if iter_counter > cfg.iter_limit_OL:
                    solver.seed_reset()
                    raise OuterLoopConvergenceError('Search for Reynolds correction took too many evaluations, '
                                                    'it may help checking iter limit for outer loops.')

            record.info('Corrección iniciada.')

            pre_rel_error_eta_TT = 2.0
            rel_error_eta_TT = 1.0
            xi_ec = None
            jam_counter = iter_counter = 0

            while fabs(rel_error_eta_TT) > relative_error:
                iter_counter += 1
                xi_ec = xi_e2 - (f2 * (xi_e2 - xi_e1) / (f2 - f1))
                if fabs(pre_rel_error_eta_TT-rel_error_eta_TT) > relative_error*fabs(rel_error_eta_TT):
                    if xi_ec < (0.8*xi_e1) + (0.2*xi_e2):
                        xi_ec = (0.8*xi_e1) + (0.2*xi_e2)
                    elif xi_ec > (0.8*xi_e2) + (0.2*xi_e1):
                        xi_ec = (0.8*xi_e2) + (0.2*xi_e1)
                else:
                    jam_counter += 1
                    if jam_counter > cfg.iter_limit_OL:
                        solver.seed_reset()
                        raise OuterLoopConvergenceError()
                if iter_counter > cfg.iter_limit_OL:
                    if fabs(rel_error_eta_TT) < 1.5*1e-4:
                        break
                    else:
                        solver.seed_reset()
                        raise OuterLoopConvergenceError()
                sifc = get_sif_output(True, False, xi_ec, rho_seed_c)
                fc, rho_seed_c = sifc[1]-target_efficiency, sifc[3]
                if fc * f2 <= 0:
                    xi_e1, rho_seed_1, f1 = xi_ec, rho_seed_c, fc
                elif fc * f1 <= 0:
                    xi_e2, rho_seed_2, f2 = xi_ec, rho_seed_c, fc

                pre_rel_error_eta_TT = rel_error_eta_TT
                rel_error_eta_TT = fc / target_efficiency
                record.info('Corrección en proceso  ...  eta_TT: %.4f  ...  Error: %.7f', sifc[1], rel_error_eta_TT)

            _, _, _, _, ll_1 = get_sif_output(True, True, xi_ec, rho_seed_c)
            record.info('Corrección finalizada.')
            return ll_1

        def AU_corrector():
            if step_corrector_memory is None:
                Re, rho_seed, _ = get_sif_output()
            else:
                Re, rho_seed = step_corrector_memory[0], step_corrector_memory[1]
            relative_deviation = None
            iter_counter = 0
            while relative_deviation is None or relative_deviation > relative_error:
                iter_counter += 1
                if iter_counter > cfg.iter_limit_OL:
                    solver.seed_reset()
                    raise OuterLoopConvergenceError('Reynolds no se estabiliza para el límite de iteraciones '
                                                    'establecido.')
                Re_n = Re
                Re, rho_seed, _ = get_sif_output(True, False, None, rho_seed, Re_n)
                relative_deviation = fabs(Re_n - Re) / Re
            _, _, ll_1 = get_sif_output(True, True, None, rho_seed, Re)
            return ll_1

        def wrapper_r():
            """ Función que evalúa el modelo de pérdidas que se ha definido y, si es el de Ainley and Mathieson, aplica
            una corrección basándose en el número de Reynolds.
                            :return: Se devuelve la lista de variables que se procesan, según el modo que se defina. """

            if cfg.loss_model == 'Ainley_and_Mathieson':
                Re, eta_TT, xi_est, rho_seed, _ = get_sif_output(rho_seed=step_corrector_memory)
                ll_1 = AM_corrector(eta_TT, Re, xi_est, rho_seed)
            else:
                ll_1 = AU_corrector()
            return ll_1.copy()

        return wrapper_r
    return Reynolds_corrector


class solver_object:
    """ Clase que define un objeto que agrupa y coordina al conjunto de métodos/atributos que conforman
    el procedimiento resolutivo propuesto para determinar unas condiciones de funcionamiento de una turbina axial."""

    def __init__(self, config: config_class, productos: gas_model_to_solver):
        """ :param config: Objeto que agrupa lo relativo a la configuración establecida para la ejecución del solver."""

        self.vmmr = []  # Almacena ciertas variables, para facilitar la comunicación de sus valores
        self.small_input_deviation_data: None | list = None
        self.cfg = config  # Objeto que contiene los parámetros de interés para la ejecución del solver.
        self.rho_seed_list = None  # Para aligerar los cálculos para variaciones pequeñas de las variables de entrada
        self.prd = productos  # Modela el comportamiento termodinámico de los productos de la combustión
        self.prd.modify_relative_error(self.cfg.relative_error)
        self.loss_model_object = None
        self.ref_values = None  # Valores de referencia en las primeras semillas dentro de "blade_outlet_calculator".
        self.first_seeds_boc = None
        # En una nueva evaluación fijando p_out próximo se ahorra tiempo por conocer la solución anterior.
        self.C_inx_register = None  # Recordatorio del valor empleado en la evaluación anterior.
        self.AU_Re_register = None  # Para que no oscile tanto durante la estimación del tramo asintótico
        self.step_iter_mode = False
        self.step_iter_end = False
        self.Re_corrector_counter = 0  # Contador del número de llamadas efectuadas durante la corrección por Re.
        self.step_counter = 0  # Número empleado para iterar por cada escalonamiento comenzando por 0.
        self.corrector_seed = None
        # corrector_seed: Lista de listas que contienen datos concluidos por el corrector en cada escalonamiento
        #                 para aprovecharlos en cálculos consecutivos.

        AM_extra_parameters = ['e', 'o', 't_max', 'r_h', 'r_t', 't_e', 'k']
        Aungier_extra_parameters = AM_extra_parameters + ['roughness_ptv', 'b_z', 'delta']
        loss_model_list_for_extra_data = [AM_extra_parameters, Aungier_extra_parameters]
        list_index = 1 if config.loss_model == 'Aungier' else 0
        for key in loss_model_list_for_extra_data[list_index]:
            if self.cfg.geom.get(key, None) is None:
                sentence = f'Para emplear el modelo de pérdidas se debe introducir {key}.'
                raise InputDataError(sentence)
        if list_index == 0:
            self.loss_model_object = Ainley_and_Mathieson_Loss_Model(config)
            self.loss_model_object.AM_diameter_def()
        else:
            self.loss_model_object = Aungier_Loss_Model(config)

        if self.cfg.preloading_for_small_input_deviations:
            self.data_collector_for_small_input_deviations()

    def seed_reset(self):
        self.corrector_seed = self.rho_seed_list = self.ref_values = self.AU_Re_register = None
        self.first_seeds_boc = self.C_inx_register = None

    def data_collector_for_small_input_deviations(self):
        record.debug('Almacenando parámetros para relacionar la presión a la salida con la velocidad a la entrada...')
        resolution = self.cfg.resolution_for_small_input_deviations
        C_in_range = self.cfg.inlet_velocity_range
        jump = (C_in_range[1] - C_in_range[0]) / (resolution - 1)
        C_in = [C_in_range[0] + (k * jump) for k in range(resolution)]

        def velocity_sweeper(T_inlet, p_inlet, n_rpm):
            k = 0
            output_pressures = np.zeros(resolution)
            while k < resolution:
                C_inlet = C_in[k]
                try:
                    if self.cfg.chain_mode:
                        _, p_outlet, _, _ = self.problem_solver(T_in=T_inlet, p_in=p_inlet, n_rpm=n_rpm, C_inx=C_inlet)
                    else:
                        self.problem_solver(T_in=T_inlet, p_in=p_inlet, n_rpm=n_rpm, C_inx=C_inlet)
                        p_outlet = copy.deepcopy(self.vmmr)[-2][1]
                except GasLibraryAdaptedException:
                    output_pressures[k] = np.NAN
                    record.warning('Se ha capturado un error inesperado, se omite esta evaluación.')
                except InnerLoopConvergenceError:
                    output_pressures[k] = np.NAN
                    record.warning('Se ha capturado un error inesperado, se omite esta evaluación.')
                else:
                    output_pressures[k] = p_outlet
                k += 1
            return output_pressures.copy()

        pressures_ref_inputs = velocity_sweeper(self.cfg.T_nominal, self.cfg.p_nominal, self.cfg.n_rpm_nominal)
        pressures_T_in_deviated = velocity_sweeper(self.cfg.T_nominal * (1 + 0.001), self.cfg.p_nominal,
                                                   self.cfg.n_rpm_nominal)
        pressures_p_in_deviated = velocity_sweeper(self.cfg.T_nominal, self.cfg.p_nominal * (1 + 0.0001),
                                                   self.cfg.n_rpm_nominal)
        pressures_rpm_deviated = velocity_sweeper(self.cfg.T_nominal, self.cfg.p_nominal,
                                                  self.cfg.n_rpm_nominal * (1 + 0.001))
        d_pout_d_Tin = (pressures_T_in_deviated - pressures_ref_inputs) / (self.cfg.T_nominal * 0.001)
        d_pout_d_pin = (pressures_p_in_deviated - pressures_ref_inputs) / (self.cfg.p_nominal * 0.0001)
        d_pout_d_rpm = (pressures_rpm_deviated - pressures_ref_inputs) / (self.cfg.n_rpm_nominal * 0.001)

        self.small_input_deviation_data = copy.deepcopy([C_in, pressures_ref_inputs, d_pout_d_Tin, d_pout_d_pin,
                                                         d_pout_d_rpm])

        return

    def problem_solver(self, T_in: float, p_in: float, n_rpm: float, C_inx=None, m_dot=None,
                       p_out=None, C_inx_ref=None) -> None | tuple[float, float, float, float]:
        """Esta función inicia la resolución del problema definido por la geometría configurada y las variables
        requeridas como argumento. (Main method)
                :param p_out: Presión a la salida (Pa).
                :param m_dot: Flujo másico (kg/s).
                :param T_in: Temperatura a la entrada de la turbina (K).
                :param p_in: Presión a la entrada de la turbina (Pa).
                :param C_inx: Velocidad a la entrada de la turbina (Supuesta completamente axial) (m/s).
                :param C_inx_ref: Valor de referencia que debe aproximarse a C_inx.
                :param n_rpm: Velocidad de giro (rpm).
                        :return: Si chain_mode se devuelven los valores a la salida de temperatura (K), presión (Pa),
                                velocidad (m/s) y ángulo del flujo con la dirección axial (degrees), en caso contrario
                                no se devuelve nada."""

        # En esta lista se almacenan variables que se requieren según el modo de funconamiento.
        ps_list: list[float] = []

        relative_error = self.cfg.relative_error
        record.debug('El error relativo establecido en el solver es: %s', relative_error)

        if m_dot is None and C_inx is None:
            if p_out is None:
                raise InputDataError('Debe indicarse un valor de referencia de velocidad a la entrada si se desea '
                                     'fijar la presión a la salida de la turbina.')
            elif C_inx_ref is None and self.small_input_deviation_data is None:
                raise InputDataError('Se debe establecer uno de los parámetros opcionales "p_outlet", '
                                     '"axial_inlet_velocity" ó "mass_flow".')
            elif self.small_input_deviation_data is None:
                if self.C_inx_register is None:
                    C_inx = self.C_inx_register = C_inx_ref
                else:
                    C_inx = self.C_inx_register

        rho_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='d')
        if m_dot is not None:
            C_inx = m_dot / (rho_in * self.cfg.geom['areas'][0])
        elif C_inx is not None:
            m_dot = rho_in * self.cfg.geom['areas'][0] * C_inx

        if self.small_input_deviation_data is not None:
            if len(self.small_input_deviation_data) == 5:
                self.small_input_deviation_data += [T_in, p_in, n_rpm].copy()
            else:
                self.small_input_deviation_data[-3] = T_in
                self.small_input_deviation_data[-2] = p_in
                self.small_input_deviation_data[-1] = n_rpm

        @solver_decorator(self, p_out, self.C_inx_register, self.small_input_deviation_data)
        def inner_solver(var_C_inx=None, solverdec_lastcall=False):
            nonlocal m_dot, C_inx, ps_list

            self.step_iter_mode = self.step_iter_end = False
            self.step_counter = 0
            self.Re_corrector_counter = 0

            if var_C_inx is not None:
                C_inx = self.C_inx_register = var_C_inx
                m_dot = rho_in * self.cfg.geom['areas'][0] * C_inx

            s_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='s')
            h_in = self.prd.get_prop(known_props={'T': T_in, 'p': p_in}, req_prop='h')
            h_0in = h_in + (0.001 * (C_inx**2) / 2)

            ps_list = [T_in, p_in, rho_in, s_in, h_in, h_0in, C_inx]
            for i in range(self.cfg.n_steps):
                list_i = ps_list[i-1] if i > 0 and not self.cfg.chain_mode else ps_list
                args = list_i[0], list_i[1], list_i[6], list_i[3], list_i[4], list_i[5], m_dot, n_rpm, list_i[2]

                if self.cfg.chain_mode:
                    ps_list = self.step_block(*args)
                else:
                    if i == 0:
                        ps_list = [self.step_block(*args)]
                    else:
                        ps_list += [self.step_block(*args)]

                self.Re_corrector_counter = 0
                self.step_counter += 1

            if not self.cfg.chain_mode and solverdec_lastcall:
                # Los subindices A y B indican, resp., los pts. inicio y fin de la turbina.
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
                eta_ss = w_total / w_ss_total
                p_0A, T_0A = self.Zero_pt_calculator(p_in, s_A, h_0A)
                eta_p = log(1 - (eta_ss * (1 - (T_0Bss / T_0A))), 10) / log(T_0Bss / T_0A, 10)
                r0_turbine = p_0A / p_0B
                r_turbine = p_in / p_B
                P_total_ss = m_dot*w_ss_total
                ps_list += [[DELTA_h, w_total, P_total, s_A, s_B, p_0B, T_0B, T_0Bss, h_0Bss, Y_maq, w_ss_total,
                             eta_ss, p_0A, T_0A, eta_p, r_turbine, m_dot, r0_turbine, P_total_ss]]

            self.vmmr = ps_list
            return copy.deepcopy(self.vmmr)

        inner_solver()

        if self.cfg.chain_mode:
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

        @step_decorator(self, corrector_memory)
        def inner_funct(iter_mode=False, iter_end=False, xi_est=None, rho_seed=None, Re_out=None):
            """ Esta función interna se crea para poder comunicar al decorador instancias de la clase.
                                :param rho_seed: Densidad que se emplea como valor semilla en blade_outlet_calculator
                                                 (kg/m^3).
                    :param xi_est: Valor opcional que permite al decorador aplicar recursividad para efectuar la
                                  corrección por dependencia de Reynolds, en el contexto del empleo del modelo AM.
                    :param iter_mode: Permite diferenciar si se está aplicando recursividad para la corrección por
                                 dependencia de Reynolds y así conocer las instrucciones más convenientes.
                    :param iter_end: Permite omitir cálculos innecesarios durante la corrección por dependencia de
                                    Reynolds.
                    :param Re_out: Lista de listas que contiene los valores de Reynolds a la salida de cada corona de
                                   cada escalonamiento.
                            :return: Se devuelven diferentes variables que se requieren según la situación."""

            count = self.step_counter
            self.step_iter_mode = iter_mode
            self.step_iter_end = iter_end

            if rho_seed is None:
                if self.rho_seed_list is not None and len(self.rho_seed_list) >= count+1:
                    rho_seed = self.rho_seed_list[count]
                else:
                    rho_seed = [rho_1*0.9, rho_1*0.75]
                    if count == 0:
                        self.rho_seed_list = []

            A_tpl = (self.cfg.geom['areas'][count * 2],
                     self.cfg.geom['areas'][count * 2 + 1],
                     self.cfg.geom['areas'][count * 2 + 2])

            eta_TT = Re_12 = Re_23 = Re = None

            if not iter_mode and not iter_end:
                pass
            else:
                self.Re_corrector_counter += 1
                if self.Re_corrector_counter > self.cfg.iter_limit_IL:
                    record.error('Se ha alcanzado el límite de iteraciones durante la corrección por el número de '
                                 'Reynolds.')
                    raise InnerLoopConvergenceError
            record.info('Modo de repetición: %s  ...  Llamadas: %s', iter_mode, self.Re_corrector_counter)

            if count > 0 and C_1 != 0.0:
                C_1x = m_dot / (rho_1 * A_tpl[0])
                c_rel = C_1x/C_1
                c_unit = c_rel/fabs(c_rel)
                if fabs(c_rel) > 1:
                    c_rel = c_unit
                alfa_1 = acos(c_rel)
            else:
                C_1x = C_1
                alfa_1 = 0.0
            record.debug('La velocidad axial establecida a la entrada del escalonamiento %s es: %.2f m/s',
                         count + 1, C_1x)

            record.info('Se va a calcular la salida del estátor del escalonamiento número %d',
                        count + 1)

            h_02 = h_01
            a_1 = self.prd.get_sound_speed(T=T_1, p=p_1)
            M_1 = C_1/a_1
            self.ref_values = (T_1, p_1 * 0.95)

            args = ['est', A_tpl[1], alfa_1, h_02, m_dot, s_1, rho_seed[0], M_1, rho_1, C_1, C_1x, T_1]
            if (not iter_mode and not iter_end) or self.cfg.loss_model == 'Aungier':
                outputs = self.blade_row_outlet_calculator(*args, Re_out=Re_out)
                if self.cfg.loss_model == 'Aungier':
                    p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2, xi_est = outputs
                else:
                    p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2, xi_est, Re_12 = outputs
            else:
                outputs = self.blade_row_outlet_calculator(*args, xi=xi_est)
                p_2, h_2, T_2, C_2, rho_2, h_2s, T_2s, C_2x, M_2, alfa_2 = outputs

            s_2 = self.prd.get_prop(known_props={'p': p_2, 'T': T_2}, req_prop='s')

            # En las líneas que siguen se determinan los triángulos de velocidades
            U = self.cfg.geom['Rm'][(count * 2) + 1] * n * 2 * pi / 60
            record.debug('La velocidad tangencial de giro en el radio medio es: %.2f m/s', U)
            C_2u = C_2 * sin(alfa_2)
            record.debug('La velocidad tangencial del flujo a la entrada del rótor es: %.2f m/s', C_2u)
            omega_2u = C_2u - U
            omega_2x = C_2x
            beta_2 = atan(omega_2u / C_2x)
            omega_2 = C_2x / cos(beta_2)

            record.info(' Se va a calcular la salida del rótor del escalonamiento número %d.     ', count + 1)
            h_r3 = h_r2 = h_2 + ((10 ** (-3)) * (omega_2 ** 2) / 2)
            self.ref_values = (T_2, p_2 * 0.95)

            outputs = self.blade_row_outlet_calculator(
                blade='rot', area_b=A_tpl[2], tau_a=beta_2, h_tb=h_r3,
                m_dot=m_dot, s_a=s_2, rho_outer_seed=rho_seed[1], M_a=M_2,
                rho_a=rho_2, C_a=C_2, C_ax=C_2x, T_a=T_2, Re_out=Re_out
            )

            if (not iter_mode and not iter_end) and self.cfg.loss_model != 'Aungier':
                p_3, h_3, T_3, omega_3, rho_3, h_3s, T_3s, C_3x, M_3r, beta_3, xi_rot, Re_23 = outputs
            else:
                p_3, h_3, T_3, omega_3, rho_3, h_3s, T_3s, C_3x, M_3r, beta_3, xi_rot = outputs

            omega_3u = omega_3 * sin(beta_3)
            omega_3x = omega_3 * cos(beta_3)
            C_3u = omega_3u - U
            C_3 = (sqrt((C_3u**2) + (C_3x**2))).real
            if self.cfg.loss_model == 'Aungier':
                Re_23 = self.AU_Re_register = Reynolds(count, rho_3, C_3, T_3, self.cfg, self.prd)
            alfa_3 = asin(C_3u / C_3)

            s_3 = self.prd.get_prop(known_props={'T': T_3, 'p': p_3}, req_prop='s')
            h_03 = h_3 + (0.001 * (C_3**2) / 2)
            local_list_1 = [T_3, p_3, rho_3, s_3, h_3, h_03, C_3, alfa_3]

            if h_02 - h_03 <= 0:
                record.warning('No se extrae energía del fluido.')

            if self.cfg.loss_model == 'Ainley_and_Mathieson':
                # Media aritmética de Re a la entrada y a la salida +- recomendada por AM para la corrección.
                if not iter_mode and not iter_end:
                    Re = (Re_12 + Re_23)/2
                    record.debug('Reynolds a la entrada: %d, Reynolds a la salida: %d, Reynolds promedio del '
                                 'escalonamiento: %d', Re_12, Re_23, Re)
                    if Re < 50_000:
                        record.warning('El número de Reynolds es demasiado bajo.')
            else:
                if Re_23 < 50_000:
                    record.warning('El número de Reynolds es demasiado bajo (%d).', Re_23)

            if not iter_mode and not iter_end:
                if len(self.rho_seed_list) < self.cfg.n_steps:
                    self.rho_seed_list.append([rho_2, rho_3].copy())
                else:
                    self.rho_seed_list[count] = [rho_2, rho_3].copy()

            # Se determina en estas líneas el rendimiento total a total para que sea posible aplicar la corrección:
            if self.cfg.loss_model == 'Ainley_and_Mathieson' and (self.cfg.chain_mode or iter_mode):
                w_esc = h_02 - h_03
                p_03, T_03 = self.Zero_pt_calculator(p_x=p_3, s_x=s_3, h_0x=h_03)
                T_03ss = self.prd.get_prop(known_props={'s': s_1, 'p': p_03}, req_prop={'T': T_03})
                h_03ss = self.prd.get_prop(known_props={'T': T_03ss, 'p': p_03}, req_prop='h')
                Y_esc = h_03 - h_03ss
                eta_TT = w_esc / (w_esc + Y_esc)

            if iter_end:
                if len(self.corrector_seed) < self.cfg.n_steps:
                    if self.cfg.loss_model == 'Ainley_and_Mathieson':
                        self.corrector_seed.append([rho_2, rho_3].copy())
                    else:
                        self.corrector_seed.append(copy.deepcopy([Re_23, [rho_2, rho_3]]))
                else:
                    if self.cfg.loss_model == 'Ainley_and_Mathieson':
                        self.corrector_seed[count] = [rho_2, rho_3].copy()
                    else:
                        self.corrector_seed[count] = copy.deepcopy([Re_23, [rho_2, rho_3]])

            if not self.cfg.chain_mode and (iter_end or not iter_mode):
                Y_est = xi_est * (0.001 * (C_2**2) / 2)
                Y_rot = xi_rot * (0.001 * (omega_3**2) / 2)
                w_esc = h_02 - h_03
                p_03, T_03 = self.Zero_pt_calculator(p_x=p_3, s_x=s_3, h_0x=h_03)
                T_03ss = self.prd.get_prop(known_props={'s': s_1, 'p': p_03}, req_prop={'T': T_03})
                h_03ss = self.prd.get_prop(known_props={'T': T_03ss, 'p': p_03}, req_prop='h')
                Y_esc = h_03 - h_03ss
                eta_TT = w_esc / (w_esc + Y_esc)
                C_1u = sqrt((C_1**2) - (C_1x**2)) if C_1 > C_1x else 0
                a_1 = self.prd.get_sound_speed(T=T_1, p=p_1)
                M_1 = C_1 / a_1
                a_3 = self.prd.get_sound_speed(T=T_3, p=p_3)
                M_3 = C_3 / a_3
                Pot_esc = w_esc * m_dot
                GR = (h_2 - h_3) / w_esc
                PSI = w_esc / (U**2)
                PHI_2 = C_2x / U
                PHI_3 = C_3x / U
                C_2s = sqrt((h_02 - h_2s) * 2000)
                omega_3s = sqrt((h_r3 - h_3s) * 2000)
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
                eta_p_esc = log(1 - (eta_TT * (1 - (T_03ss / T_01))), 10) / log(T_03ss / T_01, 10)
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
        return ll_1.copy()

    def blade_row_outlet_calculator(self, blade: str, area_b: float, tau_a: float, h_tb: float, m_dot: float,
                                    s_a: float, rho_outer_seed: float, M_a: float, rho_a: float, C_a: float,
                                    C_ax: float, T_a: float, xi=None, Re_in=None, Re_out=None):
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
            self.loss_model_object.limit_mssg = [True, True, True]

        rho_b = rho_bp = rho_outer_seed
        M_b = h_b = U_b = h_bs = C_bx = Y_total = Re = Re_AU = pr0_b = Tr0_b = pre_rel_diff = pre_pre_rel_diff = None
        rel_diff, relative_error, geom = 1.0, self.cfg.relative_error, self.cfg.geom
        trend_changes = 0

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
            tau_b = self.loss_model_object.outlet_angle_before_mod[num]
            Re_in = Reynolds(counter*2, rho_a, C_a, T_a, self.cfg, self.prd)
        else:
            tau_b = self.loss_model_object.outlet_angle_before_mod[num]  # Primera estimación
            if Re_out is None:
                record.debug('Se emplea el valor de Reynolds a la entrada como primera estimación.')
                if self.AU_Re_register is None:
                    Re_AU = Reynolds(counter * 2, rho_a, C_a, T_a, self.cfg, self.prd)
                else:
                    Re_AU = self.AU_Re_register
            else:
                Re_AU = self.AU_Re_register = Re_out

        # p: iteración previa .... b: estado que se quiere conocer, a la salida del álabe
        while fabs(rel_diff) > relative_error:
            iter_count += 1
            if rho_b < 0.2:
                record.error('It was not possible to reach convengence. Density is too low.')
                raise InnerLoopConvergenceError

            C_bx = m_dot / (area_b * rho_b)  # C_bx: velocidad axial a la salida
            tau_b_n = None
            tauloop_iter_count = 0

            while tau_b_n is None or fabs((tau_b_n - tau_b)/tau_b) > relative_error:
                tauloop_iter_count += 1
                if tauloop_iter_count > self.cfg.iter_limit_IL or iter_count > self.cfg.iter_limit_IL:
                    record.error('Iteración aboratada, no se cumple el criterio de convergencia.')
                    raise InnerLoopConvergenceError

                if tau_b_n is None:
                    tau_b_n = tau_b
                else:
                    tau_b = tau_b_n

                U_b = C_bx / cos(tau_b)  # U_b: vel. absoluta o relativa ... tau: alfa o beta ... según el caso
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
                    tau_b_n = self.loss_model_object.tau2_corrector(num, M_b)
                    # Se ejecuta el primer bloque a excepción de si es estátor y step_iter_mode/end.
                    if (not self.step_iter_mode and not self.step_iter_end) or blade == 'rot':
                        args = [num, degrees(tau_a), degrees(tau_b), self.step_iter_mode or self.step_iter_end]
                        Y_total = self.loss_model_object.Ainley_and_Mathieson_Loss_Model(*args)
                        xi = Y_total / (1 + (0.5*gamma_b*(M_b**2)))
                    else:
                        Y_total = xi * (1 + (0.5*gamma_b*(M_b**2)))
                        args = [num, degrees(tau_a), degrees(tau_b), True, Y_total]
                        self.loss_model_object.Ainley_and_Mathieson_Loss_Model(*args)

                elif self.cfg.loss_model == 'Aungier':
                    s_b = self.prd.get_prop(known_props={'T': T_b, 'p': p_b}, req_prop='s')
                    pr0_b, Tr0_b = self.Zero_pt_calculator(p_b, s_b, h_tb, p_0x=pr0_b, T_0x=Tr0_b)
                    Y_total, tau_b_n = self.loss_model_object.Aungier_operations(
                        num=num, Min=M_a, Mout=M_b, Re_c=Re_AU, tau_1=tau_a, V_2x=C_bx,
                        V_1x=C_ax, p_2=p_b, pr0_2=pr0_b, d2=rho_b, d1=rho_a, U_2=U_b
                    )
                    xi = Y_total / (1 + (0.5*gamma_b*(M_b**2)))

            h_b = (0.001 * xi * (U_b**2) / 2) + h_bs
            rho_b = self.prd.get_prop({'p': p_b, 'h': h_b}, {'d': rho_b})
            rel_diff = (rho_b - rho_bp) / rho_b

            if pre_rel_diff is not None and pre_pre_rel_diff is not None:
                for sign in [-1, 1]:
                    if sign*pre_pre_rel_diff > sign*pre_rel_diff and sign*pre_rel_diff < sign*rel_diff:
                        if iter_count > 5:
                            trend_changes += 1
            pre_pre_rel_diff = pre_rel_diff
            pre_rel_diff = rel_diff

            rho_bp = rho_b

            iter_string = f'{iter_count}'.center(3)
            record.debug('Iter counter: %s  ...  Density: %.12f kg/m^3  ...  Relative variation: %.12f  ...  '
                         'Trend change counter: %s', iter_string, rho_b, rel_diff, trend_changes)

            if trend_changes >= self.cfg.max_trend_changes:
                record.error('Se ha excedido el valor límite de oscilaciones establecido.')
                raise InnerLoopConvergenceError

        if M_b > 0.5:
            record.warning('Mout %sa la salida superior a 0.5 ... Valor: %.2f',
                           '' if num % 2 == 0 else 'relativo ', M_b)
        else:
            record.debug('Valor del número de Mach %sa la salida: %.2f',
                         '' if num % 2 == 0 else 'relativo ', M_b)

        if (self.step_iter_mode or self.step_iter_end) and self.cfg.loss_model == 'Ainley_and_Mathieson':
            record.debug('La relación entre el valor actual y el inicial de las pérdidas adimensionales de presión '
                         'en ambas coronas del escalonamiento %s es: %.3f\n  ...   ',
                         1 + (num//2), Y_total / self.loss_model_object.Y_t_preiter[num])
        record.debug('Incidencia: %.2f°  ...  tau_in: %.2f°  ...  Ángulo del B.A.: %.2f°',
                     degrees(tau_a) - degrees(alfap_1), degrees(tau_a), degrees(alfap_1))
        record.debug('Desviación: %.2f°  ...  tau_out: %.2f°  ...  Ángulo del B.S.: %.2f°',
                     degrees(tau_b) - degrees(alfap_2), degrees(tau_b), degrees(alfap_2))

        if self.cfg.loss_model == 'Ainley_and_Mathieson':
            if not self.step_iter_mode and not self.step_iter_end:
                Yp = self.loss_model_object.Yp_preiter[num]
            else:
                Yp = self.loss_model_object.Yp_iter_mode
        else:
            Yp = self.loss_model_object.Yp_iter_mode
        record.debug('Pérdidas:  ...  Y_total: %.4f  ...  Yp: %.4f   ',
                     Y_total, Yp)

        self.first_seeds_boc = [T_b, T_bs, p_b].copy()
        return_vars = [p_b, h_b, T_b, U_b, rho_b, h_bs, T_bs, C_bx, M_b, tau_b]

        if blade == 'est' and self.cfg.loss_model != 'Aungier':
            if not self.step_iter_mode and not self.step_iter_end:
                return *return_vars, xi, Re
            else:
                return return_vars
        else:
            if not self.step_iter_mode and not self.step_iter_end and self.cfg.loss_model != 'Aungier':
                return *return_vars, xi, Re
            else:
                return *return_vars, xi

    def Zero_pt_calculator(self, p_x: float, s_x: float, h_0x: float, p_0x=None, T_0x=None):
        """ Este método es para determinar presiones y temperaturas de remanso.
                        :param p_0x: Estimación inicial del valor de presión de remanso (Pa).
                :param T_0x: Estimación inicial del valor de temperatura de remanso (K).
                :param p_x: Presión en la sección x (Pa).
                :param s_x: Entropía en la sección x (kJ/kgK).
                :param h_0x: Entalpía total en la sección x (kJ/kg).
                        :return: Devuelve la presión total (Pa) y la temperatura total (K) en una sección x. """

        if p_0x is None and T_0x is None:
            p_0x, T_0x = p_x * 1.1, self.ref_values[0] * 1.1
        end, relative_error = False, self.cfg.relative_error

        while not end:
            T_0x = self.prd.get_prop(known_props={'h': h_0x, 'p': p_0x}, req_prop={'T': T_0x})
            if self.cfg.ideal_gas:
                p_0x = self.prd.get_prop(known_props={'T': T_0x, 's': s_x}, req_prop={'p': p_0x})
                end = True
            else:
                p_0x_iter = self.prd.get_prop(known_props={'T': T_0x, 's': s_x}, req_prop={'p': p_0x})
                if fabs(p_0x - p_0x_iter) / p_0x < relative_error:
                    end = True
                p_0x = p_0x_iter

        return p_0x, T_0x


def main():
    chain_mode = False
    settings = config_class(relative_error=1E-9, n_steps=1, jump=2, loss_model='Aungier',
                            ideal_gas=True, chain_mode=chain_mode, iter_limit_IL=1200)

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

    settings.set_geometry(B_A_est=0, theta_est=70, B_A_rot=36, theta_rot=100, areas=areas, cuerda=chord,
                          radio_medio=Rm, e=e_param, o=blade_opening, s=pitch, H=heights, b_z=chord_proj_z,
                          t_max=t_max, r_r=0.002, r_c=0.001, t_e=t_e, k=tip_clearance, delta=tip_clearance,
                          roughness_ptv=blade_roughness_peak_to_valley, holgura_radial=False,
                          gauge_adimensional_position=0.2)

    gas_model = gas_model_to_solver(thermod_mode="ig")
    solver = solver_object(settings, gas_model)

    if chain_mode:
        output = solver.problem_solver(T_in=900, p_in=170_000, n_rpm=17_000, p_out=100_000, C_inx_ref=130)
        T_salida, p_salida, C_salida, alfa_salida = output
        print(' T_out =', T_salida, '\n', 'P_out =', p_salida,
              '\n', 'C_out =', C_salida, '\n', 'alfa_out =', alfa_salida)

    else:
        solver.problem_solver(T_in=900, p_in=170_000, n_rpm=17_000, p_out=100_000, C_inx_ref=130)


if __name__ == '__main__':
    main()
