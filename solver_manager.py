"""
Se almacenan y procesan los datos haciendo uso de las librerías Pickle y Pandas.
"""
import pickle

from axial_turb_solver import *
import pandas as pd
import matplotlib.pyplot as plt

# https://conocepython.blogspot.com/2019/05/ (este es interesante)
# https://www.youtube.com/watch?v=eWrTSBIQess (tutorial de Pickle útil)
# http://josearcosaneas.github.io/python/serializaci%C3%B3n/persistencia/2016/12/26/serializacion-persistencia.html

s_keys = 'T_3, p_3, rho_3, s_3, h_3, h0_3, C_3, alfa_3, Cx_3, Cu_3, T0_3, p0_3, h_r3, h_3s, T_3s, omega_3, omegax_3, ' \
         'omegau_3, beta_3, M_3r, M_3, PHI_3, T_2, p_2, rho_2, s_2, h_2, h0_2, C_2, Cx_2, Cu_2, alfa_2, T0_2, p0_2, ' \
         'h_r2, h_2s, T_2s, omega_2, omegax_2, omegau_2, beta_2, M_2, PHI_2, T_1, p_1, rho_1, s_1, h_1, h0_1, C_1, ' \
         'Cx_1, Cu_1, alfa_1, T0_1, p0_1, M_1, Y_est, xi_est, Y_rot, xi_rot, w_esc, Pot_esc, eta_p_esc, r_esc, GR, ' \
         'PSI, speed_loss_coeff_E, speed_loss_coeff_R, DELTA_h_esc, eta_TT, eta_TE, Y_esc, w_s_esc, w_ss_esc, C_2s, ' \
         'T0_2s, h0_2s, omega_3s, T_3ss, T0_3s, T0_3ss, h_3ss, h0_3s, h0_3ss, U'

s_units = ' (K), (Pa), (kg/m^3), (kJ/kgK), (kJ/kg), (kJ/kg), (m/s), (°), (m/s), (m/s), (K), (Pa), (kJ/kg), (kJ/kg), ' \
          '(K), (m/s), (m/s), (m/s), (°), (-), (-), (-), (K), (Pa), (kg/m^3), (kJ/kgK), (kJ/kg), (kJ/kg), (m/s), ' \
          '(m/s), (m/s), (°), (K), (Pa), (kJ/kg), (kJ/kg), (K), (m/s), (m/s), (m/s), (°), (-), (-), (K), (Pa), ' \
          '(kg/m^3), (kJ/kgK), (kJ/kg), (kJ/kg), (m/s), (m/s), (m/s), (°), (K), (Pa), (-), (kJ/kg), (-), (kJ/kg), ' \
          '(-), (kJ/kg), (kW), (-), (-), (-), (-), (-), (-), (kJ/kg), (-), (-), (kJ/kg), (kJ/kg), (kJ/kg), (m/s), ' \
          '(K), (kJ/kg), (m/s), (K), (K), (K), (kJ/kg), (kJ/kg), (kJ/kg), (m/s)'

t_keys = 'DELTA_h, w_total, P_total, s_A, s_B, p0_B, T0_B, T0_Bss, h0_Bss, Y_maq, w_ss_total, eta_ss, p0_A, T0_A, ' \
         'eta_p, r_turbine, m_dot, r0_turbine, P_total_ss'

t_units = '(kJ/kg), (kJ/kg), (kW), (kJ/kgK), (kJ/kgK), (Pa), (K), (K), (kJ/kg), (kJ/kg), (kJ/kg), (-), (Pa), (K), ' \
          '(-), (-), (kg/s), (-), (kW)'

tpl_s_keys, tpl_t_keys = tuple(s_keys.split(', ')), tuple(t_keys.split(', '))
tpl_s_units, tpl_t_units = tuple(s_units.split(',')), tuple(t_units.split(','))

global_list_a = ['M', 'C', 'Cx', 'Cu', 'omega', 'omegax', 'omegau', 'alfa', 'beta', 'h', 'h0', 'T', 'T0',
                 'p', 'p0', 'rho', ]
global_list_b = ['GR', 'w_esc', 'w_s_esc', 'w_ss_esc', 'eta_TT', 'eta_TE', 'Y_est', 'Y_rot', 'U', ]
global_list_c = ['w_total', 'w_ss_total', 'eta_ss', 'P_total', 'r_turbine', 'm_dot', 'r0_turbine', 'P_total_ss', ]


# https://www.freecodecamp.org/news/with-open-in-python-with-statement-syntax-example/ (funcionamiento de with open as)
def solver_data_saver(file: str, process_object: solver_object) -> None:
    """ Se almacena la información necesaria para poder restablecer la situación del programa en la siguiente
    ejecución.
            :param file: Nombre de los archivos .db generados.
            :param process_object: Objeto proceso (Solver).
                    :return: No se devuelve nada. """

    prd = process_object.prd
    product_attr = {'tm': prd.thermod_mode, 'tol': prd.relative_error,
                    'C': prd.C_atoms, 'H': prd.H_atoms, 'N': prd.air_excess}
    process_object.prd = None  # Esto se hace así para evitar un error que surge

    with open(file, 'wb') as solver_pickle:
        pickle.dump(process_object, solver_pickle)
        pickle.dump(product_attr, solver_pickle)

    return


def solver_data_reader(file: str) -> solver_object:
    """ Se restablece el estado del programa en la ejecución en que fue efectuada la función 'solver_data_saver',
    reestableciendo la configuración y el estado del solver.
            :param file: Nombre de los archivos generados.
                    :return: Se devuelve el solver ya configurado. """

    with open(file, 'rb') as obj_pickle:
        solver_obj: solver_object = pickle.load(obj_pickle)
        prd_attr = pickle.load(obj_pickle)

    solver_obj.prd = gas_model_to_solver(thermod_mode=prd_attr['tm'], relative_error=prd_attr['tol'],
                                         C_atoms=prd_attr['C'], H_atoms=prd_attr['H'], air_excess=prd_attr['N'])

    return solver_obj


def data_to_df(process_object: solver_object, req_vars=None) -> [pd.DataFrame | None, pd.DataFrame | None,
                                                                 pd.DataFrame | None]:
    """ Se trata la información que se obtiene del solver empleando la librería 'pandas'. Se posibilita modificar el
    contenido almacenado, en caso de que no se especifique se establecen unas por defecto.
            :param process_object: Objeto 'solver'.
            :param req_vars: Identificador de las variables que se requieren. Parámetro opcional.
                    :return: Devuelve los dataframes generados. """

    total_vars, dict_df = process_object.vmmr, dict()
    df_a = df_b = df_c = None

    lista_a = global_list_a
    lista_b = global_list_b
    lista_c = global_list_c

    if req_vars is None:
        pass
    else:
        lista_a = [item for item in req_vars if item in lista_a]
        lista_b = [item for item in req_vars if item in tpl_s_keys and item not in lista_a]
        lista_c = [item for item in req_vars if item in tpl_t_keys]

    def default_a(step_count: int) -> pd.DataFrame:
        for k_var in lista_a:
            v_list, index_mmr = list(), int()
            for final in ['_1', '_2', '_2s', '_3', '_3s', '_3ss']:
                item_id = k_var + final
                not_a_number = bool()
                if final in ['_2s', '_3s', '_3ss', ]:
                    not_a_number = False if k_var in ['h', 'h0', 'T', 'T0', ] else True
                if item_id in ['C_2s', 'omega_3s']:
                    not_a_number = False
                if k_var in ['omega', 'omegax', 'omegau', 'beta', ] and final == '_1':
                    not_a_number = True
                v_list.append(total_vars[step_count][tpl_s_keys.index(item_id)] if not not_a_number else np.NAN)
                if k_var in ['alfa', 'beta', ]:
                    v_list[-1] = degrees(v_list[-1])
                if not not_a_number:
                    index_mmr = tpl_s_keys.index(item_id)
            dict_df[k_var + tpl_s_units[index_mmr]] = v_list.copy()

        pt_list_a = list()
        for j in ['1', '2', '2s', '3', '3s', '3ss']:
            pt_list_a.append(f'Step_{step_count + 1}_pt_' + j)
        local_df = pd.DataFrame(data=dict_df, index=pt_list_a) if len(lista_a) != 0 else None

        return local_df

    def default_b(step_count: int) -> pd.DataFrame:
        dict_serie = dict()
        for item in lista_b:
            dict_serie[item + tpl_s_units[tpl_s_keys.index(item)]] = total_vars[step_count][tpl_s_keys.index(item)]
        local_serie = pd.Series(data=dict_serie) if len(lista_b) != 0 else None
        local_df = local_serie.to_frame() if len(lista_b) != 0 else None
        return local_df

    def default_c() -> pd.DataFrame:
        dict_serie = dict()
        for item in lista_c:
            dict_serie[item + tpl_t_units[tpl_t_keys.index(item)]] = total_vars[-1][tpl_t_keys.index(item)]
        local_serie = pd.Series(data=dict_serie) if len(lista_c) != 0 else None
        local_df = local_serie.to_frame(name='Turbina') if len(lista_c) != 0 else None
        return local_df

    for n in range(process_object.cfg.n_steps):
        if df_a is None and df_b is None and df_c is None:
            df_a = default_a(n)
            df_b = default_b(n)
            df_b = df_b.T if df_b is not None else None
            df_c = default_c()
            df_c = df_c.T if df_c is not None else None
        else:
            df_a = pd.concat([df_a, default_a(n)], axis=0) if df_a is not None else None
            df_b = pd.concat([df_b, default_b(n).T], axis=0) if df_b is not None else None

    if df_b is not None:
        df_b['Steps'] = [i for i in range(process_object.cfg.n_steps + 1) if i != 0]
        df_b.set_index('Steps', inplace=True)

    return df_a, df_b, df_c


def problem_data_viewer(solver: solver_object, req_vars=None) -> None:
    """ Se visualiza el dataframe generado. """

    df_a, df_b, df_c = data_to_df(solver, req_vars)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None,
                           'display.width', 1000, 'display.precision', 4):
        if df_a is not None:
            print(df_a, '\n')
        if df_b is not None:
            print(df_b, '\n')
        if df_c is not None:
            print(df_c, '\n')

    return


def var_sweeping(solver: solver_object, n_rpm, T_in: float | list, p_in, var_to_sweep: str, C_inx=None,
                 m_dot=None, p_out=None, C_inx_ref=None, sweep_resolution=200, req_vars: list = None):
    """ El rango debe ser creciente. """
    t_1 = time()
    k = 0
    sweeping_data = {}
    lista_df_a, lista_df_b, lista_df_c = [], [], []

    if var_to_sweep == 'T_inlet':
        sweeping_data[var_to_sweep] = T_in
        sweeping_data['units'] = 'K'
    elif var_to_sweep == 'p_inlet':
        sweeping_data[var_to_sweep] = p_in
        sweeping_data['units'] = 'Pa'
    elif var_to_sweep == 'rpm':
        sweeping_data[var_to_sweep] = n_rpm
        sweeping_data['units'] = 'rpm'
    elif var_to_sweep == 'p_outlet':
        sweeping_data[var_to_sweep] = p_out
        sweeping_data['units'] = 'Pa'
    elif var_to_sweep == 'mass_flow':
        sweeping_data[var_to_sweep] = m_dot
        sweeping_data['units'] = 'kg'
    elif var_to_sweep == 'axial_inlet_velocity':
        sweeping_data[var_to_sweep] = C_inx
        sweeping_data['units'] = 'm/s'

    def set_value(value):
        nonlocal T_in, p_in, n_rpm, p_out, m_dot, C_inx
        if var_to_sweep == 'T_inlet':
            T_in = value
        elif var_to_sweep == 'p_inlet':
            p_in = value
        elif var_to_sweep == 'rpm':
            n_rpm = value
        elif var_to_sweep == 'p_outlet':
            p_out = value
        elif var_to_sweep == 'mass_flow':
            m_dot = value
        elif var_to_sweep == 'axial_inlet_velocity':
            C_inx = value

    jump = (sweeping_data[var_to_sweep][1] - sweeping_data[var_to_sweep][0]) / (sweep_resolution - 1)
    while k < sweep_resolution:

        value_k = sweeping_data[var_to_sweep][0] + (k * jump)
        set_value(value_k)

        try:
            if not solver.cfg.preloading_for_small_input_deviations:
                solver.problem_solver(T_in=T_in, p_in=p_in, n_rpm=n_rpm,
                                      m_dot=m_dot, C_inx=C_inx, p_out=p_out,
                                      C_inx_ref=C_inx_ref)
            else:
                solver.problem_solver(T_in=T_in, p_in=p_in, n_rpm=n_rpm,
                                      m_dot=m_dot, C_inx=C_inx, p_out=p_out)
        except GasLibraryAdaptedException:
            record.error('An error has been handled during variable sweeping operation. Evaluated point '
                         'has been omited.')
        except InnerLoopConvergenceError:
            record.error('An error has been handled during variable sweeping operation. Evaluated point '
                         'has been omited.')
        except OuterLoopConvergenceError:
            record.error('An error has been handled during variable sweeping operation. Evaluated point '
                         'has been omited.')
        else:
            df_a, df_b, df_c = data_to_df(solver, req_vars)
            lista_df_a.append(copy.deepcopy(df_a))
            lista_df_b.append(copy.deepcopy(df_b))
            lista_df_c.append(copy.deepcopy(df_c))
        record.info('Barrido de la variable completo en un %d%s', 100*(k+1)/sweep_resolution, '%')
        k += 1

    df_a_packg = pd.concat(
        [*lista_df_a], keys=[f'{v}' for v in range(sweep_resolution)], names=['Aux_Index', 'Spec_Index'])
    df_b_packg = pd.concat(
        [*lista_df_b], keys=[f'{v}' for v in range(sweep_resolution)], names=['Aux_Index', 'Spec_Index'])
    df_c_packg = pd.concat(
        [*lista_df_c], keys=[f'{v}' for v in range(sweep_resolution)], names=['Aux_Index', 'Spec_Index'])

    t_2 = (time() - t_1).__round__(0)
    m, s = divmod(t_2, 60)
    h, m = divmod(m, 60)
    record.info('El tiempo de cálculo durante todo el barrido ha sido: %s horas, %s minutos y %s segundos.',
                int(h), int(m), int(s))

    return df_a_packg, df_b_packg, df_c_packg


def txt_reader():
    with open('turbine_data_template_v1.txt') as file:
        for line in file:
            declaration = ''
            for char in line.strip():
                if char != ';':
                    if char not in [' ', '#']:
                        if char not in ['=', ',', ':']:
                            declaration += char
                        elif char == ',':
                            declaration += ', '
                        elif char == '=':
                            declaration += ' = '
                        else:
                            declaration += ': '
                    elif char == '#':
                        break
                else:
                    if '$' in declaration:
                        splitted_str = declaration.split(sep='=')
                        splitted_str[1] = splitted_str[1].replace('for', ' for ').replace('in', ' in ').replace('$', '')
                        exec(splitted_str[0] + ' = ' + splitted_str[1])
                    else:
                        exec(declaration)
                    declaration = ''
        return locals()


def main():
    data_dictionary = txt_reader()
    settings = None

    def aux_reading_operations():
        nonlocal settings
        try:
            settings = config_class(
                relative_error=data_dictionary['relative_error'],
                ideal_gas=data_dictionary['ideal_gas'],
                n_steps=data_dictionary['n_steps'],
                jump=data_dictionary['jump'],
                chain_mode=data_dictionary['chain_mode'],
                loss_model=data_dictionary['loss_model'],
                iter_limit_IL=data_dictionary['iter_limit_at_inner_loops'],
                iter_limit_OL=data_dictionary['iter_limit_at_outer_loops'],
                max_trend_changes=data_dictionary['max_trend_changes'],
                T_nominal=data_dictionary['T_nominal'],
                preloading_for_small_input_deviations=data_dictionary['preloading_for_small_input_deviations'],
                p_nominal=data_dictionary['p_nominal'],
                resolution_for_small_input_deviations=data_dictionary['resolution_for_small_input_deviations'],
                inlet_velocity_range=data_dictionary['inlet_velocity_range'],
                n_rpm_nominal=data_dictionary['n_rpm_nominal'],
                )
            LE_stator = data_dictionary['stator_leading_edge_angle']
            LE_rotor = data_dictionary['rotor_leading_edge_angle']
            TE_stator = data_dictionary['stator_trailing_edge_angle']
            TE_rotor = data_dictionary['rotor_trailing_edge_angle']
            theta_stator = data_dictionary['stator_blade_curvature']
            theta_rotor = data_dictionary['rotor_blade_curvature']
            hub_radius = data_dictionary['hub_radius']
            tip_radius = data_dictionary['tip_radius']
            Rm = data_dictionary['mean_radius']
            heights = data_dictionary['heights']
            areas = data_dictionary['areas']
            chord = data_dictionary['chord']
            t_max = data_dictionary['maximum_thickness']
            pitch = data_dictionary['pitch']
            t_e = data_dictionary['outlet_thickness']
            blade_opening = data_dictionary['blade_opening']
            e_param = data_dictionary['blade_mean_radius_of_curvature']
            tip_clearance = data_dictionary['tip_clearance']
            chord_proj_z = data_dictionary['chord_z']
            wire_diameter = data_dictionary['wire_diameter']
            lashing_wires = data_dictionary['lashing_wires']
            holgura_radial = data_dictionary['radial_clearance']
            blade_roughness_peak_to_valley = data_dictionary['blade_roughness_peak_to_valley']
            design_factor = data_dictionary['design_factor']
        except NameError:
            raise InputDataError('Non-valid text file, please, stick to the template.')

        settings.set_geometry(B_A_est=LE_stator, theta_est=theta_stator, B_A_rot=LE_rotor, theta_rot=theta_rotor,
                              H=heights, B_S_est=TE_stator, B_S_rot=TE_rotor, areas=areas, cuerda=chord, radio_medio=Rm,
                              e=e_param, b_z=chord_proj_z, o=blade_opening, s=pitch, t_max=t_max, r_h=hub_radius,
                              delta=tip_clearance, r_t=tip_radius, k=tip_clearance, t_e=t_e,
                              roughness_ptv=blade_roughness_peak_to_valley, lashing_wires=lashing_wires,
                              wire_diameter=wire_diameter, holgura_radial=holgura_radial, design_factor=design_factor)
        return settings

    mode = data_dictionary['mode']

    if mode == 'process_nominal_values':
        settings = aux_reading_operations()
        gas_model = gas_model_to_solver(thermod_mode=data_dictionary.get('thermo_mode_in_gas_model.py', 'ig'))
        solver = solver_object(settings, gas_model)
        solver_data_saver('process_object.pkl', solver)

    elif mode == 'solve':
        try:
            try:
                solver = solver_data_reader('process_object.pkl')
            except FileNotFoundError:
                settings = aux_reading_operations()
                gas_model = gas_model_to_solver(thermod_mode=data_dictionary.get('thermod_mode_in_gas_model_module',
                                                                                 'ig'))
                solver = solver_object(settings, gas_model)
            if data_dictionary['chain_mode']:
                output = solver.problem_solver(
                    T_in=data_dictionary['T_inlet'],
                    p_in=data_dictionary['p_inlet'],
                    n_rpm=data_dictionary['rpm'],
                    p_out=data_dictionary['p_outlet'],
                    m_dot=data_dictionary['mass_flow'],
                    C_inx=data_dictionary['axial_inlet_velocity'],
                    C_inx_ref=data_dictionary['reference_inlet_velocity'],
                )
                T_salida, p_salida, C_salida, alfa_out = output
                print(' T_out', T_salida, '\n', 'p_out', p_salida, '\n', 'C_out', C_salida, '\n', 'alfa_out', alfa_out)
            else:
                solver.problem_solver(
                    T_in=data_dictionary['T_inlet'],
                    p_in=data_dictionary['p_inlet'],
                    n_rpm=data_dictionary['rpm'],
                    p_out=data_dictionary['p_outlet'],
                    m_dot=data_dictionary['mass_flow'],
                    C_inx=data_dictionary['axial_inlet_velocity'],
                    C_inx_ref=data_dictionary['reference_inlet_velocity'],
                )
                solver_data_saver('process_object.pkl', solver)
        except NameError:
            raise InputDataError('Non-valid text file, please, stick to the template.')

    elif mode == 'display_values':
        solver = solver_data_reader('process_object.pkl')
        problem_data_viewer(solver)

    elif mode == 'sweep_variable':
        settings = aux_reading_operations()
        gas_model = gas_model_to_solver(thermod_mode=data_dictionary.get('thermod_mode_in_gas_model_module', 'ig'))
        solver = solver_object(settings, gas_model)
        try:
            a_filename, b_filename, c_filename = [root + data_dictionary['csv_filename_extension'] for root in
                                                  ['df_a_', 'df_b_', 'df_c_']]
            df_a, df_b, df_c = var_sweeping(
                solver,
                T_in=data_dictionary['T_inlet'],
                p_in=data_dictionary['p_inlet'],
                n_rpm=data_dictionary['rpm'],
                m_dot=data_dictionary['mass_flow'],
                C_inx=data_dictionary['axial_inlet_velocity'],
                C_inx_ref=data_dictionary['reference_inlet_velocity'],
                p_out=data_dictionary['p_outlet'],
                req_vars=data_dictionary['req_vars'],
                var_to_sweep=data_dictionary['var_to_sweep'],
                sweep_resolution=data_dictionary['sweep_resolution'])
        except NameError:
            raise InputDataError('Non-valid text file, please, stick to the template.')
        df_a.to_csv(a_filename)
        df_b.to_csv(b_filename)
        df_c.to_csv(c_filename)
        solver_data_saver('process_object.pkl', solver)

    elif mode == 'display_graphs_with_sweep_data':
        x_label_name_and_units = x_label_name = None
        dep_ids_dict: dict = {}
        try:
            logic_limit_for_independent_variable: bool = data_dictionary['logic_limit_for_independent_variable']
            req_vars = data_dictionary['req_vars']
            plot_together = data_dictionary['plot_together']
            WtE: dict = data_dictionary['where_to_evaluate']
            independent_var = data_dictionary['independent_variable']
            dependent_vars = data_dictionary['dependent_variables']
            a_filename, b_filename, c_filename = [root + data_dictionary['csv_filename_extension'] for root in
                                                  ['df_a_', 'df_b_', 'df_c_']]
        except NameError:
            raise InputDataError('Non-valid text file, please, stick to the template.')

        lista_a = global_list_a
        lista_b = global_list_b
        lista_c = global_list_c

        if req_vars is None:
            pass
        else:
            lista_a = [item for item in req_vars if item in lista_a]
            lista_b = [item for item in req_vars if item in tpl_s_keys and item not in lista_a]
            lista_c = [item for item in req_vars if item in tpl_t_keys]

        custom_df = pd.DataFrame()

        df_a = pd.read_csv(a_filename, index_col='Aux_Index')
        df_b = pd.read_csv(b_filename, index_col='Aux_Index')
        df_c = pd.read_csv(c_filename, index_col='Aux_Index')

        variable_list = dependent_vars.copy()
        variable_list.append(independent_var)
        for item in variable_list:
            if item in lista_a or item in lista_b:
                if WtE is None or item not in WtE or 'step' not in WtE[item] or \
                        ('point' not in WtE[item] and item in lista_a):
                    raise InputDataError('It is needed complementary information for any variable at the text file to '
                                         'process saved data to be plotted. Please, check items that need to be added '
                                         'to "where_to_evaluate" dictionary.')
            elif item in lista_c:
                units_extension = tpl_t_units[tpl_t_keys.index(item)]
                custom_df[item] = df_c[item + units_extension]
                if item == independent_var:
                    x_label_name = item
                    x_label_name_and_units = x_label_name + units_extension
                else:
                    dep_ids_dict[item] = [item]
                    dep_ids_dict[item].append(item + units_extension)  # For y-labels when not multiplotting.
                    dep_ids_dict[item].append(item)  # This one is for leyends when needed.
                    dep_ids_dict[item].append(item + units_extension)  # For y-labels when multiplotting.
            else:
                sentence = f'{item} is not considered as a possible input. Use req_vars for considering variables ' \
                           f'listed in the handbook.'
                raise InputDataError(sentence)

        if WtE is not None:
            for key in WtE.keys():
                if key in lista_a:
                    step_point = WtE[key]['point']
                    step_id = WtE[key]['step']
                    if not isinstance(step_id, list) and not isinstance(step_point, list):
                        step_point, step_id = [step_point], [step_id]
                    for num, st_id in enumerate(step_id):
                        units_extension = tpl_s_units[tpl_s_keys.index(f'{key}_{step_point[num]}')]
                        old_var_id = key + units_extension
                        var_id = f'{key} (step {st_id}, point {step_point[num]})'
                        custom_df[var_id] = df_a[
                            df_a['Spec_Index'] == f'Step_{st_id}_pt_{step_point[num]}'][old_var_id]
                        if key == independent_var and num == 0:
                            custom_df.set_index(var_id, inplace=True)  # Index will be plotted as independent variable.
                            x_label_name = var_id
                            x_label_name_and_units = x_label_name + units_extension
                        else:
                            dep_ids_dict[var_id] = [key]
                            dep_ids_dict[var_id].append(var_id + units_extension)  # For y-labels when plotting.
                            dep_ids_dict[var_id].append(var_id)  # This one is for leyends when needed.
                            dep_ids_dict[var_id].append(key + units_extension)  # For y-labels when multiplotting.
                elif key in lista_b:
                    step_id = WtE[key]['step']
                    units_extension = tpl_s_units[tpl_s_keys.index(key)]
                    old_var_id = key + units_extension
                    if not isinstance(step_id, list):
                        step_id = [step_id]
                    for num, st_id in enumerate(step_id):
                        var_id = f'{key} (step {st_id})'
                        custom_df[var_id] = df_b[df_b['Spec_Index'] == st_id][old_var_id]
                        if key == independent_var and num == 0:
                            x_label_name = var_id
                            x_label_name_and_units = x_label_name + units_extension
                        else:
                            dep_ids_dict[var_id] = [key]
                            dep_ids_dict[var_id].append(var_id + units_extension)  # This one is for y-labels (plot).
                            dep_ids_dict[var_id].append(var_id)  # This one is for leyends when needed.
                            dep_ids_dict[var_id].append(key + units_extension)  # For y-labels (multiplot).

        # This next section is needed after declaring all columns of custom_df in order to set the index for
        # custom_df and keeping the integrity of every value of its columns.
        indep_id = None
        if independent_var in lista_a:
            step_point = WtE[independent_var]['point']
            step_id = WtE[independent_var]['step']
            if not isinstance(step_id, list) and not isinstance(step_point, list):
                step_point, step_id = [step_point], [step_id]
            indep_id = f'{independent_var} (step {step_id[0]}, point {step_point[0]})'
        elif independent_var in lista_b:
            step_id = WtE[independent_var]['step']
            if not isinstance(step_id, list):
                step_id = [step_id]
            indep_id = f'{independent_var} (step {step_id[0]})'
        elif independent_var in lista_c:
            indep_id = independent_var
        custom_df.set_index(indep_id, inplace=True)  # Index will be plotted as independent variable.

        IV_limits_refs = DV_limits = None

        # The following function stablishes the range of values of the independent variable to be displayed
        # using as condition that resulting exchange of energy is given from the gas to the turbine.
        def x_axis_limits_algorithm():
            def IV_limits_finder():
                nonlocal IV_limits_refs
                IV_limits_refs = []
                for i in range(custom_df.shape[0]):
                    if i == 0 and df_c['w_total (kJ/kg)'][0] > 0:
                        IV_limits_refs.append(i)
                    elif i > 0 and df_c['w_total (kJ/kg)'][i]*df_c['w_total (kJ/kg)'][i-1] < 0:
                        if df_c['w_total (kJ/kg)'][i] > 0:
                            IV_limits_refs.append(i)
                        else:
                            IV_limits_refs.append(i-1)
                    if i == custom_df.shape[0]-1 and df_c['w_total (kJ/kg)'][i]*df_c['w_total (kJ/kg)'][i-1] > 0 and \
                            len(IV_limits_refs) == 1:
                        IV_limits_refs.append(i)
                if IV_limits_refs is None:
                    raise InputDataError('No data meets the condition. There is no value to be displayed.')
                else:
                    p1, p2 = IV_limits_refs[0], IV_limits_refs[1]
                    if custom_df.index[p2] < custom_df.index[p1]:
                        IV_limits_refs[0], IV_limits_refs[1] = IV_limits_refs[1], IV_limits_refs[0]
                return
            if not isinstance(IV_limits_refs, list):
                IV_limits_finder()
            plt.xlim(custom_df.index[IV_limits_refs[0]], custom_df.index[IV_limits_refs[1]])
            return

        # The following function adapts the range of values of the axis for the dependent variable to the .
        # The function "x_axis_limits_algorithm()" is expected to be executed before.
        def y_axis_limits_algorithm(DV_identifiers_list: list[str]):
            nonlocal DV_limits
            DV_limits = []

            def DV_limits_finder():
                nonlocal DV_limits
                if isinstance(IV_limits_refs, list):
                    for DV_ID in DV_identifiers_list:
                        for i in range(int(IV_limits_refs[0]), int(IV_limits_refs[1]+1)):
                            if len(DV_limits) < 2:
                                DV_limits.append(custom_df[DV_ID][i])
                            else:
                                if custom_df[DV_ID][i] < DV_limits[0]:
                                    DV_limits[0] = custom_df[DV_ID][i]
                                if custom_df[DV_ID][i] > DV_limits[1]:
                                    DV_limits[1] = custom_df[DV_ID][i]
            DV_limits_finder()
            diff = 0.5*(DV_limits[1]-DV_limits[0])
            plt.ylim(DV_limits[0]-diff, DV_limits[1]+diff)
        # Section for plotting only one independent variable at the time.
        for dep_var_id in dep_ids_dict:
            skip = False
            for lista in plot_together:
                if dep_ids_dict[dep_var_id][0] in lista:
                    skip = True  # Skipping this to retake it later at multiplot section.
            if not skip:
                plt.plot(custom_df[dep_var_id])
                if logic_limit_for_independent_variable:
                    x_axis_limits_algorithm()
                    y_axis_limits_algorithm([dep_var_id, ])
                title_str = dep_ids_dict[dep_var_id][0] + '   -   ' + x_label_name
                plt.title(title_str)
                plt.xlabel(x_label_name_and_units)
                plt.ylabel(dep_ids_dict[dep_var_id][1])
                plt.minorticks_on()
                plt.grid(which='both')
                plt.show()
        # Multiplotting section.
        for lista in plot_together:
            multiplot_legend_dict = {}
            y_label_ref_u = []
            for item in lista:
                for dep_var_id in dep_ids_dict:
                    if dep_ids_dict[dep_var_id][0] == item:
                        multiplot_legend_dict[dep_var_id] = dep_ids_dict[dep_var_id][2]
                        if dep_ids_dict[dep_var_id][3] not in y_label_ref_u:
                            y_label_ref_u.append(dep_ids_dict[dep_var_id][3])
            y_label_ref = lista
            y_label_ref_as_str = y_label_ref_u_as_str = ''
            for number, ref in enumerate(y_label_ref):
                if number < len(y_label_ref)-1:
                    y_label_ref_u_as_str += y_label_ref_u[number] + ',  '
                    y_label_ref_as_str += ref + ',  '
                else:
                    y_label_ref_u_as_str += y_label_ref_u[number]
                    y_label_ref_as_str += ref
            title_str = f'{y_label_ref_as_str}   -   {x_label_name}'
            for var_id in multiplot_legend_dict:
                plt.plot(custom_df[var_id], label=multiplot_legend_dict[var_id])
            if logic_limit_for_independent_variable:
                x_axis_limits_algorithm()
                y_axis_limits_algorithm(list(multiplot_legend_dict.keys()))
            plt.legend(loc='lower right')
            plt.title(title_str)
            plt.xlabel(x_label_name_and_units)
            plt.ylabel(y_label_ref_u_as_str)
            plt.minorticks_on()
            plt.grid(which='both')
            plt.show()

    else:
        raise InputDataError('No predefined mode selected.')


if __name__ == '__main__':
    main()
