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

t_keys = 'DELTA_h, w_total, P_total, s_A, s_B, p0_B, T0_B, T0_Bss, h0_Bss, Y_maq, w_ss_total, eta_maq, p0_A, T0_A, ' \
         'eta_p, r_turbina, m_dot'

t_units = '(kJ/kg), (kJ/kg), (kW), (kJ/kgK), (kJ/kgK), (Pa), (K), (K), (kJ/kg), (kJ/kg), (kJ/kg), (-), (Pa), (K), ' \
          '(-), (-), (kg/s)'

tpl_s_keys, tpl_t_keys = tuple(s_keys.split(', ')), tuple(t_keys.split(', '))
tpl_s_units, tpl_t_units = tuple(s_units.split(',')), tuple(t_units.split(','))


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

    lista_a = ['M', 'C', 'Cx', 'Cu', 'omega', 'omegax', 'omegau', 'alfa', 'beta', 'h', 'h0', 'T', 'T0',
               'p', 'p0', 'rho', ]
    lista_b = ['GR', 'w_esc', 'w_s_esc', 'w_ss_esc', 'eta_TT', 'eta_TE', 'Y_est', 'Y_rot', 'U']
    lista_c = ['w_total', 'w_ss_total', 'eta_maq', 'P_total', 'r_turbina', 'm_dot', ]

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
                 m_dot=None, p_out=None, C_inx_ref=None, sweep_resolution: float | int = None, req_vars: list = None):
    """ El rango debe ser creciente. """
    k = 0
    sweep_data = {}
    lista_df_a, lista_df_b, lista_df_c = [], [], []

    if var_to_sweep == 'T_inlet':
        sweep_data[var_to_sweep] = T_in
        sweep_data['units'] = 'K'
    elif var_to_sweep == 'p_inlet':
        sweep_data[var_to_sweep] = p_in
        sweep_data['units'] = 'Pa'
    elif var_to_sweep == 'rpm':
        sweep_data[var_to_sweep] = n_rpm
        sweep_data['units'] = 'rpm'
    elif var_to_sweep == 'p_outlet':
        sweep_data[var_to_sweep] = p_out
        sweep_data['units'] = 'Pa'
    elif var_to_sweep == 'mass_flow':
        sweep_data[var_to_sweep] = m_dot
        sweep_data['units'] = 'kg'
    elif var_to_sweep == 'axial_inlet_velocity':
        sweep_data[var_to_sweep] = C_inx
        sweep_data['units'] = 'm/s'

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

    if sweep_resolution is None:
        sweep_resolution = 200
    jump = (sweep_data[var_to_sweep][1] - sweep_data[var_to_sweep][0]) / (sweep_resolution - 1)
    while k < sweep_resolution:

        value_k = sweep_data[var_to_sweep][0] + (k * jump)
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
            pass
        except NonConvergenceError:
            pass
        else:
            df_a, df_b, df_c = data_to_df(solver, req_vars)
            lista_df_a.append(copy.deepcopy(df_a))
            lista_df_b.append(copy.deepcopy(df_b))
            lista_df_c.append(copy.deepcopy(df_c))
        record.info('Barrido de la variable completo en un %s', k/sweep_resolution)
        k += 1

    df_a_packg = pd.concat(
        [*lista_df_a], keys=[f'{v}' for v in range(sweep_resolution)], names=['Aux_Index', 'Spec_Index'])
    df_b_packg = pd.concat(
        [*lista_df_b], keys=[f'{v}' for v in range(sweep_resolution)], names=['Aux_Index', 'Spec_Index'])
    df_c_packg = pd.concat(
        [*lista_df_c], keys=[f'{v}' for v in range(sweep_resolution)], names=['Aux_Index', 'Spec_Index'])

    return df_a_packg, df_b_packg, df_c_packg


def txt_reader():
    with open('turbine_data_template.txt') as file:
        for line in file:
            declaration = ''
            for char in line.strip():
                if char != ';':
                    if char not in [' ', '#']:
                        if char not in ['=', ',']:
                            declaration += char
                        elif char == ',':
                            declaration += ', '
                        else:
                            declaration += ' = '
                    elif char == '#':
                        break
                else:
                    if '$' in declaration:
                        exec(declaration.replace('for', ' for ').replace('in', ' in ').replace('$', ''))
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
                iter_limit=data_dictionary['iter_limit'],
                max_trend_changes=data_dictionary['max_trend_changes'],
                T_nominal=data_dictionary['T_nominal'],
                preloading_for_small_input_deviations=data_dictionary[
                    'preloading_for_small_input_deviations'
                ],
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
            root_radius = data_dictionary['root_radius']
            head_radius = data_dictionary['head_radius']
            Rm = data_dictionary['radio_medio']
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
            holgura_radial = data_dictionary['holgura_radial']
            blade_roughness_peak_to_valley = data_dictionary['blade_roughness_peak_to_valley']
            design_factor = data_dictionary['design_factor']
        except NameError:
            raise InputDataError('Non-valid text file, please, stick to the template.')

        settings.set_geometry(B_A_est=LE_stator, theta_est=theta_stator, B_A_rot=LE_rotor, theta_rot=theta_rotor,
                              H=heights, B_S_est=TE_stator, B_S_rot=TE_rotor, areas=areas, cuerda=chord, radio_medio=Rm,
                              e=e_param, b_z=chord_proj_z, o=blade_opening, s=pitch, t_max=t_max, r_r=root_radius,
                              delta=tip_clearance, r_c=head_radius, k=tip_clearance, t_e=t_e,
                              roughness_ptv=blade_roughness_peak_to_valley, lashing_wires=lashing_wires,
                              wire_diameter=wire_diameter, holgura_radial=holgura_radial, design_factor=design_factor)
        return settings

    mode = data_dictionary['modo']

    if mode == 'procesar_valores_nominales':
        settings = aux_reading_operations()
        gas_model = gas_model_to_solver(thermod_mode=data_dictionary.get('thermo_mode_in_gas_model.py', 'ig'))
        solver = solver_object(settings, gas_model)
        solver_data_saver('process_object.pkl', solver)

    elif mode == 'resolver':
        try:
            solver = solver_data_reader('process_object.pkl')
        except FileNotFoundError:
            settings = aux_reading_operations()
            gas_model = gas_model_to_solver(thermod_mode=data_dictionary.get('thermod_mode_in_gas_model_module', 'ig'))
            solver = solver_object(settings, gas_model)
        try:
            if data_dictionary['chain_mode']:
                output = solver.problem_solver(
                    T_in=data_dictionary['T_inlet'],
                    p_in=data_dictionary['p_inlet'],
                    n_rpm=data_dictionary['rpm'],
                    p_out=data_dictionary['p_outlet'],
                    m_dot=data_dictionary['mass_flow'],
                    C_inx=data_dictionary['inlet_velocity'],
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
                    C_inx=data_dictionary['inlet_velocity'],
                    C_inx_ref=data_dictionary['reference_inlet_velocity'],
                )
                solver_data_saver('process_object.pkl', solver)
        except NameError:
            raise InputDataError('Non-valid text file, please, stick to the template.')

    elif mode == 'visualizar_valores':
        solver = solver_data_reader('process_object.pkl')
        problem_data_viewer(solver)

    elif mode == 'recorrer_variable':
        settings = aux_reading_operations()
        gas_model = gas_model_to_solver(thermod_mode=data_dictionary.get('thermod_mode_in_gas_model_module', 'ig'))
        solver = solver_object(settings, gas_model)
        try:
            df_a, df_b, df_c = var_sweeping(
                solver,
                T_in=data_dictionary['T_inlet'],
                p_in=data_dictionary['p_inlet'],
                n_rpm=data_dictionary['rpm'],
                m_dot=data_dictionary['mass_flow'],
                C_inx=data_dictionary['inlet_velocity'],
                C_inx_ref=data_dictionary['reference_inlet_velocity'],
                p_out=data_dictionary['p_outlet'],
                req_vars=data_dictionary['req_vars'],
                var_to_sweep=data_dictionary['var_to_sweep'],
                sweep_resolution=data_dictionary['sweep_resolution'])
        except NameError:
            raise InputDataError('Non-valid text file, please, stick to the template.')
        df_a.to_csv('df_a.csv')
        df_b.to_csv('df_b.csv')
        df_c.to_csv('df_c.csv')
        solver_data_saver('process_object.pkl', solver)

    elif mode == 'visualizar_recorrido':
        df_c = pd.read_csv('df_c.csv', index_col='m_dot (kg/s)')
        eta_s = df_c['eta_maq (-)']
        Potencia = df_c['P_total (kW)']
        Potencia_ss = df_c['w_ss_total (kJ/kg)'] * df_c.index

        plt.plot(eta_s)
        plt.minorticks_on()
        plt.grid(which='both')
        plt.title('Rendimiento isentrópico - Flujo másico')
        plt.xlabel(r'$\dot{m}$ (kg/s)')
        plt.ylabel(r'$\eta_{s}$ (-)')
        plt.show()

        plt.plot(Potencia)
        plt.plot(Potencia_ss)
        plt.title('Potencia - Flujo másico')
        plt.xlabel(r'$\dot{m}$ (kg/s)')
        plt.ylabel(r'$P$ (kW)')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.show()

        df_a = pd.read_csv('df_a.csv', index_col='Aux_Index')
        df_a_pt_3 = df_a[df_a['Spec_Index'] == 'Step_1_pt_3']
        df_a_pt_2 = df_a[df_a['Spec_Index'] == 'Step_1_pt_2']
        df_a_pt_1 = df_a[df_a['Spec_Index'] == 'Step_1_pt_1']
        p_out_m_dot = pd.DataFrame(
            (df_a_pt_3['p (Pa)']/1000).values.tolist(), columns=['p'], index=df_c.index)
        p_out_C_inx = pd.DataFrame(
            (df_a_pt_3['p (Pa)']/1000).values.tolist(), columns=['p'], index=df_a_pt_1['C (m/s)'])
        p_out_h0 = pd.DataFrame(
            (df_a_pt_3['p (Pa)']/1000).values.tolist(), columns=['p'], index=df_a_pt_1['h0 (kJ/kg)'])
        p_out_T0 = pd.DataFrame(
            (df_a_pt_3['p (Pa)']/1000).values.tolist(), columns=['p'], index=df_a_pt_1['T0 (K)'])
        p_out = p_out_m_dot['p']
        plt.plot(p_out)
        plt.title('Presión a la salida - Flujo másico')
        plt.xlabel(r'$\dot{m}$ (kg/s)')
        plt.ylabel(r'$p_{out}$ (kPa)')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.show()

        p_out = p_out_C_inx['p']
        plt.plot(p_out)
        plt.title('Presión a la salida - Velocidad a la entrada')
        plt.xlabel(r'$\dot{C}_{in}$ (m/s)')
        plt.ylabel(r'$p_{out}$ (kPa)')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.show()

        p_out = p_out_h0['p']
        plt.plot(p_out)
        plt.title('Presión a la salida - Entalpía de remanso a la entrada')
        plt.xlabel(r'$h_{0in}$ (kJ/kg)')
        plt.ylabel(r'$p_{out}$ (kPa)')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.show()

        p_out = p_out_T0['p']
        plt.plot(p_out)
        plt.title('Presión a la salida - Temperatura de remanso a la entrada')
        plt.xlabel(r'$T_{0in}$ (K)')
        plt.ylabel(r'$p_{out}$ (kPa)')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.show()

        df_b = pd.read_csv('df_b.csv')
        eta_TT_m_dot = pd.DataFrame((df_b['eta_TT (-)']).values.tolist(), columns=['eta_TT'], index=df_c.index)
        eta_TE_m_dot = pd.DataFrame((df_b['eta_TE (-)']).values.tolist(), columns=['eta_TE'], index=df_c.index)
        xi_est_m_dot = pd.DataFrame((df_b['Y_est (kJ/kg)']/(0.0005*(df_a_pt_2['C (m/s)'] *
                                                                    df_a_pt_2['C (m/s)']))).values.tolist(),
                                    columns=['xi_est'], index=df_c.index)
        xi_rot_m_dot = pd.DataFrame((df_b['Y_rot (kJ/kg)']/(0.0005*(df_a_pt_3['omega (m/s)'] *
                                                                    df_a_pt_3['omega (m/s)']))).values.tolist(),
                                    columns=['xi_rot'], index=df_c.index)
        plt.plot(eta_TT_m_dot['eta_TT'], label='Rendimiento total a total')
        plt.plot(eta_TE_m_dot['eta_TE'], label='Rendimiento total a estático')
        plt.plot(xi_est_m_dot['xi_est'], label='Coeficiente adimensional de pérdidas en estátor')
        plt.plot(xi_rot_m_dot['xi_rot'], label='Coeficiente adimensional de pérdidas en rótor')
        plt.minorticks_on()
        plt.grid(which='both')
        plt.show()


if __name__ == '__main__':
    main()
