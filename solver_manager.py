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
    product_attr = {'tm': prd.thermo_mode, 'tol': prd.relative_error,
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

    solver_obj.prd = gas_model_to_solver(thermo_mode=prd_attr['tm'], relative_error=prd_attr['tol'],
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
                 m_dot=None, p_out=None, C_inx_ref=None, resolution: float | int = None, req_vars: list = None):
    """ El rango debe ser creciente. """
    k = 0
    sweep_data = {}
    lista_df_a, lista_df_b, lista_df_c = [], [], []

    if var_to_sweep == 'T_in':
        sweep_data[var_to_sweep] = T_in
        sweep_data['units'] = 'K'
    elif var_to_sweep == 'p_in':
        sweep_data[var_to_sweep] = p_in
        sweep_data['units'] = 'Pa'
    elif var_to_sweep == 'n_rpm':
        sweep_data[var_to_sweep] = n_rpm
        sweep_data['units'] = 'rpm'
    elif var_to_sweep == 'p_out':
        sweep_data[var_to_sweep] = p_out
        sweep_data['units'] = 'Pa'
    elif var_to_sweep == 'm_dot':
        sweep_data[var_to_sweep] = m_dot
        sweep_data['units'] = 'kg'
    elif var_to_sweep == 'C_inx':
        sweep_data[var_to_sweep] = C_inx
        sweep_data['units'] = 'm/s'

    def set_value(value):
        nonlocal T_in, p_in, n_rpm, p_out, m_dot, C_inx
        if var_to_sweep == 'T_in':
            T_in = value
        elif var_to_sweep == 'p_in':
            p_in = value
        elif var_to_sweep == 'n_rpm':
            n_rpm = value
        elif var_to_sweep == 'p_out':
            p_out = value
        elif var_to_sweep == 'm_dot':
            m_dot = value
        elif var_to_sweep == 'C_inx':
            C_inx = value

    if resolution is None:
        resolution = 200
    jump = (sweep_data[var_to_sweep][1] - sweep_data[var_to_sweep][0]) / (resolution - 1)
    while k < resolution:

        value_k = sweep_data[var_to_sweep][0] + (k * jump)
        set_value(value_k)

        try:
            solver.problem_solver(T_in=T_in, p_in=p_in, n_rpm=n_rpm, m_dot=m_dot, C_inx=C_inx, p_out=p_out,
                                  C_inx_ref=C_inx_ref)
        except GasLibraryAdaptedException:
            pass
        except NonConvergenceError:
            pass
        else:
            df_a, df_b, df_c = data_to_df(solver, req_vars)
            lista_df_a.append(copy.deepcopy(df_a))
            lista_df_b.append(copy.deepcopy(df_b))
            lista_df_c.append(copy.deepcopy(df_c))
        k += 1

    df_a_packg = pd.concat([*lista_df_a], keys=[f'{v}' for v in range(resolution)], names=['Aux_Index', 'Spec_Index'])
    df_b_packg = pd.concat([*lista_df_b], keys=[f'{v}' for v in range(resolution)], names=['Aux_Index', 'Spec_Index'])
    df_c_packg = pd.concat([*lista_df_c], keys=[f'{v}' for v in range(resolution)], names=['Aux_Index', 'Spec_Index'])

    return df_a_packg, df_b_packg, df_c_packg


def main_1(fast_mode, action):
    if action == 'procesar_y_guardar':
        settings = config_class(relative_error=1E-12, n_steps=1, jump=0.004, loss_model='Aungier',
                                ideal_gas=True, chain_mode=fast_mode, iter_limit=800)

        Rm = 0.1429
        heights = [0.0445 for _ in range(3)]
        areas = [0.0399 for _ in range(3)]
        chord = [0.0338, 0.0241]
        t_max = [0.2 * chord[0], 0.15 * chord[1]]
        pitch = [0.0249, 0.0196]
        t_e = [0.01 * s for s in pitch]
        blade_opening = [0.01090, 0.01354]
        e_param = [0.0893, 0.01135]
        tip_clearance = [0.0004, 0.0008]
        # 'wire_diameter' 'lashing_wires'
        chord_proj_z = [0.9 * b for b in chord]
        blade_roughness_peak_to_valley = [0.00001 for _ in chord]

        settings.set_geometry(B_A_est=0, theta_est=70, B_A_rot=55, theta_rot=105, areas=areas, cuerda=chord,
                              radio_medio=Rm, e=e_param, o=blade_opening, s=pitch, H=heights, b_z=chord_proj_z,
                              t_max=t_max, r_r=0.002, r_c=0.001, t_e=t_e, k=tip_clearance, delta=tip_clearance,
                              roughness_ptv=blade_roughness_peak_to_valley, holgura_radial=False)

        gas_model = gas_model_to_solver(thermo_mode="ig")

        solver = solver_object(settings, gas_model)

        if fast_mode:
            output = solver.problem_solver(T_in=1100, p_in=600_000, n_rpm=20_000, p_out=120_000, C_inx_ref=110)
            T_salida, p_salida, C_salida, alfa_salida = output
            print(' T_out', T_salida, '\n', 'P_out', p_salida, '\n', 'C_out', C_salida, '\n', 'alfa_out', alfa_salida)
        else:
            solver.problem_solver(T_in=1100, p_in=600_000, n_rpm=20_000, p_out=120_000, C_inx_ref=110)
            solver_data_saver('process_object.pkl', solver)

    elif action == 'cargar_y_visualizar':
        solver = solver_data_reader('process_object.pkl')
        problem_data_viewer(solver)

    elif action == 'cargar_reprocesar_y_guardar':
        # Se usan semillas de la ejecución anterior. Se leen, se guardan y se visualizan los datos.
        # Esta ejecución es más rápida que la ejecución normal, ya que se aprovechan las semillas del objeto cargado.
        solver = solver_data_reader('process_object.pkl')
        solver.cfg.set_geometry(B_A_est=[0, 5], theta_est=[70, 75], B_A_rot=[55, 55], theta_rot=[105, 105], b_z=0.027,
                                cuerda=0.03, radio_medio=0.30, H=[0.030, 0.035, 0.041, 0.048, 0.052], e=0.015, o=0.015,
                                t_max=0.006, r_r=0.003, r_c=0.002, t_e=0.004, k=0.001, delta=0.001,
                                roughness_ptv=0.00001, holgura_radial=False)

        solver.problem_solver(T_in=1800, p_in=1_000_000, n_rpm=6_000, m_dot=18.0)
        solver_data_saver('process_object.pkl', solver)
        problem_data_viewer(solver)


def main_2():
    settings = config_class(relative_error=1E-11, ideal_gas=True, n_steps=1, jump=0.5,
                            loss_model='Ainley_and_Mathieson', chain_mode=False,
                            iter_limit=2000, max_trend_changes=30)

    Rm = 0.1429
    heights = [0.0445 for _ in range(3)]
    areas = [0.0399 for _ in range(3)]
    chord = [0.0338, 0.0241]
    t_max = [0.2 * chord[0], 0.15 * chord[1]]
    pitch = [0.0249, 0.0196]
    t_e = [0.01 * s for s in pitch]
    blade_opening = [0.01090, 0.01354]
    e_param = [0.0893, 0.01135]
    tip_clearance = [0.0004, 0.0008]
    # 'wire_diameter' 'lashing_wires'
    chord_proj_z = [0.9 * b for b in chord]
    blade_roughness_peak_to_valley = [0.00001 for _ in chord]

    settings.set_geometry(B_A_est=0, theta_est=70, B_A_rot=55, theta_rot=105, areas=areas, cuerda=chord,
                          radio_medio=Rm, e=e_param, o=blade_opening, s=pitch, H=heights, b_z=chord_proj_z,
                          t_max=t_max, r_r=0.002, r_c=0.001, t_e=t_e, k=tip_clearance, delta=tip_clearance,
                          roughness_ptv=blade_roughness_peak_to_valley, holgura_radial=False)

    gas_model = gas_model_to_solver(thermo_mode="ig")
    solver = solver_object(settings, gas_model)

    df_a, df_b, df_c = var_sweeping(solver, T_in=1100, p_in=400_000, n_rpm=17_000, C_inx=[0.01, 180],
                                    var_to_sweep='C_inx', resolution=1000)

    df_a.to_csv('df_a_AM.csv')
    df_b.to_csv('df_b_AM.csv')
    df_c.to_csv('df_c_AM.csv')


def main_3():

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
    p_out_m_dot = pd.DataFrame((df_a_pt_3['p (Pa)']/1000).values.tolist(), columns=['p'], index=df_c.index)
    p_out_C_inx = pd.DataFrame((df_a_pt_3['p (Pa)']/1000).values.tolist(), columns=['p'], index=df_a_pt_1['C (m/s)'])
    p_out_h0 = pd.DataFrame((df_a_pt_3['p (Pa)']/1000).values.tolist(), columns=['p'], index=df_a_pt_1['h0 (kJ/kg)'])
    p_out_T0 = pd.DataFrame((df_a_pt_3['p (Pa)']/1000).values.tolist(), columns=['p'], index=df_a_pt_1['T0 (K)'])
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

    return


if __name__ == '__main__':
    main_2()
