"""
Se almacenan y procesan los datos haciendo uso de las librerías Pickle y Pandas.
"""

import pickle
from axial_turb_solver import *
import pandas as pd

# https://conocepython.blogspot.com/2019/05/ (este es interesante)
# https://docs.python.org/3/library/shelve.html (documentación)
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
# https://docs.python.org/es/dev/library/dbm.html#dbm.open (flag keyword)
def solver_data_saver(file: str, process_object: solver_process) -> None:
    """ Se almacena la información necesaria para poder restablecer la situación del programa en la siguiente
    ejecución.
            :param file: Nombre de los archivos .db generados.
            :param process_object: Objeto proceso (Solver).
                    :return: No se devuelve nada. """

    prd = process_object.prd
    product_attr = {'tm': prd.thermo_mode, 'tol': prd.rel_error,
                    'C': prd.C_atoms, 'H': prd.H_atoms, 'N': prd.air_excess}
    process_object.prd = None   # Esto se hace así para evitar un error que surge

    with open(file, 'wb') as solver_pickle:
        pickle.dump(process_object, solver_pickle)
        pickle.dump(product_attr, solver_pickle)

    return


def solver_data_reader(file: str) -> solver_process:
    """ Se restablece el estado del programa en la ejecución en que fue efectuada la función 'solver_data_saver',
    reestableciendo la configuración y el estado del solver.
            :param file: Nombre de los archivos generados.
                    :return: Se devuelve el solver ya configurado. """

    with open(file, 'rb') as obj_pickle:
        solver_obj: solver_process = pickle.load(obj_pickle)
        prd_attr = pickle.load(obj_pickle)

    solver_obj.prd = gas_model_to_solver(thermo_mode=prd_attr['tm'], rel_error=prd_attr['tol'],
                                         C_atoms=prd_attr['C'], H_atoms=prd_attr['H'], air_excess=prd_attr['N'])

    return solver_obj


def data_to_df(process_object: solver_process, req_vars=None) -> [pd.DataFrame | None, pd.DataFrame | None,
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
    lista_b = ['GR', 'w_esc', 'w_s_esc', 'w_ss_esc', 'eta_TT', 'Y_est', 'Y_rot', 'Y_esc', 'U']
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


def problem_data_viewer(solver_object: solver_process, req_vars=None) -> None:
    """ Se visualiza el dataframe generado. """

    df_a, df_b, df_c = data_to_df(solver_object, req_vars)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None,
                           'display.width', 1000, 'display.precision', 4):
        if df_a is not None:
            print(df_a, '\n')
        if df_b is not None:
            print(df_b, '\n')
        if df_c is not None:
            print(df_c, '\n')

    return


def main(fast_mode, action):

    if action == 'w':
        settings = config_param(TOL=1E-9, n_steps=2, ideal_gas=True, fast_mode=fast_mode,
                                loss_model='ainley_and_mathieson')

        settings.set_geometry(B_A_est=[0, 39], theta_est=[70, 90], B_A_rot=[35, 52], theta_rot=[100, 90],
                              cuerda=0.03, radio_medio=0.3, H=[0.009, 0.016, 0.026, 0.0300, 0.0380],
                              A_rel=0.75, t_max=0.008, r_r=0.003, r_c=0.002, t_e=0.004, K=0.0)

        gas_model = gas_model_to_solver(thermo_mode="ig", rel_error=1E-9)

        solver = solver_process(settings, gas_model)

        if fast_mode:
            output = solver.problem_solver(T_in=1800, p_in=1_200_000, n=6_500, m_dot=7.0)
            T_salida, p_salida, C_salida, alfa_salida = output
            print(' T_out', T_salida, '\n', 'P_out', p_salida, '\n', 'C_out', C_salida, '\n', 'alfa_out', alfa_salida)
        else:
            solver.problem_solver(T_in=1800, p_in=1_200_000, n=6_500, m_dot=7.0)
            solver_data_saver('process_object.pkl', solver)

    elif action == 'r':
        solver = solver_data_reader('process_object.pkl')
        problem_data_viewer(solver)

    elif action == 'wr':
        # Se usan semillas de la ejecución anterior. Se leen, se guardan y se visualizan los datos.
        solver = solver_data_reader('process_object.pkl')
        solver.cfg.set_geometry(B_A_est=[0, 39], theta_est=[70, 90], B_A_rot=[35, 52], theta_rot=[100, 90],
                                cuerda=0.03, radio_medio=0.3, H=[0.009, 0.016, 0.026, 0.0300, 0.0380],
                                A_rel=0.75, t_max=0.008, r_r=0.003, r_c=0.002, t_e=0.004, K=0.0)

        solver.problem_solver(1800, 1_200_000, 6_500, m_dot=7.0)
        solver_data_saver('process_object.pkl', solver)
        problem_data_viewer(solver)


if __name__ == '__main__':
    main(fast_mode=False, action='wr')
