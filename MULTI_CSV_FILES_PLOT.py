from config_class import *
import pandas as pd
import matplotlib.pyplot as plt

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


def function_for_itering_csvfiles(data_dictionary):
    csv_extensions_list = data_dictionary.get('csv_filename_extension')
    legend_loc = data_dictionary.get('legend_loc')
    title_str = x_label_name_and_units = None
    dep_ids__dict = dict()
    dependent_vars = list()

    def functionwithoutplot(csv_filename_extension):
        nonlocal title_str, x_label_name_and_units, dep_ids__dict, dependent_vars
        x_label_name_and_units = None
        dep_ids__dict = {}
        try:
            logic_limit_for_independent_variable: bool = data_dictionary['logic_limit_for_independent_variable']
            req_vars = data_dictionary['req_vars']
            WtE: dict = data_dictionary['where_to_evaluate']
            independent_var = data_dictionary['independent_variable']
            dependent_vars = data_dictionary['dependent_variables']
            a_filename, b_filename, c_filename = [root + csv_filename_extension for root in
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
                                         'process saved data to be plotted. \nPlease, check items that need to be added'
                                         ' to "where_to_evaluate" dictionary.')
            elif item in lista_c:
                units_extension = tpl_t_units[tpl_t_keys.index(item)]
                custom_df[item] = df_c[item + units_extension]
                if item == independent_var:
                    x_label_name = item
                    x_label_name_and_units = x_label_name + units_extension
                else:
                    dep_ids__dict[item] = [item]
                    dep_ids__dict[item].append(item + units_extension)  # For y-labels when not plotting.
                    dep_ids__dict[item].append(item)  # This one is for leyends when needed.
                    dep_ids__dict[item].append(item + units_extension)  # For y-labels when multiplotting.
            else:
                sentence = f'{item} is not considered as a possible input. Use req_vars for considering variables ' \
                           f'listed in the handbook.'
                raise InputDataError(sentence)

        if WtE is not None:
            for key in WtE:
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
                            x_label_name = var_id
                            x_label_name_and_units = x_label_name + units_extension
                        else:
                            dep_ids__dict[var_id] = [key]
                            dep_ids__dict[var_id].append(var_id + units_extension)  # For y-labels when plotting.
                            dep_ids__dict[var_id].append(var_id)  # This one is for leyends when needed.
                            dep_ids__dict[var_id].append(key + units_extension)  # For y-labels when multiplotting.
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
                            dep_ids__dict[var_id] = [key]
                            dep_ids__dict[var_id].append(var_id + units_extension)  # This one is for y-labels (plot).
                            dep_ids__dict[var_id].append(var_id)  # This one is for leyends when needed.
                            dep_ids__dict[var_id].append(key + units_extension)  # For y-labels (multiplot).

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

        IV_limits_refs = []

        # The following function stablishes the range of values of the independent variable to be displayed
        # using as condition that resulting exchange of energy is given from the gas to the turbine.
        def x_axis_limits_algorithm():
            nonlocal IV_limits_refs
            IV_limits_refs = []

            def IV_limits_finder():
                nonlocal IV_limits_refs
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
                if len(IV_limits_refs) == 0:
                    raise InputDataError('No data meets the condition. There is no value to be displayed.')
                elif len(IV_limits_refs) == 1:
                    raise InputDataError('Only one value available, plotting process cannot be completed.')
                else:
                    p1, p2 = IV_limits_refs[0], IV_limits_refs[1]
                    if custom_df.index[p2] < custom_df.index[p1]:
                        IV_limits_refs[0], IV_limits_refs[1] = IV_limits_refs[1], IV_limits_refs[0]
                return
            IV_limits_finder()
            plt.xlim(custom_df.index[IV_limits_refs[0]], custom_df.index[IV_limits_refs[1]])
            return

        # The following function adapts the range of values of the axis for the dependent variable to the plotted
        # function interval.
        # The function "x_axis_limits_algorithm()" is expected to be executed before.
        def y_axis_limits_algorithm(DV_identifiers_list: list[str]):
            DV_limits = []

            def DV_limits_finder():
                for DV_ID in DV_identifiers_list:
                    ivra, ivrb = IV_limits_refs[0], IV_limits_refs[1]
                    if ivra > ivrb:
                        ivra, ivrb = ivrb, ivra
                    for i in range(ivra, ivrb+1):
                        if len(DV_limits) == 0:
                            if i == ivra+1:
                                DV_limits.append(custom_df.iloc[i-1][DV_ID])
                                DV_limits.append(custom_df.iloc[i][DV_ID])
                                if custom_df.iloc[i][DV_ID] < custom_df.iloc[i-1][DV_ID]:
                                    DV_limits[0], DV_limits[1] = DV_limits[1], DV_limits[0]
                        else:
                            if custom_df.iloc[i][DV_ID] < DV_limits[0]:
                                DV_limits[0] = custom_df.iloc[i][DV_ID]
                            if custom_df.iloc[i][DV_ID] > DV_limits[1]:
                                DV_limits[1] = custom_df.iloc[i][DV_ID]
            eta_checker = False
            for DV_ident in DV_identifiers_list:
                if 'eta' in DV_ident:
                    eta_checker = True
            if eta_checker:
                DV_limits = [0, 1]
                diff = 0
            else:
                DV_limits_finder()
                diff = 0.22*(DV_limits[1]-DV_limits[0])
            plt.ylim(DV_limits[0]-diff, DV_limits[1]+diff)
        # Section for plotting only one independent variable at the time (Modified)
        plt.plot(custom_df[dep_ids__dict[dependent_vars[0]][0]])
        if logic_limit_for_independent_variable:
            x_axis_limits_algorithm()
            y_axis_limits_algorithm(dep_ids__dict[dependent_vars[0]])
        # Aquí acaba la función local
    for csv_extension in csv_extensions_list:
        functionwithoutplot(csv_extension)
    if legend_loc is not None:
        plt.legend(loc=legend_loc)
    plt.title(title_str)
    plt.xlabel(x_label_name_and_units)
    plt.ylabel(dep_ids__dict[dependent_vars[0]][1])
    plt.minorticks_on()
    plt.grid(which='both')
    plt.show()
