"""
En este módulo se define una clase que caracteriza un objeto que contiene la configuración y los parámetros geométricos
que se emplearán en el cálculo. Se requiere un módulo aparte para esta clase para evitar importaciones circulares.
"""
import logging
import sys
from math import pi, radians, cos, tan


class configuration_parameters:
    """ Objeto que agrupa los parámetros necesarios para configurar la ejecución de todo el programa. """

    def __init__(self, rel_error: float, number_steps: int, fast_mode: bool, loss_model_id: str, thermo_mode="ig",
                 C_atoms=12.0, H_atoms=23.5, N=2):
        """ :param number_steps: Número de escalonamientos.
        :param rel_error: Error relativo máximo permitido.
        :param fast_mode: Modo de funcionamiento en que, mediante el mínimo número de cálculos, se calculan
                          Temperatura, presión y vector velocidad a la salida de la turbina.
        :param loss_model_id: Cadena de caracteres identificadora del modelo de pérdidas seleccionado.
        :param thermo_mode: Cadena de caracteres identificadora del modelo termodinámico seleccionado: "ig" u otro.
        :param C_atoms: Átomos de carbono en un átomo del hidrocarburo en los reactivos.
        :param H_atoms: Átomos de hidrógeno en un átomo del hidrocarburo en los reactivos.
        :param N: Exceso de aire en los reactivos."""
        self.TOL = rel_error  # Máximo error relativo que se permite en los cálculos iterativos
        self.n_step = number_steps  # Número de escalonamientos que se definen
        self.fast_mode = fast_mode  # Limitar cálculos y determinar temperatura, presión y velocidad a la salida
        self.geom = {}  # Diccionario para almacenar parámetros geométricos de la turbina
        self.loss_model = loss_model_id  # String con el nombre del modelo de pérdidas empleado
        if loss_model_id not in ['soderberg_correlation', 'ainley_and_mathieson']:
            logging.critical('Los identificadores de los modelos de pérdidas disponibles son "soderberg_correlation" y '
                             '"ainley_and_mathieson"')
            sys.exit()
        self.thermo_mode = thermo_mode
        self.C_atoms, self.H_atoms, self.N = C_atoms, H_atoms, N

    def set_geometry(self, alfap_i_est, theta_e, alfap_i_rot, theta_r, b, R_m, **kwargs):
        """ Se establece la geometría de la turbina del problema.

        Los parámetros pueden indicarse en forma de lista o en forma de un único número, entero o de coma flotante, en
        el caso de que este se repita para cada escalonamiento.

        Usar sistema internacional excepto para los ángulos, que se deben indicar en grados sexagecimales.
                :param b: Cuerda de cada álabe.
                :param R_m: Radio medio de un álabe de escalonamiento, si se indica un valor único se considera
                           constante a lo largo de los escalonamientos (indicar número o lista).
                :param alfap_i_est: Ángulo del borde de ataque del álabe de estátor.
                :param theta_e: Ángulo de curvatura del álabe de estátor.
                :param alfap_i_rot: Ángulo del borde de ataque del álabe de rótor.
                :param theta_r: Ángulo de curvatura del álabe de rótor.
                :param kwargs: Se requiere definir cada corona de manera suficiente según el modelo de pérdidas que se
                               establezca. Se permite definir las secciones de manera explícita o de manera implícita,
                               fijando el radio medio y las alturas. Si no se establece un paso se establecerá
                               automáticamente el paso óptimo según el criterio de Zweifel.
                               Las claves son: Rm (radio medio), H (alturas), areas,
                               A_rel (relacion área álabes - área total), t_max (espesor máximo de un álabe),
                               r_r (radio raiz), r_c (radio cabeza), t_e (espesor del borde de salida),
                               K (parámetro para el hueco libre de Ainley and Mathieson), s (paso, distancia
                               tangencial entre álabes).
                        :return: No se devuelve nada. """
        ns = self.n_step
        local_dict1 = {'alfap_i_est': alfap_i_est, 'theta_e': theta_e, 'alfap_i_rot': alfap_i_rot, 'theta_r': theta_r}
        list_items2, local_dict2 = ['A_rel', 't_max', 'r_r', 'r_c', 't_e', 'K'], {}
        for id_ in list_items2:
            local_dict2[id_] = kwargs.get(id_, 0.0)
        local_dict2['b'], local_dict2['Rm'] = b, R_m
        s = kwargs.get('s', 0.0)
        if not s == 0.0:
            local_dict2['s'] = s
        ld_list = [local_dict1, local_dict2]
        for index, ld in enumerate(ld_list):
            for i, v in ld.items():
                if not isinstance(v, int) and not isinstance(v, float):
                    if index == 0:
                        v = [radians(elem) for elem in v]
                    self.geom[i] = tuple(v)
                else:
                    self.geom[i] = []
                    for _ in range(ns if index == 0 else ns*2):
                        self.geom[i] += [radians(v) if index == 0 else v]
                    self.geom[i] = tuple(self.geom[i])
        Rm = self.geom['Rm']
        for i1, i2 in [['areas', 'H'], ['H', 'areas']]:
            if i1 not in kwargs:
                self.geom[i1] = list()
                self.geom[i2] = kwargs[i2]
                for num, v2 in enumerate(self.geom[i2]):
                    num2 = num - 1 if num > 0 else 0
                    self.geom[i1].append(2 * pi * Rm[num2] * v2 if (i1 == 'areas') else v2 / (2 * pi * Rm[num2]))
                self.geom[i1] = tuple(self.geom[i1])
        for num, h in enumerate(self.geom['H']):  # h: 0 1 2 3 4 ... Rm: 0 0 1 2 3
            num2 = num - 1 if num > 0 else 0
            if h/(2*Rm[num2]) > 0.3:
                logging.warning('No se verifica la hipótesis de bidimensionalidad')
                sys.exit()
        if s == 0.0:
            ap_i_est, ap_i_rot, b = self.geom['alfap_i_est'], self.geom['alfap_i_rot'], self.geom['b']
            self.geom['s'] = [b[(i//2)*2]*0.4/((tan(ap_i_est[i//2])+tan(ap_i_rot[i//2])) *
                                               (cos(ap_i_rot[i//2])**2)) for i in range(ns*2)]
        self.geom['alfap_o_est'] = [self.geom['theta_e'][i] - self.geom['alfap_i_est'][i] for i in range(ns)]
        self.geom['alfap_o_rot'] = [self.geom['theta_r'][i] - self.geom['alfap_i_rot'][i] for i in range(ns)]
        return
