"""
En este módulo se define una clase que caracteriza un objeto que contiene la configuración y los parámetros geométricos
que se emplearán en el cálculo. Se emplea un módulo aparte para esta clase para evitar importaciones circulares, ya que
se va a almacenar información que será necesaria en varios módulos. Además, también se crea una clase que facilita
el intercambio del módulo "gas_modeling.py" por otro.
"""

import logging  # https://docs.python.org/es/3/howto/logging.html
from gas_modeling import mixpm
import sys
from math import pi, radians, cos, tan
from dataclasses import dataclass


@dataclass(frozen=True)
class config_param:
    """ Objeto que agrupa los parámetros necesarios para configurar la ejecución del solver. """

    TOL: float  # Máximo error relativo que se tolera en los cálculos iterativos
    n_steps: int  # Número de escalonamientos que se definen
    fast_mode: bool  # Limitar cálculos y determinar temperatura, presión y velocidad a la salida
    loss_model: str  # Cadena identificador del modelo de pérdidas establecido
    ideal_gas: bool  # True cuando se establece hipótesis de gas ideal
    geom: dict = None  # Diccionario para almacenar parámetros geométricos de la turbina

    def __post_init__(self):
        if self.loss_model not in ['soderberg_correlation', 'ainley_and_mathieson']:
            logging.critical('Los identificadores de los modelos de pérdidas disponibles son "soderberg_correlation" y '
                             '"ainley_and_mathieson".')
            sys.exit()

    def set_geometry(self, B_A_est: int | float | list, B_A_rot: int | float | list, cuerda: int | float | list,
                     radio_medio: int | float | list, B_S_est: int | float | list = None,
                     B_S_rot: int | float | list = None, theta_est: int | float | list = None,
                     theta_rot: int | float | list = None, **kwargs: int | float | list[float]):
        """ Se establece la geometría de la turbina del problema. Se debe indicar el borde de salida o la deflexión de
        cada álabe. Algunos parámetros siguen el mismo razonamiento de alternancia, por cuestión de flexibilidad.\n

        Los parámetros pueden indicarse en forma de lista o en forma de un único número, entero o de coma flotante, en
        el caso de que este se repita para cada escalonamiento.\n

        Usar sistema internacional excepto para los ángulos, que se deben indicar en grados sexagecimales.

                :param cuerda: Cuerda de cada álabe.
                :param radio_medio: Radio medio de un álabe de escalonamiento, si se indica un valor único se considera
                           constante a lo largo de los escalonamientos (indicar número o lista).
                :param B_A_est: Ángulo del borde de ataque de un álabe del estátor.
                :param B_A_rot: Ángulo del borde de ataque de un álabe del rótor.
                :param B_S_est: Ángulo del borde de salida de un álabe del estátor.
                :param B_S_rot: Ángulo del borde de salida de un álabe del rótor.
                :param theta_est: Ángulo de curvatura del álabe de estátor.
                :param theta_rot: Ángulo de curvatura del álabe de rótor.
                :param kwargs: Se requiere definir cada corona de manera suficiente según el modelo de pérdidas que se
                               establezca. Se permite definir las secciones de manera explícita o de manera implícita,
                               fijando el radio medio y las alturas. Si no se define un paso, se establecerá
                               automáticamente el paso óptimo según el criterio de Zweifel.
                               Las claves son: Rm (radio medio), H (alturas), areas,
                               A_rel (relacion área álabes - área total), t_max (espesor máximo de un álabe),
                               r_r (radio raiz), r_c (radio cabeza), t_e (espesor del borde de salida),
                               K (parámetro para el hueco libre de Ainley and Mathieson), s (paso, distancia
                               tangencial entre álabes).
                        :return: No se devuelve nada. """

        ns, geom = self.n_steps, dict()
        local_dict1 = {'alfap_i_est': B_A_est, 'alfap_i_rot': B_A_rot}

        for B_S, B_S_name, theta, theta_name in [[B_S_rot, 'alfap_o_rot', theta_rot, 'theta_r'],
                                                 [B_S_est, 'alfap_o_est', theta_est, 'theta_e']]:
            if B_S is not None:
                local_dict1[B_S_name] = B_S
            elif theta_rot is not None:
                local_dict1[theta_name] = theta
            else:
                logger.critical('Se precisa definir de alguna manera la deflexión de cada álabe.')
                sys.exit()

        list_items2, local_dict2 = ['A_rel', 't_max', 'r_r', 'r_c', 't_e', 'K'], {}
        for par_id in list_items2:
            local_dict2[par_id] = kwargs.get(par_id, 0.0)
        local_dict2['b'], local_dict2['Rm'] = cuerda, radio_medio
        s = kwargs.get('s', 0.0)
        if not s == 0.0:
            local_dict2['s'] = s
        ld_list = [local_dict1, local_dict2]
        for index, ld in enumerate(ld_list):
            for i, v in ld.items():
                if not isinstance(v, int) and not isinstance(v, float):
                    if index == 0:
                        v = [radians(elem) for elem in v]
                    geom[i] = tuple(v)
                else:
                    geom[i] = []
                    for _ in range(ns if index == 0 else ns*2):
                        geom[i] += [radians(v) if index == 0 else v]
                    geom[i] = tuple(geom[i])

        Rm = geom['Rm']
        for i1, i2 in [['areas', 'H'], ['H', 'areas']]:
            if i1 not in kwargs:
                geom[i1] = list()
                geom[i2] = kwargs[i2]
                for num, v2 in enumerate(geom[i2]):
                    num2 = num - 1 if num > 0 else 0
                    geom[i1].append(2 * pi * Rm[num2] * v2 if (i1 == 'areas') else v2 / (2 * pi * Rm[num2]))
                geom[i1] = tuple(geom[i1])

        for num, h in enumerate(geom['H']):  # h: 0 1 2 3 4 ... Rm: 0 0 1 2 3
            num2 = num - 1 if num > 0 else 0
            if h/(2*Rm[num2]) > 0.3:
                logging.warning('No se verifica la hipótesis de bidimensionalidad.')
                sys.exit()

        if s == 0.0:
            ap_i_est, ap_i_rot, cuerda = geom['alfap_i_est'], geom['alfap_i_rot'], geom['b']
            geom['s'] = [cuerda[(i // 2) * 2] * 0.4 / ((tan(ap_i_est[i // 2]) + tan(ap_i_rot[i // 2])) *
                                                       (cos(ap_i_rot[i//2])**2)) for i in range(ns*2)]

        for B_S, B_S_name, B_A_name, theta_name in [[B_S_rot, 'alfap_o_rot', 'alfap_i_rot', 'theta_r'],
                                                    [B_S_est, 'alfap_o_est', 'alfap_i_est', 'theta_e']]:
            if B_S is None:
                geom[B_S_name] = [geom[theta_name][i] - geom[B_A_name][i] for i in range(ns)]
            else:
                geom[theta_name] = [geom[B_S_name][i] + geom[B_A_name][i] for i in range(ns)]

        object.__setattr__(self, 'geom', geom)
        return


@dataclass
class gas_model_to_solver:
    """Esta clase se crea para permitir emplear otro módulo alternativo que describa el comportamiento termodinámico
    del gas sin tener que modificar el módulo 'axial_turb_solver.py. Es decir, solo sería necesario modificar esta clase
    de manera adecuada para adaptar un módulo alternativo a "gas_modeling.py"."""

    thermo_mode: str = "ig"  # Cadena que identifica si se establece modelo multifase o ideal para el vapor de agua.
    rel_error: float = 1E-7  # Máximo error relativo que se permite en los cálculos.
    C_atoms: float | int = 12.0  # Átomos de carbono en cada átomo de hidrocarburo en los reactivos.
    H_atoms: float | int = 23.5  # Átomos de hidrógeno en cada átomo de hidrocarburo en los reactivos.
    air_excess: float | int = 4  # Exceso de aire considerado en el ajuste estequiométrico de la combustión completa.
    _memory: list = None
    _gamma: float = None

    def __post_init__(self):
        if self.thermo_mode not in ("ig", "mp", ):
            logger.critical("Los modos a elegir son ig (gas ideal con NASA coefs) o mp ('multi-phase').\n "
                            "El modo mp se aplica solo en el vapor de agua, en caso contrario los límites del cálculo "
                            "disminuirían demasiado y se realentiza todo el cálculo innecesariamente. \n "
                            "Para  el modelo del Çengel usar el módulo IGmodelfromCengel.py. \n")
            sys.exit()
        self.gas_model = mixpm(self.thermo_mode)
        self.gas_model.setmix(self.C_atoms, self.H_atoms, self.air_excess)
        self.gas_model.rel_error = self.rel_error

    def get_prop(self, known_props: dict[str, float], req_prop: dict[str, float] | str):
        if 'h' in known_props or 's' in known_props:
            output_value = self.gas_model.get_props_with_hs(known_props, req_prop)
        else:
            output_value = self.gas_model.get_props_by_Tpd(known_props, req_prop)
        return output_value

    def get_din_visc(self, T: float):
        din_visc = self.gas_model.Wilke_visc_calculator(T)
        return din_visc

    def get_sound_speed(self, T: float, p: float):
        self._gamma, _, _, sound_speed = self.gas_model.get_a(T, p, extra=True)
        self._memory = [T, p].copy()
        return sound_speed

    def get_gamma(self, T: float, p: float):
        if self._gamma is None and self._memory is None:
            _, _, self._gamma = self.gas_model.get_coeffs(T, p)
        elif T == self._memory[0] and p == self._memory[1]:
            pass
        else:
            _, _, self._gamma = self.gas_model.get_coeffs(T, p)
        self._memory = [T, p].copy()
        return self._gamma


# https://youtu.be/KSQ4KxCtsf8
FMT = "[{levelname}]:       {message}       [FILE: {filename}   FUNC: {funcName}   LINE: {lineno}]"
FORMATS = {
    logging.DEBUG: FMT,
    logging.INFO: f"\33[36m{FMT}\33[0m",
    logging.WARNING: f"\33[33m{FMT}\33[0m",
    logging.ERROR: f"\33[31m{FMT}\33[0m",
    logging.CRITICAL: f"\33[1m\33[31m{FMT}\33[0m",
}


class CustomFormatter(logging.Formatter):
    def format(self, record) -> str:
        log_fmt = FORMATS[record.levelno]
        formatter = logging.Formatter(log_fmt, style="{")
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler])

logger = logging.getLogger("coloured-logger")
