"""
En este módulo se define una clase que caracteriza un objeto que contiene la configuración y los parámetros geométricos
que se emplearán en el cálculo (config_parameters). Se emplea un módulo aparte para esta clase para evitar importaciones
circulares, ya que se va a almacenar información que será necesaria en varios módulos. Además, también se crea una
clase que facilita el intercambio del módulo "gas_model.py" por otro similar.
"""

import logging  # https://docs.python.org/es/3/howto/logging.html
from gas_model import mixpm
import pyromat as pm
import sys
from math import pi, radians, cos, tan
from dataclasses import dataclass


@dataclass(frozen=True)
class config_parameters:
    """ Objeto que agrupa los parámetros necesarios para configurar la ejecución del solver. """

    TOL: float  # Máximo error relativo que se tolera en los cálculos iterativos del solver
    n_steps: int  # Número de escalonamientos que se definen
    fast_mode: bool  # Limitar cálculos y determinar temperatura, presión y velocidad a la salida
    loss_model: str  # Cadena identificador del modelo de pérdidas establecido
    ideal_gas: bool  # True cuando se establece hipótesis de gas ideal
    STEP_DEC_TOL: float = 1E-7  # Máximo error relativo que se tolera en la corrección de reynolds
    SOLVER_DEC_TOL: float = 1E-6  # Máxima desviación relativa que se tolera en la presión a la salida
    geom: dict = None  # Diccionario para almacenar parámetros geométricos de la turbina
    iter_limit: int = 600  # Límite de iteraciones que se establece
    relative_jump: float = 0.1  # Salto relativo durante la búsqueda cuando se conoce la presión a la salida
    accurate_approach: bool = False  # Opción recomendable cuando el salto de presión es alto.

    def __post_init__(self):
        if self.loss_model not in ['Aungier', 'Ainley_and_Mathieson']:
            logging.critical('Los identificadores de los modelos de pérdidas disponibles son "Aungier" y '
                             '"Ainley_and_Mathieson".')
            sys.exit()

    def set_geometry(self, B_A_est: int | float | list, B_A_rot: int | float | list, cuerda: int | float | list,
                     radio_medio: int | float | list, B_S_est: int | float | list = None,
                     B_S_rot: int | float | list = None, theta_est: int | float | list = None,
                     theta_rot: int | float | list = None, **kwargs: int | float | list[float]):
        """ Se establece la geometría de la turbina del problema. Se debe indicar el borde de salida o la deflexión de
        cada álabe. Algunos parámetros siguen el mismo razonamiento de alternancia, por cuestión de flexibilidad.

        Los valores de los parámetros son los que corresponden al radio de referencia.

        Los parámetros pueden indicarse en forma de lista o en forma de un único número, entero o de coma flotante, en
        el caso de que se repita para cada escalonamiento.

        Para mantener uniformidad los parámetros se almacenan en un diccionario como tuplas.

        Emplear sistema internacional excepto para los ángulos, que se deben indicar en grados sexagecimales.

                :param cuerda: Cuerda de cada álabe.
                :param radio_medio: Radio medio de un álabe de escalonamiento.
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
                               Las claves son: Rm (radio medio), H (alturas), areas, b_z (cuerda axial),
                               t_max (espesor máximo de un álabe), r_r (radio raiz), r_c (radio cabeza),
                               t_e (espesor del borde de salida), K (holgura en el extremo del álabe),
                               s (paso, distancia entre álabes de la misma corona),
                               o (distancia de la garganta, "Blade opening"),
                               delta (espacio entre la punta del álabe y la pared final),
                               e (radio medio de curvatura entre la garganta y el borde de estela),
                               holgura_radial (True o False, según si existe espacio libre en la punta del álabe)."""

        ns, geom = self.n_steps, dict()
        local_dict1 = {'alfap_i_est': B_A_est, 'alfap_i_rot': B_A_rot}

        for B_S, B_S_name, theta, theta_name in [[B_S_rot, 'alfap_o_rot', theta_rot, 'theta_r'],
                                                 [B_S_est, 'alfap_o_est', theta_est, 'theta_e']]:
            if B_S is not None:
                local_dict1[B_S_name] = B_S
            elif theta_rot is not None:
                local_dict1[theta_name] = theta
            else:
                registro.critical('Se precisa definir de alguna manera la deflexión de cada álabe.')
                sys.exit()

        list_items2, local_dict2 = ['t_max', 'r_r', 'r_c', 't_e', ], {}
        for key in ['roughness_ptv', 'lashing_wires', 'wire_diameter', 'b_z', 'delta', 'k', ]:
            if key in kwargs:
                list_items2.append(key)
        if 'o' in kwargs and 'e' in kwargs:
            list_items2.append('o')
            list_items2.append('e')
        for par_id in list_items2:
            local_dict2[par_id] = kwargs.get(par_id, 0.0)
        local_dict2['b'], local_dict2['Rm'] = cuerda, radio_medio
        s = kwargs.get('s', 0.0)
        if not s == 0.0:
            local_dict2['s'] = s

        ld_list = [local_dict1, local_dict2]
        for index, ld in enumerate(ld_list):
            for i, v in ld.items():
                if not isinstance(v, (int, float)):
                    if index == 0:
                        v = [radians(elem) for elem in v]
                    geom[i] = tuple(v)
                else:
                    geom[i] = []
                    for _ in range(ns if index == 0 else ns * 2):
                        geom[i] += [radians(v) if index == 0 else v]
                    geom[i] = tuple(geom[i])

        Rm = geom['Rm']
        for i1, i2 in [['areas', 'H'], ['H', 'areas']]:
            if i1 not in kwargs:
                geom[i1] = list()
                geom[i2] = tuple(kwargs[i2])
                for num, v2 in enumerate(geom[i2]):
                    num2 = num - 1 if num > 0 else 0
                    geom[i1].append(2 * pi * Rm[num2] * v2 if (i1 == 'areas') else v2 / (2 * pi * Rm[num2]))
                geom[i1] = tuple(geom[i1])

        if 'areas' not in geom:
            geom['areas'], geom['H'] = kwargs['areas'], kwargs['H']

        for num, h in enumerate(geom['H']):   # h: 0 1 2 3 4 ... Rm: 0 0 1 2 3
            num2 = num - 1 if num > 0 else 0
            if (2*Rm[num2]-h)/(2*Rm[num2]-h) > 1.4:
                registro.critical('No se verifica la hipótesis de bidimensionalidad.')
                sys.exit()

        if s == 0.0:
            ap_i_est, ap_i_rot, cuerda = geom['alfap_i_est'], geom['alfap_i_rot'], geom['b']
            geom['s'] = [cuerda[(i // 2) * 2] * 0.4 / ((tan(ap_i_est[i // 2]) + tan(ap_i_rot[i // 2])) *
                                                       (cos(ap_i_rot[i//2])**2)) for i in range(ns*2)]

        for B_S, B_S_name, B_A_name, theta_name in [[B_S_rot, 'alfap_o_rot', 'alfap_i_rot', 'theta_r'],
                                                    [B_S_est, 'alfap_o_est', 'alfap_i_est', 'theta_e']]:
            if B_S is None:
                geom[B_S_name] = [geom[theta_name][i] - geom[B_A_name][i] for i in range(ns)]
                geom[B_S_name] = tuple(geom[B_S_name])
            else:
                geom[theta_name] = [geom[B_S_name][i] + geom[B_A_name][i] for i in range(ns)]

        if 'holgura_radial' in kwargs:
            geom['X'] = 1.35 if kwargs['holgura_radial'] else 0.7

        object.__setattr__(self, 'geom', geom)
        return

    def edit_geom(self, key, values: tuple):
        """ Se permite modificar parámetros ya introducidos mediante este método. """

        geom = self.geom
        geom[key] = values
        object.__setattr__(self, 'geom', geom)
        return

    def edit_tol(self, key, new_tol):
        object.__setattr__(self, key, new_tol)

    def edit_relative_jump(self, new_rel_jump):
        object.__setattr__(self, 'relative_jump', new_rel_jump)


class GasLibraryAdaptedException(Exception):
    """ Excepción creada para identificar errores provenientes del módulo para modelar el gas."""
    def __init__(self, msg="Error proveniente de la librería que modela las propiedades."):
        self.msg = msg
        super().__init__(self.msg)


class NonConvergenceError(Exception):
    """ Excepción creada para identificar errores de no convergencia."""
    def __init__(self, msg="Error de no convergencia."):
        self.msg = msg
        super().__init__(self.msg)


@dataclass
class gas_model_to_solver:
    """Esta clase se crea para permitir emplear otro módulo alternativo que describa el comportamiento termodinámico
    del gas sin tener que modificar el módulo 'axial_turb_solver.py. La finalidad de hacer esto es mejorar la
    modularidad, es decir, solo sería necesario modificar esta clase de manera adecuada para implementar un módulo
    alternativo a "gas_model.py"."""

    thermo_mode: str = "ig"  # Cadena que identifica si se establece modelo multifase o ideal para el vapor de agua.
    relative_error: float = 1E-7  # Máximo error relativo que se permite en los cálculos.
    C_atoms: float | int = 12.0  # Átomos de carbono en cada átomo de hidrocarburo en los reactivos.
    H_atoms: float | int = 23.5  # Átomos de hidrógeno en cada átomo de hidrocarburo en los reactivos.
    air_excess: float | int = 4  # Exceso de aire considerado en el ajuste estequiométrico de la combustión completa.
    _memory: list = None  # La finalidad de esta memoria es no recalcular las propiedades si ya se habían determinado.
    _gamma: float = None

    def __post_init__(self):
        if self.thermo_mode not in ("ig", "mp", ):
            registro.critical("Los modos a elegir son ig (gas ideal con NASA coefs) o mp ('multi-phase').\n "
                              "El modo mp se aplica solo en el vapor de agua, en caso contrario los límites del "
                              "cálculo disminuirían demasiado y se realentiza todo el cálculo innecesariamente. \n "
                              "Para  el modelo del Çengel usar el módulo IGmodelfromCengel.py. \n")
            sys.exit()
        self.gas_model = mixpm(self.thermo_mode)
        self.gas_model.setmix(self.C_atoms, self.H_atoms, self.air_excess)
        self.gas_model.rel_error = self.relative_error

    def get_tol(self):
        return self.gas_model.rel_error

    def modify_tol(self, tol):
        self.gas_model.rel_error = tol
        return

    def get_prop(self, known_props: dict[str, float], req_prop: dict[str, float] | str):
        try:
            if 'h' in known_props or 's' in known_props:
                output_value = self.gas_model.get_props_with_hs(known_props, req_prop)
            else:
                output_value = self.gas_model.get_props_by_Tpd(known_props, req_prop)
        except pm.utility.PMParamError:
            raise GasLibraryAdaptedException
        return output_value

    def get_din_visc(self, T: float):   # Se emplea solo en el cálculo del número de Reynolds.
        din_visc = self.gas_model.Wilke_visc_calculator(T)
        return din_visc

    def get_sound_speed(self, T: float, p: float):
        sound_speed, _, _, self._gamma = self.gas_model.get_a(T, p, extra=True)
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


# Las líneas que siguen son orientadas a la customización del mensaje de salida.
# Para omitir estos mensajes, establecer el argumento level en un nivel superior al tipo de mensaje a omitir.
# https://youtu.be/KSQ4KxCtsf8
FMT = "[{levelname}]:  ...  {message}  ...  [FILE: {filename}   FUNC: {funcName}   LINE: {lineno}]"

FORMATS = {
    logging.DEBUG: f"\33[36m{FMT}\33[0m",
    logging.INFO: f"\33[92m{FMT}\33[0m",
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

registro = logging.getLogger("coloured-registro")
