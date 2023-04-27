"""
Se emplea la librería PyroMat para modelar los productos como mezcla de gases ideales no perfectos y determinar
sus propiedades a partir de las que se proporcionen.
"""

import pyromat as pm
import numpy as np
from math import fabs, sqrt

pm.config['unit_pressure'] = 'Pa'

''' 
El modelo de ig que trae la librería PyroMat alcanza valores más altos de temperatura que el del Çengel.
Todos usan ig2 en este caso (NASA polynomial) 
Referencias a 5.2.1 del Handbook (manual de la librería, pg. 91) y documento NASA_TM_4513. 

Cita de la pg. oficial de PyroMat:
'Computing states follows the state principle which requires that any two properties fully define the state. 
That means that only two properties can be specified at one time.'
'''


class mixpm:
    """ Esta clase define objetos como productos de combustión cuyos componentes se van a modelar con PyroMat"""

    def __init__(self, mode: str, TOL: float = 1E-10):
        """ :param mode: Cadena de caracteres que indica el modo de funcionamiento según si se emplea, o no, modelo de
                         gas ideal en todos los componentes de los productos. Debe ser "ig" o "mp".
            :param TOL: Error máximo relativo que se tolera en el cálculo iterativo para fijar la entalpía o la
                        entropía."""

        self.xmi = None  # xmi son las fracciones másicas de los componentes
        self.xni = None  # xni son las fracciones molares de los componentes
        self.mode = mode
        self._cpmix = None  # Calor específico a presión constante de la mezcla
        self._cvmix = None  # Calor específico a volumen constante de la mezcla
        self._gammamix = None  # Coeficiente adiabático de la mezcla
        self.__Mwtotal = None  # Molar weight de la mezcla
        self.sumprops_Tpd = None  # Almacenaje de la parte o la totalidad de la propiedad que se calcula
        self.x = None  # Átomos de carbono en cada átomo de hidrocarburo en los reactivos
        self.y = None  # Átomos de hidrógeno en cada átomo de hidrocarburo en los reactivos
        self.N = None  # Exceso de aire que se considera en el ajuste estequiométrico de la combustión completa
        self.mu_elem = None
        self.mu_mix = None
        self.rel_error = TOL  # Máximo error relativo permitido.

        ''' En las líneas que siguen se almacenan en una lista atributo de la clase 'mixpm' identificadores de 
            los componentes como objetos de la librería PyroMat según el modelo especificado. '''

        modes = ("ig", "mp", )
        componentes = ("CO2", "H2O", "N2", "O2", )
        self.components = []
        if mode in modes:
            for comp in componentes:
                if comp == "H20":
                    self.components.append(pm.get(f'{mode}.{comp}'))
                else:
                    self.components.append(pm.get(f'ig.{comp}'))
            self.components = tuple(self.components)
        lista_comps_mw = [self.components[i].mw() for i in range(4)]
        self.components_mw = np.array(lista_comps_mw)

    # den es la masa total por kmol de combustible en combustión completa
    def setmix(self, x, y, N):
        """ Se determinan las propiedades del objeto que contienen las fracciones molares (xni) y másicas (xmi) de los
        productos de una combustión completa del hidrocarburo indicado.
                :param x: Átomos de carbono en cada átomo de hidrocarburo en los reactivos.
                :param y: Átomos de hidrógeno en cada átomo de hidrocarburo en los reactivos.
                :param N: Exceso de aire que se considera en el ajuste estequiométrico.
                        :return: No se devuelve nada. """
        self.x, self.y, self.N = np.float64(x), np.float64(y), np.float64(N)

        aux = np.array([x, y/2,  N*(x+(y/4))*(79/21), (N-1)*(x+(y/4))], dtype=float)
        den = np.dot(aux, self.components_mw)
        comp_mass_per_kmol_of_kerosen = self.components_mw * aux

        self.xmi = comp_mass_per_kmol_of_kerosen / den
        self.__Mwtotal = 1 / np.dot(self.xmi / self.components_mw, np.ones(np.shape(self.xmi)))
        self.xni = self.xmi * self.__Mwtotal / self.components_mw

        return

    def get_coeffs(self, T: np.float64, p=np.float64(101_300.0)):
        """ Se calculan los calores específicos y el coeficiente adiabático correspondientes con el estado de la mezcla
            que se determine.
                :param T: Temperatura de los productos (K).
                :param p: Presión total de los productos (Pa).
                        :return: En el mismo orden: Cp, Cv y gamma. Las unidades de los calores
                                específicos son kJ/(kg*K). """

        self._cpmix = np.float64(0.0)
        self._cvmix = np.float64(0.0)

        if self.mode == "mp":
            for i, xm in enumerate(np.nditer(self.xmi)):  # i = 0, 1, 2, 3
                if i == 1:  # cp y cv se reciben como kJ/(kg*K)
                    self._cpmix += xm * self.components[i].cp(T=T, p=p * xm)
                    self._cvmix += xm * self.components[i].cv(T=T, p=p * xm)
                else:
                    self._cpmix += xm * self.components[i].cp(T)
                    self._cvmix += xm * self.components[i].cv(T)
        else:
            for i, xm in enumerate(np.nditer(self.xmi)):
                self._cpmix += xm * self.components[i].cp(T)
                self._cvmix += xm * self.components[i].cv(T)

        self._gammamix = self._cpmix / self._cvmix
        return self._cpmix, self._cvmix, self._gammamix

    def get_a(self, T: np.float64, p=np.float64(101_300.0), extra=False):  # Este cálculo asume gas ideal

        C_p, C_v, gamma = self.get_coeffs(T, p)
        a = np.sqrt(gamma * (C_p - C_v) * 1000 * T)

        if not extra:
            return a
        else:
            return a, C_p, C_v, gamma

    # https://es.stackoverflow.com/questions/383654/llamar-din%C3%A1micamente-a-m%C3%A9todos-dentro-de-una-clase-con-python
    def get_props_by_Tpd(self, known_props: dict, req_prop: str):
        """ Con este método, habiendo determinado el estado de la mezcla empleando una o dos propiedades de entre 'T',
        'p' o 'd' (Temperatura, presión o densidad), se podrá obtener cualquiera que se desconozca junto con 'h' y 's'.
                :param known_props: Un diccionario que debe contener, como clave, el caracter que represente la/s
                                   propiedad/es conocidas y, como valor, el valor de la misma.
                :param req_prop: El caracter que identifique la propiedad que se requiere.
                        :return: El valor de la propiedad que se requiere con las unidades de Pyromat. """

        frac_prop = self.sumprops_Tpd = np.float64(0.0)
        code_get_sumando, xa, xb, xc = '', [], [], []
        k_0 = v_0 = k_1 = v_1 = None
        xj = [xa, xb]
        if self.mode == "ig":
            if req_prop == 'h':
                if len(known_props) == 2:
                    if list(known_props.keys())[1] == 'T':
                        k_0, v_0 = list(known_props.keys())[1], list(known_props.values())[1]

        if v_0 is None:
            k_0, v_0 = list(known_props.keys())[0], list(known_props.values())[0]
        var_list_gr = [req_prop, k_0]

        if not ('T' in known_props and req_prop == 'h' and self.mode == "ig") or self.mode == "mp":
            k_1 = list(known_props.keys())[1]
            v_1 = list(known_props.values())[1]
            var_list_gr.append(k_1)
            xj.append(xc)

        for j in range(len(xj)):
            for i in range(len(self.xmi)):
                exp, xji = 1, 1.0  # Establecido así considerando 'T'
                if var_list_gr[j] == 'p':
                    exp, xji = 1, self.xni[i]  # Ley de Dalton
                elif var_list_gr[j] == 'd':
                    exp, xji = 1, self.xmi[i]  # Masa del componente en el mismo volumen
                elif var_list_gr[j] in ['h', 's']:
                    exp, xji = -1, self.xmi[i]  # Solo interesa j=0, por lo que interpretar únicamente con exp=1
                if j == 0:
                    if var_list_gr[j] in ['p', 'd']:
                        exp = 0  # Suma de las presiones parciales y de las densidades parciales
                    elif var_list_gr[j] in ['h', 's']:
                        exp = exp * (-1)  # De esta manera resultan kg de productos en denominador y la suma es el total
                xj[j].append(xji ** exp)

        for i in range(len(self.xmi)):  # i = 0, 1, 2, 3
            if v_1 is None:
                code_get_sumando = f'self.sumprops_Tpd = xa[i]*self.components[i].{req_prop}({k_0}=' \
                                   f'v_0*xb[i]);'
            else:
                code_get_sumando = f'self.sumprops_Tpd = xa[i]*self.components[i].{req_prop}({k_0}=' \
                                   f'v_0*xb[i], {k_1}=v_1 * xc[i]);'
            exec(code_get_sumando)
            frac_prop += self.sumprops_Tpd
            if req_prop == 'T':
                break

        return frac_prop

    def get_props_with_hs(self, known_props: dict, init_guess: dict):
        """ Con este método y mediante el método numérico régula falsi, se posibilita usar como propiedad conocida la
        entropía o la entalpía de la mezcla de gases, sin embargo, no será posible establecer la otra como output
        requerido, por no poder ser entrada del método 'get_props_with_Tpd'.
                :param known_props: Diccionario que a cada clave, caracter que identifica una propiedad que se conoce,
                                   le asigna el valor de la misma.
                :param init_guess: Diccionario que indica la clave y la estimación inicial de la propiedad a determinar.
                        :return: El valor de la propiedad calculada. """

        k_0, v_0 = list(known_props.keys())[0], list(known_props.values())[0]
        k_1 = v_1 = k_b = v_b = None

        if len(known_props) == 2:
            k_1 = list(known_props.keys())[1]
            v_1 = list(known_props.values())[1]

        if k_0 in ['h', 's']:
            k_a, v_a = k_0, v_0
            if len(known_props) == 2:
                k_b, v_b = k_1, v_1
        else:
            k_a, v_a, k_b, v_b = k_1, v_1, k_0, v_0

        fvg = fabs(2 * v_a * self.rel_error)
        k_g, v_g, vp_g, bolz = list(init_guess.keys())[0], list(init_guess.values())[0], None, 1.0
        v_g1, v_g2, start, f1, f2 = v_g, v_g*1.05, True, None, None

        # Este bloque while es para que se verifique la condición del tma. de Bolzano
        while bolz > 0:
            if 'T' in init_guess and k_a == 'h' and self.mode == "ig":
                f1 = self.get_props_by_Tpd({k_g: v_g1}, k_a) - v_a
                f2 = self.get_props_by_Tpd({k_g: v_g2}, k_a) - v_a
            else:
                f1 = self.get_props_by_Tpd({k_g: v_g1, k_b: v_b}, k_a) - v_a
                f2 = self.get_props_by_Tpd({k_g: v_g2, k_b: v_b}, k_a) - v_a
            bolz = f1 * f2
            if bolz < 0:
                pass
            else:
                v_g1 = v_g1 * 0.999
                v_g2 = v_g2 * 1.001

        c = v_g2 - (f2 * (v_g2 - v_g1) / (f2 - f1))
        if len(known_props) == 2:
            fc = self.get_props_by_Tpd({k_g: c, k_b: v_b}, k_a) - v_a
        else:
            fc = self.get_props_by_Tpd({k_g: c}, k_a) - v_a

        iter_counter = 0

        while fvg > self.rel_error:
            iter_counter += 1
            if iter_counter > 500:
                raise ValueError('Se ha alcanzado el número máximo de iteraciones.')
            if not start:
                if 'T' in init_guess and k_a == 'h' and self.mode == "ig":
                    f1 = self.get_props_by_Tpd({k_g: v_g1}, k_a) - v_a
                    f2 = self.get_props_by_Tpd({k_g: v_g2}, k_a) - v_a
                    c = v_g2 - (f2 * (v_g2 - v_g1) / (f2 - f1))
                    fc = self.get_props_by_Tpd({k_g: c}, k_a) - v_a
                else:
                    f1 = self.get_props_by_Tpd({k_g: v_g1, k_b: v_b}, k_a) - v_a
                    f2 = self.get_props_by_Tpd({k_g: v_g2, k_b: v_b}, k_a) - v_a
                    c = v_g2 - (f2 * (v_g2 - v_g1) / (f2 - f1))
                    fc = self.get_props_by_Tpd({k_g: c, k_b: v_b}, k_a) - v_a
            else:
                start = False
            if fc * f2 <= 0:
                v_g1 = c
            elif fc * f1 <= 0:
                v_g2 = c
            fvg = fabs(fc / v_a)

        return c

    def getTsat(self, p: np.float64):
        H2O = pm.get('mp.H2O')
        Tsat = H2O.T(p=p * self.xni[1], x=0.00001)
        return Tsat  # K

    @staticmethod
    def getpsat(T: np.float64):
        H2O = pm.get('mp.H2O')
        psat = H2O.p(T=T, x=0.00001)
        return psat  # Pa

    # https://efrainpuerto.com/tag/sutherland/
    # https://www.youtube.com/watch?v=tTG5Re5G7B8&ab_channel=Petr%C3%B3leoyprogramaci%C3%B3nSMAE
    # https://doc.comsol.com/5.5/doc/com.comsol.help.cfd/cfd_ug_fluidflow_high_mach.08.27.html
    # https://repositorio.upct.es/bitstream/handle/10317/6091/tfm-mu%C3%B1-est.pdf?sequence=1 (Wilke)
    def Wilke_visc_calculator(self, T: np.float64):

        def Sutherlands_law(comp: str):
            mu_0 = T_0 = S_mu = None
            if comp == 'CO2':
                mu_0, T_0, S_mu = 1.370E-5, 273, 222
            elif comp == 'N2':
                mu_0, T_0, S_mu = 1.663E-5, 273, 107
            elif comp == 'O2':
                mu_0, T_0, S_mu = 1.919E-5, 273, 139
            mu = mu_0 * ((T / T_0) ** (3 / 2)) * (T_0 + S_mu) / (T + S_mu)
            return mu

        def Wilke_mix_rule():
            den = []
            for i in range(len(self.components)):
                den_i = np.float(0.0)
                for j, xn_j in enumerate(np.nditer(self.xni)):
                    mw_rel = self.components_mw[i] / self.components_mw[j]
                    op = xn_j * (1 / np.sqrt(8)) * ((1 + mw_rel) ** (-0.5))
                    den_i += op * ((1 + (((self.mu_elem[i] / self.mu_elem[j]) ** 0.5) * (mw_rel ** 0.25))) ** 2)
                den.append(den_i)
            den = np.array(den, dtype=float)
            eta_frac = np.dot(self.xni, self.mu_elem / den)
            return eta_frac

        frac = 647.096 / T
        mu_w = sqrt(1 / frac) / (0.0167752 + 0.0220462 * frac + 0.0063666 * (frac ** 2) - 0.0024161 * (frac ** 3))
        mu_CO2, mu_N2, mu_O2 = Sutherlands_law('CO2'), Sutherlands_law('N2'), Sutherlands_law('O2')
        self.mu_elem = np.array([mu_CO2, mu_w, mu_N2, mu_O2], dtype=float)
        self.mu_mix = Wilke_mix_rule()

        return self.mu_mix


def main():

    mezcla = mixpm("ig", 1E-9)
    mezcla.setmix(12.0, 23.5, 2)

    press = float(mezcla.get_props_by_Tpd({'T': 1800, 'd': 2}, 'p'))
    print(f'Resulta una presión de {(press / 101300):.7f} atm')
    enthalpy = float(mezcla.get_props_by_Tpd({'T': 1800, 'd': 2}, 'h'))
    print(f'Resulta una entalpía de {enthalpy:.7f} kJ/kg')
    entropy = float(mezcla.get_props_by_Tpd({'T': 1800, 'd': 2}, 's'))
    print(f'Resulta una entropía de {entropy:.7f} kJ/kgK')
    pr = float(mezcla.get_props_with_hs({'s': entropy, 'T': 1800}, {'p': 10 * 101300}))
    print(f'Resulta una presión de {(pr / 101300):.7f} atm')
    Temperature = float(mezcla.get_props_with_hs({'h': enthalpy}, {'T': 1500}))
    print(f'Resulta una temperatura de {Temperature:.7f} K')
    entropy2 = float(mezcla.get_props_by_Tpd({'T': Temperature, 'p': press}, 's'))
    print(f'Resulta una entropía de {entropy2:.7f} kJ/kgK')

    return


if __name__ == '__main__':
    main()
