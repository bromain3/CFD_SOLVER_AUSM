from air import *


class cell(object):
    def __init__(self,x0,x1,S0,S1):
        """
        Initilaisation function of the cell class
        :param x0 (float): position of the left side of the cell (the smallest of both x)
        :param x1 (float): position of the right side of the cell (the biggest of both x)
        :param S0 (float): surface of the left side of the cell
        :param S1 (float): surface of the right side of the cell
        """
        self.x0 = x0
        self.x1 = x1
        self.S0 = S0
        self.S1 = S1
        self.dx = x1 - x0
        self.centroid = (x0+x1)/2
        min_S = min(S1, S0)
        max_S = max(S1, S0)
        self.V = min_S * self.dx + (max_S - min_S) * self.dx / 2

class computational_cell(cell):
    def __init__(self,new_cell):
        """
        Initilaisation function of the computational_cell class
        :param new_cell (cell): cell object that will be converted to a computational cell
        """
        cell.__init__(self,new_cell.x0,new_cell.x1,new_cell.S0,new_cell.S1)
        self._rho = None
        self._rho_u = None
        self._rho_et = None
        self._Q = [self.rho, self.rho_u, self.rho_et]

    @property
    def Q(self):
        """
        Getter function for self._Q
        :return self._Q (list) : state vector of the cell
        """
        return (self._Q)
    @Q.setter
    def Q(self, new_value):
        """
        Setter function of Q. It updates all state variables as well as the state vector
        :param new_value (list): new cell state vector
        :return: None
        """
        self._Q = new_value
        self._rho = self._Q[0]
        self._rho_u = self._Q[1]
        self._rho_et = self._Q[2]
    @property
    def rho(self):
        """
        Getter function for self._rho
        :return self._rho (float) : density value of the cell
        """
        return (self._rho)
    @rho.setter
    def rho(self, new_value):
        """
        Setter function of rho. It updates the state vector as well as the state variable
        :param new_value (float): new value for rho
        :return: None
        """
        self._rho = new_value
        self.update_Q()
    @property
    def rho_u(self):
        """
        Getter function for self._rho_u
        :return self._rho_u (float) : x-momentum value of the cell
        """
        return (self._rho_u)
    @rho_u.setter
    def rho_u(self, new_value):
        """
        Setter function of rho_u. It updates the state vector as well as the state variable
        :param new_value (float): new value for rho_u
        :return: None
        """
        self._rho_u = new_value
        self.update_Q()
    @property
    def rho_et(self):
        """
        Getter function for self._rho_et
        :return self._rho_et (float) : energy value of the cell
        """
        return (self._rho_et)
    @rho_et.setter
    def rho_et(self, new_value):
        """
        Setter function of rho_et. It updates the state vector as well as the state variable
        :param new_value (float): new value for rho_et
        :return: None
        """
        self._rho_et = new_value
        self.update_Q()

    def update_Q(self):
        """
        function used to update the state vector with current state variable values
        :return:
        """
        self.Q[0] = self.rho
        self.Q[1] = self.rho_u
        self.Q[2] = self.rho_et


    ######################################################################
    ####################  physical data extraction  ######################
    ######################################################################
    def gam(self):
        """
        Gives the value of gamma (constant as in air.py file for the moment)
        :return gamma (float)
        """
        return (gamma)

    def c(self, T='not_given'):
        """
        Gives the speed of sound in the cell if T is not given.
        It gives the speed of sound the cell would have at a temperature of T if T is given
        :param T (float): If given, it represents the hypothetical temperature with which we want to compute the speed of sound
        :return c (float): Speed of sound
        """
        if T == 'not_given':
            T = self.T()
        return ((self.gam() * r * T)**(1/2))

    def u(self):
        """
        Gives the value of the speed in the cell
        :return u (float): speed of the fluid in the cell
        """
        return (self.rho_u / self.rho)

    def et(self):
        """
        Gives the value of energy in the cell
        :return et (float): energy of the fluid in the cell
        """
        return (self.rho_et / self.rho)

    def P(self):
        """
        Gives the pressure in the fluid
        :return P (float): Pressure
        """
        return ((self.gam() - 1) * self.rho * (self.et() - 0.5 * self.u() ** 2))

    def ht(self):
        """
        Gives the enthalpy in the fluid
        :return ht (float): enthalpy
        """
        return ((self.et() + self.P()) / self.rho)

    def T(self):
        """
        Gives the temperature of the fluid.
        As we work with gaseous state, we give the lowest limit on T at 100K
        :return T (float): temperature limited at a lowest of 100K
        """
        return max((((self.gam() - 1) * (self.rho_et - 1 / 2 * (self.rho_u ** 2) / self.rho)) / r / self.rho),100) # by taking T values no smaller than 100K, We avoid too large values of M (because small c), so it helps convergence a lot. In a second time, we could use this formula for transiant, and after enough step, go back to the normal formula

    def M(self):
        """
        Gives the Mach number of the cell
        :return M (float): Mach number
        """
        return (self.u() / self.c())

    def P0(self):
        """
        Gives the total pressure of the cell
        :return P0 (float): Total pressure
        """
        return (self.P() * (1 + (self.gam() - 1) / 2 * self.M() ** 2) ** (self.gam() / (self.gam() - 1)))
