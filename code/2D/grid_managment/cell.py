import numpy as np
from air import *
import geometry


class cell(object):
    def __init__(self,lower_left_node,lower_right_node,upper_left_node,upper_right_node):
        self.lower_left_node = lower_left_node
        self.lower_right_node = lower_right_node
        self.upper_left_node = upper_left_node
        self.upper_right_node = upper_right_node
        self.Sbottom, self.Sleft, self.Stop, self.Sright = self.getSurfaces()
        self.V = self.getVolume()
        self.centroid = (self.lower_left_node + self.lower_right_node + self.upper_left_node + self.upper_right_node)/4

    def getSurfaces(self):
        """
        returns faces area vectors of a 2d cells given the nodes of the cell
        :param nodes: (list) of the cell nodes (np.array 1D) numbered as in (AIAA J, 21, 1983)
        :return: cell area vectors
        """
        r12 = self.lower_right_node - self.lower_left_node  # computing cells' sides lengths
        r23 = self.upper_right_node - self.lower_right_node  # computing cells' sides lengths
        r34 = self.upper_left_node - self.upper_right_node  # computing cells' sides lengths
        r41 = self.lower_left_node - self.upper_left_node  # computing cells' sides lengths
        Sbottom_beginig_point = (self.lower_right_node + self.lower_left_node)/2
        Sbottom = geometry.vector(geometry.node(Sbottom_beginig_point.x + r12.y, Sbottom_beginig_point.y - r12.x), Sbottom_beginig_point) # creating the surface vector perpendicular to the cell face
        Sright_beginig_point = (self.lower_right_node + self.upper_right_node) / 2
        Sright = geometry.vector(geometry.node(Sright_beginig_point.x + r23.y, Sright_beginig_point.y - r23.x), Sright_beginig_point)  # creating the surface vector perpendicular to the cell face
        Stop_beginig_point = (self.upper_left_node + self.upper_right_node) / 2
        Stop = geometry.vector(geometry.node(Stop_beginig_point.x + r34.y, Stop_beginig_point.y - r34.x), Stop_beginig_point)  # creating the surface vector perpendicular to the cell face
        Sleft_beginig_point = (self.upper_left_node + self.lower_left_node) / 2
        Sleft = geometry.vector(geometry.node(Sleft_beginig_point.x + r41.y, Sleft_beginig_point.y - r41.x), Sleft_beginig_point)  # creating the surface vector perpendicular to the cell face
        return (Sbottom, Sleft, Stop, Sright)

    def getVolume(self):
        """
        returns volume of a 2d cells given the nodes of the cell
        :param nodes: (list) of the cell nodes (np.array 1D) numbered as in (AIAA J, 21, 1983)
        :return: cell volume
        """
        r13 = self.upper_right_node - self.lower_left_node
        r24 = self.upper_left_node - self.lower_right_node
        matrix = np.vstack((r13.vector_representation,r24.vector_representation))
        res = 0.5 * np.linalg.det(matrix)
        return (res)

    def n_xi_plus_half(self):
        """
        Extrapolate the value of xi and eta in the middle of vertical surfaces
        :param Sb: Surface vectors of surfaces parallel to xi of the augmented grid (3D-array)
        :param Sl: Surface vectors of surfaces parallel to eta of the augmented grid (3D-array)
        :return: interpolated direction vectors components (2D-array)
        """
        n_xi = self.Sright.get_normalized()
        return(n_xi)

    def n_xi_minus_half(self):
        """
        Extrapolate the value of xi and eta in the middle of vertical surfaces
        :param Sb: Surface vectors of surfaces parallel to xi of the augmented grid (3D-array)
        :param Sl: Surface vectors of surfaces parallel to eta of the augmented grid (3D-array)
        :return: interpolated direction vectors components (2D-array)
        """
        n_xi = self.Sleft.get_normalized()
        return(n_xi)

    def n_eta_plus_half(self):
        """
        Extrapolate the value of xi and eta in the middle of horizontal surfaces
        :param Sb: Surface vectors of surfaces parallel to xi of the augmented grid (3D-array)
        :param Sl: Surface vectors of surfaces parallel to eta of the augmented grid (3D-array)
        :return: interpolated direction vectors components (2D-array)
        """
        n_eta = self.Stop.get_normalized()
        return(n_eta)

    def n_eta_minus_half(self):
        """
        Extrapolate the value of xi and eta in the middle of horizontal surfaces
        :param Sb: Surface vectors of surfaces parallel to xi of the augmented grid (3D-array)
        :param Sl: Surface vectors of surfaces parallel to eta of the augmented grid (3D-array)
        :return: interpolated direction vectors components (2D-array)
        """
        n_eta = self.Sbottom.get_normalized()
        return(n_eta)



class computational_cell(cell):
    def __init__(self,geometric_cell):
        if type(geometric_cell) == cell:
            cell.__init__(self,geometric_cell.lower_left_node,geometric_cell.lower_right_node,geometric_cell.upper_left_node,geometric_cell.upper_right_node)
            self._rho = None
            self._rho_u = None
            self._rho_v = None
            self._rho_et = None
            self._Q = np.array([self.rho, self.rho_u, self.rho_v, self.rho_et])
        else:
            raise ValueError("You cannot define a computational cell by any other way than giving a geometric cell")

    @property
    def Q(self):
        return(self._Q)
    @Q.setter
    def Q(self,new_value):
        self._Q = new_value
        self._rho = self._Q[0]
        self._rho_u = self._Q[1]
        self._rho_v = self._Q[2]
        self._rho_et = self._Q[3]
    @property
    def rho(self):
        return(self._rho)
    @rho.setter
    def rho(self,new_value):
        self._rho = new_value
        self.update_Q()
    @property
    def rho_u(self):
        return(self._rho_u)
    @rho_u.setter
    def rho_u(self,new_value):
        self._rho_u = new_value
        self.update_Q()
    @property
    def rho_v(self):
        return(self._rho_v)
    @rho_v.setter
    def rho_v(self,new_value):
        self._rho_v = new_value
        self.update_Q()
    @property
    def rho_et(self):
        return(self._rho_et)
    @rho_et.setter
    def rho_et(self,new_value):
        self._rho_et = new_value
        self.update_Q()

    def update_Q(self):
        self.Q[0] = self.rho
        self.Q[1] = self.rho_u
        self.Q[2] = self.rho_v
        self.Q[3] = self.rho_et

    ######################################################################
    ####################  physical data extraction  ######################
    ######################################################################

    def gam(self):
        """
        Returns the value of gamma
        """
        return(gamma)

    def c(self, T='not_given'):
        """
        Returns the value of speed of sound
        :param g: gamma at the points considered (float or 2D-array)
        :param T: temperature at a given point (float or 2D-array)
        :return: speed of sound (float or 2D-array)
        """
        if T == 'not_given':
            T = self.T()
        return (np.sqrt(self.gam() * r * T))

    def u(self):
        """
        Extract the value of u form Q
        :param Q: array of state vectors (3D-array)
        :return: array containing the value of u (2D-array)
        """
        return (self.rho_u / self.rho)

    def v(self):
        """
        Extract the value of v form Q
        :param Q: array of state vectors (3D-array)
        :return: array containing the value of v (2D-array)
        """
        return (self.rho_v / self.rho)

    def et(self):
        """
        Extract the value of et form Q
        :param Q: array of state vectors (3D-array)
        :return: array containing the value of et (2D-array)
        """
        return (self.rho_et / self.rho)

    def P(self):
        """
        Extract the value of P form Q
        :param Q: array of state vectors (3D-array)
        :return: array containing the value of P (2D-array)
        """
        return ((self.gam() - 1) * self.rho * (self.et() - 0.5 * (self.u() ** 2 + self.v() ** 2)))

    def ht(self):
        """
        Extract the value of ht form Q
        :param Q: array of state vectors (3D-array)
        :return: array containing the value of ht (2D-array)
        """
        return (self.et() + self.P() / self.rho)

    def T(self):
        """
        Extract the value of T form Q
        :param Q: array of state vectors (3D-array)
        :return: array containing the value of T (2D-array)
        """
        return max((((self.gam() - 1) * (self.rho_et - 1 / 2 * (self.rho_u ** 2) / self.rho)) / r / self.rho),100) # by taking T values no smaller than 100K, We avoid too large values of M (because small c), so it helps convergence a lot. In a second time, we could use this formula for transiant, and after enough step, go back to the normal formula

    def M(self):
        return np.sqrt(self.u() ** 2 + self.v() ** 2) / self.c()

    def P0(self):
        return (self.P() * (1 + (self.gam() - 1) / 2 * self.M() ** 2) ** (self.gam() / (self.gam() - 1)))


if __name__ == '__main__':
    lower_left_node = geometry.node(0, 0)
    lower_right_node = geometry.node(1, 0)
    upper_left_node = geometry.node(0, 1)
    upper_right_node = geometry.node(1, 1)
    gcell = cell(lower_left_node,lower_right_node,upper_left_node,upper_right_node)
    ccell = computational_cell(gcell)
