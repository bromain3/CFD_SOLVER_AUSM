import numpy as np


class node(object):
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)
        self.vector_representation = np.array([x,y])

    def __sub__(self, other):
        if type(other) == node:
            return( vector(self, other) )
        else:
            raise TypeError("You cannot subtract anything else to a node that another node")

    def __add__(self, other):
        if type(other) == node:
            vector = self.vector_representation + other.vector_representation
            return( node(vector[0], vector[1]) )
        else:
            raise TypeError("You cannot add anything else to a node that another node")

    def __truediv__(self, other):
        if type(other) == float or type(other) == int:
            vector = self.vector_representation/other
            return( node(vector[0], vector[1]) )
        else:
            raise TypeError("You cannot divide a node by anything else that an integer")

class vector(object):
    def __init__(self,final_node,initial_node):
        self.final_node, self.initial_node = final_node, initial_node
        self.x = self.final_node.x - self.initial_node.x
        self.y = self.final_node.y - self.initial_node.y
        self.vector_representation = np.array([self.x,self.y])

    def __neg__(self):
        final_node = node(self.initial_node.x - self.x, self.initial_node.y - self.y)
        return(vector(final_node, self.initial_node))

    def norm(self):
        return((self.x**2+self.y**2)**(1/2))

    def get_normalized(self):
        size_x = self.x/self.norm()
        size_y = self.y/self.norm()
        final_node = node(self.initial_node.x + size_x, self.initial_node.y + size_y)
        return(vector(final_node, self.initial_node))
