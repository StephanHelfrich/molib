# -*- coding: utf-8 -*-
import numpy as np

class Image(np.ndarray):
    '''
    Container of np.ndarray such that extra attribute 'solutions' can be assigned
    '''
    def __new__(cls, input_array, solution=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.flags.writeable = False
        # add the new attribute to the created instance
        if solution is not None:
            obj.solutions = [solution]
        else:
            obj.solutions = list()
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.solutions = getattr(obj, 'solutions', None)

    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self,other):
        return bool(np.ndarray.__eq__(self, other).all())

    def add_solution(self,solution):
        self.solutions.append(solution)
