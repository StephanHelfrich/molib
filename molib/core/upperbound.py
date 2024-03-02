# -*- coding: utf-8 -*-
import math
from .image import *
from typing import List

class UpperBound:
    "an implementation of an upper bound, cf Klamroth et al."
    def __init__(self, images) -> None:
        self.images = images
        self.dimension = len(images[0].coordinates)

        # calculate coordinates (point-wise maximum)
        self.coordinates = images[0].coordinates.copy()
        for image in images[1:]:
            for i in range(self.dimension):
                if self.coordinates[i] < image.coordinates[i]:
                    self.coordinates[i] = image.coordinates[i]
        
        # determine contributing images
        self.contributing_images = [list() for _ in range(self.dimension)]
        for image_nr, image in enumerate(self.images):
            bool =[False for _ in range(self.dimension)]
            for i in range(self.dimension):
                if image.coordinates[i] >= self.coordinates[i]:
                    self.contributing_images[i].append(image_nr)
                    bool[i] = True
            if sum(bool) == 0: # image does not contribute at all
                raise TypeError(" Image " + str(image_nr) + " is in search space of upper bound")
        del bool


    def __del__(self) -> None:
        pass

class UpperBoundSet:
    "an implementation of an upper bound set, cf. Klamroth et al."
    def __init__(self, LB, UB) -> None:
        """
        LB vector of lower bounds on components of (nondomianted) images
        UB vector of upper bounds on components of (nondominated) images
        """
        self.dimension = len(LB)
        self.LB = LB
        self.UB = UB

        self.images = []
        self.upper_bounds = []

        # add dummy images
        for i in range(self.dimension):
            temp = [self.UB[j] for j in range(self.dimension)]
            temp[i] = self.LB[i]
            temp = Image(temp)
            self.images.append(temp)
        self.upper_bounds.append(UpperBound(self.images))

    def __del__(self) -> None:
        pass

    def __str__(self):
        return "makes print(UpperBoundSet) possible"

    def addImage(self,image: Image) -> None:
        pass


if __name__ == '__main__': 
    LB = [1,2,3]
    UB = [10,15,20]

    upper_bound_set = UpperBoundSet(LB,UB)

    print([image.coordinates for image in upper_bound_set.images])
    print([upper_bound.coordinates for upper_bound in upper_bound_set.upper_bounds])