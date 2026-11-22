# -*- coding: utf-8 -*-
"""
==============
Scalarizations
==============

Single-objective substitute problems of a multiobjective optimization problem. A scalarization
object only holds its parameters and knows its value and feasibility in the image space. How
a scalarization is solved for a concrete problem is decided by ``molib.core.solve``.
"""
from .base import *
from .weighted_sum import *
from .weighted_tchebycheff import *
from .epsilon_constraint import *