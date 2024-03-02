# -*- coding: utf-8 -*-
'''
    Implementation of double description variant of Böklers Outer Approximation Algorithm to enumerate all extreme supported nondominated images
    Assumption: Image set is positive.
    Gives set of supported solutions, including exteme nondominated ones, if solver gives exact solution for non-parametric optimization problem
    If solver routine solver lexicographic variant lexmin f(x,\lambda), F_1(x), \ots, F_k(X), it returns extreme supported nondominated images only
'''
import sys
import numpy as np
from numpy.core.fromnumeric import product
from scipy.spatial import HalfspaceIntersection, ConvexHull
import gurobipy as gp
import datetime

from molib.core.problem import Problem
from molib.core.kdTree import make_kd_tree,get_nearest


TOL = 1e-6

class OuterApproximation_DoubleDescription:
    def __init__(self, solver: Problem, debug_bool = False) -> None:
        self.solver = solver
        if self.solver.solution_quality > 1:
            print('solver is no exact routine!!!')
        
        self.debug_bool = debug_bool
        self.type = solver.type
        self.solver_counter = 0
        self.running_time = 0

        if self.debug_bool:
            if self.type == "min":
                print("We minimize")
            if self.type == "max":
                print("We maximize")

            print("LB: ", self.solver.LB)
            print("UB: ", self.solver.UB)
        
    def __del__(self):
        self.outer_approximation.close()
        self.inner_approximation.close()

    def run(self) -> None:
        start = datetime.datetime.now()
        print('Initialization...')
        if self.debug_bool:
            print('\nSolve initial call to weighted with weight [1/nr_criteria , ..... 1 / nr_criteria]')
        
        y = self.solver.weighted_sum([1 for _ in range(self.solver.nr_obj)])
        self.solver_counter +=1

        self.supported_images = set()
        self.supported_images.add(y)        

        if self.debug_bool:
            print("image returned: ", y)

        ''' initialize outer approximation with initial solution'''
        A = []

        if self.type == "min":
            A.append([-i + y[-1] for i in y[:-1]] + [1] + [-y[-1]])
        if self.type == "max":
            A.append([i - y[-1] for i in y[:-1]] + [-1] + [y[-1]])

         # add that the w's can only add up to 1
        A.append([1 for _ in range(0, self.solver.nr_obj-1)] + [0, -1])

        # add w__i >= 0
        for i in range(0, self.solver.nr_obj-1):
            A.append([0 for _ in range(i)] + [-1] + [0 for _ in range(i+1,self.solver.nr_obj - 1)] + [0,0])
         
        if self.type == "min":
            # add that y >= 0
            A.append([0 for i in range(0, self.solver.nr_obj-1)] + [-1, 0])
        elif self.type == "max":
            # add that y <= UB
            A.append([0 for _ in range(self.solver.nr_obj - 1)] + [1, -self.solver.UB])

        # make A np array
        A = np.array(A)

        if self.type == "min":
            # determine interior point: take y = f(x,w)/2 and w = (1/K,...(1/K))
            interior_point = np.full((self.solver.nr_obj-1), 1/(self.solver.nr_obj))
            interior_point = np.append(interior_point, np.matmul(np.append(interior_point, 0), y)/2)
            interior_point = np.array(interior_point)
        elif self.type == "max":
            # determine interior point: take y = f(x,y) + (UB - f(x,w))/2 and w = (1/K,...(1/K))
            interior_point = np.full((self.solver.nr_obj-1), 1/(self.solver.nr_obj))
            interior_point = np.append(interior_point, np.matmul(np.append(interior_point, 0), y)/2 + self.solver.UB/2)
            interior_point = np.array(interior_point)

        self.outer_approximation = HalfspaceIntersection(A, interior_point, incremental=True)
        
        ''' initalize inner approximation with initial solution -> syipy.spatial class ConvexHull to reduce number of points given to lp solver'''

        if self.type == "min":
            # add extreme points of weight set
            I = [ [0 for _ in range(self.solver.nr_obj)] ]
            for i in range(self.solver.nr_obj -1):
                I.append( [0 for _ in range(i)] + [1] + [0 for _ in range(i+1, self.solver.nr_obj-1)] + [0] )
           
        elif self.type == "max":
            I = [ [0 for _ in range(self.solver.nr_obj - 1)] + [self.solver.UB] ]
            for i in range(self.solver.nr_obj -1):
                I.append( [0 for _ in range(i)] + [1] + [0 for _ in range(i+1, self.solver.nr_obj-1)] + [self.solver.UB] )

        # add point implied by found image y
        I.append([  1/self.solver.nr_obj for _ in range(self.solver.nr_obj -1) ] +  [sum(i for i in y)/( self.solver.nr_obj)] )
        
        self.inner_approximation = ConvexHull(I, incremental=True)

        if self.debug_bool:
            print("outer approximation: ", self.outer_approximation.intersections.tolist())
            print("inner approximation: ", self.inner_approximation.points[self.inner_approximation.vertices].tolist())

        ''' main loop ''' 
        print('Start outer approximation....')

        # get extreme points of outer_approximation as list
        vertices = self.outer_approximation.intersections.tolist()

        # initialize list of investigated weights as kdTree with dimension number_criteria - 1
        self.L = make_kd_tree([[1/self.solver.nr_obj for i in range(0, self.solver.nr_obj -1)]],self.solver.nr_obj-1)

        while len(vertices) > 0: 
            if self.debug_bool:
                print('remaining vertices: ', len(vertices))
            next_vertex = vertices.pop()
            if self.debug_bool:
                print('inverstigate vertex ', next_vertex)

            if next_vertex[-1] < TOL and self.type == "min":
                if self.debug_bool:
                    print('skip vertex because y <= 0')
                continue
            if next_vertex[-1] > self.solver.UB - TOL and self.type == "max":
                if self.debug_bool:
                    print('skip vertex because y >= UB')
                continue
            
            if self.already_checked(next_vertex):
                if self.debug_bool:
                    print('skip vertex because already checked before')
                continue

            # correct vertice
            for i in range(len(next_vertex)-1):
                if next_vertex[i] < 0:
                    next_vertex[i] = 0
                elif next_vertex[i] > 1:
                    next_vertex[i] = 1
            if self.debug_bool:
                print( 'vertex corrected to', next_vertex)

            # check distance to inner approximation
            distance = self.distanzConvexHull(self.inner_approximation.points[self.inner_approximation.vertices] ,  np.array(next_vertex))
            if self.debug_bool:
                print('distance inner to outer approximation at weight next_vertex[:-1]', distance,next_vertex[-1],  abs(distance - next_vertex[-1])  )
        
            if abs(distance - next_vertex[-1]) < TOL:
                if self.debug_bool:
                    print('vertex is optimal, skip vertex')
                #approximation error is alrady achive a vertice -> go to next vertice
                continue

            #solve routine and update outer and inner approximation
            # get new objectives
            if self.debug_bool:
                print('Call of ALG with weight ', next_vertex[:-1] + [1 - sum(i for i in next_vertex[:-1])])
            obj_values = self.solver.weighted_sum( next_vertex[:-1] + [1 - sum(i for i in next_vertex[:-1])]     )
            self.solver_counter +=1

            self.L.append(next_vertex[:-1])

            self.supported_images.add(obj_values)

            if self.debug_bool:
                print('image obtain', obj_values,' with weighted sum ',\
                    sum(next_vertex[i]*obj_values[i] for i in range(len(next_vertex[:-1]))) + (1 - sum(i for i in next_vertex[:-1])) * obj_values[-1] )
            if abs( sum( obj_values[i] *next_vertex[i] for i in range(len(obj_values)-1)) + (1 - sum(j for j in next_vertex[:-1])) * obj_values[-1] - next_vertex[-1]) < TOL:
                if self.debug_bool:
                    print('weighted sum has same objective than current vertice, skip')
                continue
            
            if self.debug_bool:
                print('Add Halfspace of image and Update Inner and Outer Approximation')
            
            # build halfspace and add it to outer_approximation
            halfspace = [  obj_values[-1] - i for i in obj_values[:-1] ] + [1,-obj_values[-1]]
            self.outer_approximation.add_halfspaces([halfspace])
            vertices = self.outer_approximation.intersections.tolist()[1:] # better routine needed to avoid redundant tests!!!

            # update inner_approximation
            self.inner_approximation.add_points(   [[ i for i in next_vertex[:-1]] + [  sum(next_vertex[i]*obj_values[i] for i in range(len(obj_values)-1)) + (1 - sum(i for i in next_vertex[:-1]))*obj_values[-1]   ] ])
            
        
        end = datetime.datetime.now()
        self.running_time=(end-start).total_seconds() * 1000
        print('Terminated!')
        return self.supported_images
    
    def already_checked(self,next_vertex):
        # get nearest weights in L with respect to weighted Tchebycheff distance. If tcheb_dist <= TOL, skip vertex
        distance, _ = get_nearest(self.L, next_vertex[:-1], len(next_vertex)-1, lambda a,b: self.tcheb(a,b))

        if distance <= TOL:
            return True
        else:
            return False

    def tcheb(self,a, b):
            value = max([abs(a[i] - b[i]) for i in range(len(a))])
            return value
 
    def get_WeightSetDecomposition(self):
        verts = self.outer_approximation.intersections
        wsc = {}
        for nd_image in self.supported_images:
            nd_image = nd_image
            y = tuple(nd_image.tolist())

            wsc[y] = []
            for vertex in verts:
                # correct vertice
                for i in range(len(vertex)-1):
                    if vertex[i] < TOL:
                        vertex[i] = 0
                    elif vertex[i] > 1-TOL:
                        vertex[i] = 1            
                if self.debug_bool:
                    print('check if vertex ', vertex, 'is in wsc of ', nd_image)
                
                #calculate weighted sum for extreme points. if w^T ND_image == vertex[-1] for all vertives of the simplex, assign extreme points to wsc
                weighted_sum = sum(vertex[i]*nd_image[i] for i in range(len(vertex)-1)) + (1 - sum(j for j in vertex[:-1])) * nd_image[-1]
                if self.debug_bool:
                    print('weighted sum', weighted_sum)
                    print(abs(weighted_sum - vertex[-1]) < TOL)
                if abs(weighted_sum - vertex[-1]) < TOL:
                    wsc[y].append(vertex[:-1])
            if len(wsc[y]) > 0:
                wsc[y] = self.polar_sort(self.lexsort(np.array(wsc[y])))
        if self.debug_bool:
            for key, item in wsc.items():
                print(key, ':')
                print(item)
        return wsc
        
    def distanzConvexHull(self,points, point):
        # points: np.array of dimension (nr_point, nr_criteria)
        # point: np.array of dimension [nr_criteria,]
        # output distance point, conv(points)

        # print('Calculate distance to inner approximation polytope..........')
        # Create Model
        model = gp.Model('convex hull')
        model.Params.LogToConsole = 0
        # create variable per point in points
        x = model.addVars(points.shape[0], lb = 0, ub = 1, name = 'x')
        # create obj variable
        y = model.addVar(lb = 0, name = 'y')

        # add summ constraint
        model.addConstr( gp.quicksum( x[i] for i in range(len(x))  ) == 1 )

        # add pont constraints
        for j in range(points.shape[1] - 1):
            model.addConstr( gp.quicksum( points[i,j] * x[i] for i in range(len(x)) )  == point[j] )
        model.addConstr( gp.quicksum(points[i,-1] * x[i] for i in range(len(x)) ) == y )

        # set objective
        model.setObjective(y, gp.GRB.MAXIMIZE  )
        model.optimize()

        # print(f"optimal value = {model.objVal:.2f}")
        # for v in model.getVars():
        #     print(f"optimal {v.varName} = {v.x:.2f}")
    
        if model.status == gp.GRB.OPTIMAL:
            # print('Distance sucessfully caluclated ' )
            return model.objVal
        if model.status == gp.GRB.INF_OR_UNBD:
            # print('Unbbounded Solution in Convex Hull Distance Calculation')
            sys.exit()
        elif model.status != gp.GRB.INFEASIBLE:
            # print('Optimization was stopped with status %d' % model.status)
            sys.exit(0) 
    
    def lexsort(self,data): 
        # delete multiple rows                
        sorted_data =  data[np.lexsort(data.T),:]
        row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
        return sorted_data[row_mask]

    def polar_sort(self,points):
        x = points[:,0]
        y = points[:,1]
        x0 = np.mean(x)
        y0 = np.mean(y)

        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        if np.all(r>0):
            angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))
            mask = np.argsort(angles)
            return points[mask]
        else:
            return points

