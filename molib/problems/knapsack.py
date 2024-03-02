# -*- coding: utf-8 -*-
from molib.core.problem import Problem, Image
import numpy as np
import math
import ast
try:
    import gurobipy as gp
except:
    print('gurobi not found')


class Knapsack(Problem):
    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int):
        '''
        profits: ndarray of shape nr_items x nr_obj
        weights: ndarray of shape nr_items,
        capacity: integer
        '''
        super().__init__("max",profits.shape[1])

        self.items = [ (i,weights[i],profits[i,:]) for i in range(len(weights)) ]
        self.capacity = capacity
        
        self.LB = np.min(profits[profits > 0])
        self.UB = len(self.items) * np.max(profits[profits > 0])
        
    def weighted_sum(self, weights: list) -> Image:
        raise NotImplementedError()
    
class Knapsack_Exact(Knapsack):
    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int):
        super().__init__(profits, weights, capacity)
        self.solution_quality_weighted_sum = 1

        self.model = gp.Model('knapsack')
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0


        self.x = self.model.addVars(len(self.items), vtype=gp.GRB.BINARY, name = "x")

        self.model.addConstr( gp.quicksum(self.x[item[0]] * item[1] for item in self.items  ) <= self.capacity)

        self.model.setObjective(gp.quicksum(self.x[item[0]] * item[2][0] for item in self.items  ) , gp.GRB.MAXIMIZE )
    
        self.model.update()

    def weighted_sum(self, weights: list) -> Image:

        self.model.setObjective(gp.quicksum(self.x[item[0]] * sum(item[2][dim] * weights[dim] for dim in range(self.nr_obj)) for item in self.items  ) , gp.GRB.MAXIMIZE )
    
        self.model.update()

        self.model.optimize()


        # get solution
        sol = []
        image = [0 for _ in range(self.nr_obj)]
        

        if self.model.status == gp.GRB.OPTIMAL:
            x = self.model.getVars()
            for item in self.items:
                if x[item[0]].X >= 0.5:
                    sol.append(item[0])
                    for dim in range(self.nr_obj):
                        image[dim] += item[2][dim]
            image = Image(np.array(image),solution=sorted(sol))
            return image
        else:
            print('gurobi has solved knapsack problem not to optimality.')



class Knapsack_Greedy(Knapsack):
    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int):
        super().__init__(profits, weights, capacity)
        self.solution_quality_weighted_sum = 2

    def weighted_sum(self, weights: list) -> Image:

        # sort items in decreasing efficiency (item_profit * weights) / item_weight, lexicographic: efficency, then profit vector
        items = sorted(self.items, key = lambda item: (sum(item[2][dim] * weights[dim] for dim in range(self.nr_obj)) / item[1] , item[2].tolist()), reverse = True)
        
        # add items according to order until capacity is exhausted
        opt_sol = []
        image = [0 for _ in range(self.nr_obj)]
        current_capacity = 0
        for item in items:
            if current_capacity + item[1] <= self.capacity:
                opt_sol.append(item[0])
                for dim in range(self.nr_obj):
                    image[dim] += item[2][dim]
                current_capacity += item[1]

        ## check if a single item has higher profit
        # get best item
        best_item = items[0]
        for item in items[0:]:
            if sum(best_item[2][dim] * weights[dim] for dim in range(self.nr_obj) ) < sum(item[2][dim] * weights[dim] for dim in range(self.nr_obj) ):
                best_item = item

        #check if best item is better than greedy solution
        if sum(best_item[2][dim] * weights[dim] for dim in range(self.nr_obj) ) > sum(image[dim] * weights[dim] for dim in range(self.nr_obj) ):
            opt_sol = [best_item[0]]
            image = best_item[2]
        
        image = Image(np.array(image),solution=sorted(opt_sol))
        return image

def get_knapsack_instance(nr_items: int, nr_obj:int, bounds = [0,1000],seed = None, method = 'uniform'):
    np.random.seed(seed)


    if method == 'uniform':
        profits = np.random.randint(bounds[0],bounds[1],size= [nr_items,nr_obj])
        weights = np.random.randint(max(bounds[0],1),bounds[1],size= nr_items)
        capacity= int(np.sum(weights)/2)

        return profits, weights, capacity

    elif method == 'conflicting' and nr_obj == 3:
        profit1 = np.random.randint(bounds[0],bounds[1],size= [nr_items])
        profit2 = bounds[1] - profit1
        profit3 = []
        for item in range(len(profit1)):
            lb = max(int(0.9*bounds[1]) - int(0.1*bounds[1]) - profit1[item] - profit2[item] , 0)
            ub = min(int(1.1*bounds[1]) - profit1[item] - profit2[item], bounds[1] + 1 - profit1[item])
            if lb > ub:
                ub = lb
            profit3.append(np.random.randint(lb,ub))
        profit3 = np.array(profit3)


        profits = np.column_stack([profit1,profit2,profit3])
        weights = np.random.randint(max(bounds[0],1),bounds[1],size= nr_items)
        capacity= int(np.sum(weights)/2)
        return profits, weights, capacity
    else:
        print('Method for', nr_obj, 'funtions not implemented.')
        return 

def read_knapsack_instance(path: str):
    profits = ""
    with open(path) as file:
        lines = file.readlines()
        capacity = int(lines[2])
        weights = np.array(ast.literal_eval(lines[-1]))

        for line in lines[3:-1]:
            profits = profits + line
        profits = np.asarray(ast.literal_eval(profits))
        return profits.T, weights, capacity

def write_knapsack_instance(profits, weights, capacity, path:str):
    with open(path,'w') as file:
        file.writelines( [str(profits.shape[1]),'\n',
                            str(profits.shape[0]),'\n',
                            str(capacity),'\n',
                            np.array2string(profits.T, separator=',',suppress_small=True,precision = 'int',max_line_width = 40000000000000000000).strip(),'\n',
                            np.array2string(weights.T, separator=',',suppress_small=True,precision = 'int',max_line_width = 40000000000000000000).strip(),
                            ])