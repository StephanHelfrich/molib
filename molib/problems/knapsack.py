# -*- coding: utf-8 -*-
from molib.core.problem import Problem, Image
import numpy as np
import math
import ast


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
    """Exact weighted sum algorithm by dynamic programming over the capacity.

    Among all optimal solutions of the weighted sum scalarization, a solution whose image is
    lexicographically maximal is returned, so the image is nondominated also for weights with a
    zero entry. Requires integer item weights and capacity. Running time O(n * capacity * nr_obj).
    """

    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int):
        super().__init__(profits, weights, capacity)
        self.solution_quality_weighted_sum = 1

        self._weights = np.asarray(weights)
        if not np.all(np.equal(np.mod(self._weights, 1), 0)) or int(capacity) != capacity:
            raise ValueError("Knapsack_Exact requires integer item weights and capacity")
        self._weights = self._weights.astype(int)
        self._profits = np.asarray(profits, dtype=float)

    def weighted_sum(self, weights: list) -> Image:
        weights = np.asarray(weights, dtype=float)
        n, C = len(self.items), int(self.capacity)
        values = self._profits @ weights

        # best[k, c]: lexicographically maximal (weighted value, f_1, ..., f_p) with capacity c,
        # k = 0 is the weighted value and k = 1, ..., p are the objectives
        keys = np.vstack([values[:, None].T, self._profits.T])          # (p + 1, n)
        best = np.zeros((keys.shape[0], C + 1))
        take = np.zeros((n, C + 1), dtype=bool)
        tol = 1e-9 * max(1.0, float(np.abs(values).sum()))

        for i in range(n):
            w = self._weights[i]
            if w > C:
                continue
            candidate = best[:, :C + 1 - w] + keys[:, i:i + 1]             # states c = w, ..., C
            current = best[:, w:]
            better = self._lex_greater(candidate, current, tol)
            take[i, w:] = better
            current[:, better] = candidate[:, better]

        # reconstruct the solution
        solution, c = [], C
        for i in range(n - 1, -1, -1):
            if take[i, c]:
                solution.append(i)
                c -= self._weights[i]
        image = self._profits[solution].sum(axis=0) if solution else np.zeros(self.nr_obj)
        return Image(image, solution=sorted(solution))

    @staticmethod
    def _lex_greater(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
        """Columnwise lexicographic comparison a > b (first row with tolerance)."""
        greater = a[0] > b[0] + tol
        undecided = np.abs(a[0] - b[0]) <= tol
        for k in range(1, a.shape[0]):
            greater |= undecided & (a[k] > b[k])
            undecided &= a[k] == b[k]
        return greater



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
                            np.array2string(profits.T, separator=',',suppress_small=True,precision = 0,max_line_width = 40000000000000000000).strip(),'\n',
                            np.array2string(weights.T, separator=',',suppress_small=True,precision = 0,max_line_width = 40000000000000000000).strip(),
                            ])