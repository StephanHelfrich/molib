from molib.core import Problem, Image
import numpy as np
import itertools
import networkx as nx
from collections import deque,defaultdict

try:
    import gurobipy as gp
except:
    print('gurobi not found')

class TravelingSalesman(Problem):
    def __init__(self, C: np.ndarray):
        '''
        C : ndarray of shape nr_cities x nr_cities x nr_obj specifying the distance matrices
        '''        
        super().__init__("min", C.shape[2])

        self.C = C
        self.LB = np.min(self.C[self.C > 0])
        self.UB = C.shape[0] * np.max(self.C[self.C > 0])

    def weighted_sum(self,weights:np.ndarray):
        raise NotImplementedError()
    

class TravelingSalesman_Gurobi(TravelingSalesman):
    def __init__(self, C: np.ndarray):
        
        # code adapted from
        # https://colab.research.google.com/github/Gurobi/modeling-examples/blob/master/traveling_salesman/tsp_gcl.ipynb
        
        super().__init__(C)
        self.solution_quality_weighted_sum = 1


        self.nr_cities = self.C.shape[0]
        self.dist = { (c1,c2) : C[c1,c2,:] for c1, c2 in itertools.combinations(range(self.nr_cities),2) }


        self.model = gp.Model('TSP')
        self.model.Params.LogToConsole = 0
        self.model.Params.OutputFlag = 0
        self.model.Params.lazyConstraints = 1
        

        ### variables
        # add variable for each pair of city
        self.vars = self.model.addVars(self.dist.keys(), vtype=gp.GRB.BINARY, name = "x")

        # Symmetric direction: Copy the object
        for i, j in self.vars.keys():
            self.vars[j, i] = self.vars[i, j]  # edge in opposite direction
        
        
        # Constraints: two edges incident to each city
        self.cons = self.model.addConstrs(self.vars.sum(c, '*') == 2 for c in range(self.nr_cities))

        # dummy objecitve
        self.model.setObjective(gp.quicksum(self.vars[i,j] * self.dist[i,j][0]  for i,j in itertools.combinations(range(self.nr_cities),2)) , gp.GRB.MINIMIZE)

    def weighted_sum(self, weights: np.ndarray):        
        self.model.setObjective(gp.quicksum( self.vars[i,j] * np.matmul(self.dist[i,j],weights)  for i,j in itertools.combinations(range(self.nr_cities),2)) , gp.GRB.MINIMIZE)
        self.model.update()
        self.model.reset()

        def subtourelim(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                # make a list of edges selected in the solution
                vals = model.cbGetSolution(model._vars)
                selected = gp.tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
                # find the shortest cycle in the selected edge list
                tour = subtour(selected)
                if len(tour) < self.nr_cities:
                    # add subtour elimination constr. for every pair of cities in subtour
                    model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in itertools.combinations(tour, 2))
                                <= len(tour)-1)

        def subtour(edges):
            unvisited = list(range(self.nr_cities))
            cycle = list(range(self.nr_cities)) # Dummy - guaranteed to be replaced
            while unvisited:  # true if list is non-empty
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j in edges.select(current, '*')
                                if j in unvisited]
                if len(thiscycle) <= len(cycle):
                    cycle = thiscycle # New shortest subtour
            return cycle

        self.model._vars = self.vars

        self.model.optimize(subtourelim)


        #retrieve solution
        vals = self.model.getAttr('x', self.vars)
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        tour = subtour(selected)
        
        assert len(tour) == self.nr_cities

        tour.append(tour[0])

        image = [0 for _ in range(self.nr_obj)]
        for i in range(len(tour) -1) :
            for dim in range(self.nr_obj):
                image[dim] += self.C[tour[i],tour[i+1],dim]
  
        image = Image(image, solution = tour)
        return image



class TravelingSalesman_DoubleTree(TravelingSalesman):
    def __init__(self, C: np.ndarray):
        super().__init__(C)

        self.solution_quality_weighted_sum = 2 

    def weighted_sum(self, weights: np.ndarray):
 
        A = np.matmul(self.C, weights)
 
        #Bestimme MST
        T = self.kruskal(A)

        #konvertiere MST in Adjazenzlistendarstellung:
        Adjazenzlisten = defaultdict(list)
        for [u,v] in T:
            Adjazenzlisten[u].append(v)
            Adjazenzlisten[v].append(u)

        #Tiefensuche
        T = [0]
        image = np.zeros(self.C.shape[2])

        Q = deque(Adjazenzlisten[0])
        Adjazenzlisten.pop(0)            
        while not len(Q) == 0:
            u = Q.pop()
            if u in Adjazenzlisten:
                image = image + self.C[T[-1],u,:]

                T.append(u)

                Q.extend(Adjazenzlisten[u])
                Adjazenzlisten.pop(u)                
        
        # make Hamiltonian
        image = image + self.C[T[-1],0,:]
        T.append(0)

        image = Image(image, solution = T)
        return image

    # Helperfunktopon MST
    def kruskal(self,A):
        n,_ = A.shape
        #Kanten (i,j), mit i<j (OK, da symmetrisch)
        Kanten = sorted([[i,j] for i in range(n-1) for j in range(i+1,n)], key=lambda item: A[item[0]][item[1]])
        #Kantenliste des Baums
        T = []
        #Initialisiere UnionFind Struktur
        parent = []
        rank = []
        for node in range(n):
            parent.append(node)
            rank.append(0)
        #überprüfe Kanten (bis der Baum vollständig ist.)
        e=0
        for [i,j] in Kanten:
            if e >= n:
                break
            x = self._find(parent, i)
            y = self._find(parent, j)
            if x != y:
                e = e + 1
                T.append((i, j))
                self._apply_union(parent, rank, i, j)
        return T

    def _find(self, parent, i):
        if parent[i] == i:
            return i
        return self._find(parent, parent[i])

    def _apply_union(self, parent, rank, x, y):
        xroot = self._find(parent, x)
        yroot = self._find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

class TravelingSalesman_refinedDoubleTree(TravelingSalesman):
    def __init__(self, C: np.ndarray):
        super().__init__(C)
        self.solution_quality_weighted_sum = 2 

    def weighted_sum(self, weights: np.ndarray):
 
        A = np.matmul(self.C, weights)
 
        #Bestimme MST
        T = self.kruskal(A)
       
        #konvertiere MST in Adjazenzlistendarstellung:
        Adjazenzlisten = defaultdict(list)
        for [u,v] in T:
            Adjazenzlisten[u].append(v)
            Adjazenzlisten[v].append(u)

        # nette Beobachtung: da T aufsteigend nach Kantengröße sortiert ist, 
        # ist automiatisch die Liste Adjazenzliste[u] aufsteigen nach Kosten c(u,v) sortiert :)

        #vorsichtige Tiefensuche
        T = [0]
        image = np.zeros(self.C.shape[2])

        Q = deque(sorted(Adjazenzlisten[0], key = lambda v:A[0,v]))
       
        Adjazenzlisten.pop(0)            
        while not len(Q) == 0:
            u = Q.pop()
            if u in Adjazenzlisten:
                # u will be added -> update T and add Adjanzliste[u] sorted by costs on (u,v)
                image = image + self.C[T[-1],u,:]
                
                # Q.extend(sorted(Adjazenzlisten[u],key = lambda v:A[u,v],reverse=True))
                # sortierung nach obiger Beobachtung nicht nötig.
                Adjazenzlisten[u].reverse()
                Q.extend(Adjazenzlisten[u])
                T.append(u)
                # remove u from dict to mark that u is already visited
                Adjazenzlisten.pop(u)                
        
        # make Hamiltonian
        image = image + self.C[T[-1],0,:]
        T.append(0)
        
        image = Image(image, solution = T)
        return image

    # Helperfunktopon MST
    def kruskal(self,A):
        n,_ = A.shape
        #Kanten (i,j), mit i<j (OK, da symmetrisch)
        Kanten = sorted([[i,j] for i in range(n-1) for j in range(i+1,n)], key=lambda item: A[item[0]][item[1]])
        #Kantenliste des Baums
        T = []
        #Initialisiere UnionFind Struktur
        parent = []
        rank = []
        for node in range(n):
            parent.append(node)
            rank.append(0)
        #überprüfe Kanten (bis der Baum vollständig ist.)
        e=0
        for [i,j] in Kanten:
            if e >= n:
                break
            x = self._find(parent, i)
            y = self._find(parent, j)
            if x != y:
                e = e + 1
                T.append((i, j))
                self._apply_union(parent, rank, i, j)
        return T

    def _find(self, parent, i):
        if parent[i] == i:
            return i
        return self._find(parent, parent[i])

    def _apply_union(self, parent, rank, x, y):
        xroot = self._find(parent, x)
        yroot = self._find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1


class TravelingSalesman_Christofides(TravelingSalesman):
    def __init__(self, C):
        super().__init__(C)
        self.solution_quality_weighted_sum = 1.5

        self.G = nx.Graph()

        
        for i,j in itertools.combinations(range(self.C.shape[0]),2):
            self.G.add_edge(i,j,costs = C[i,j,:])           


    def weighted_sum(self, weights: np.ndarray):
        for i,j in self.G.edges:
            self.G[i][j]['weight'] = np.matmul(self.G[i][j]['costs'],weights) 
            

        tour = nx.algorithms.approximation.christofides(self.G)

        image = [0 for _ in range(self.nr_obj)]
        for i in range(len(tour) -1) :
            for dim in range(self.nr_obj):
                image[dim] += self.C[tour[i],tour[i+1],dim]

        image = Image(image, solution = tour)
        return image

# class to obtain mst nd set as reference set
class Traveling_Salesman_as_MST(TravelingSalesman):
    def __init__(self, C: np.ndarray):
        super().__init__(C)
        self.solution_quality_weighted_sum = 1 

    def weighted_sum(self, weights: np.ndarray):
 
        A = np.matmul(self.C, weights)
 
        #Bestimme MST
        T = self.kruskal(A)


        #extract image set
        image = [0 for _ in range(self.nr_obj)]
        for i,j in T:
            for dim in range(self.nr_obj):
                image[dim] += self.C[i,j,dim]

        image = Image(image, solution = T)
        return image

        # Helperfunktopon MST
    def kruskal(self,A):
        n,_ = A.shape
        #Kanten (i,j), mit i<j (OK, da symmetrisch)
        Kanten = sorted([[i,j] for i in range(n-1) for j in range(i+1,n)], key=lambda item: A[item[0]][item[1]])
        #Kantenliste des Baums
        T = []
        #Initialisiere UnionFind Struktur
        parent = []
        rank = []
        for node in range(n):
            parent.append(node)
            rank.append(0)
        #überprüfe Kanten (bis der Baum vollständig ist.)
        e=0
        for [i,j] in Kanten:
            if e >= n:
                break
            x = self._find(parent, i)
            y = self._find(parent, j)
            if x != y:
                e = e + 1
                T.append((i, j))
                self._apply_union(parent, rank, i, j)
        return T

    def _find(self, parent, i):
        if parent[i] == i:
            return i
        return self._find(parent, parent[i])

    def _apply_union(self, parent, rank, x, y):
        xroot = self._find(parent, x)
        yroot = self._find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
       


def get_traveling_salesman_instance(nr_cities: int, nr_obj:int, bounds = [0,3163], seed = None) -> np.ndarray:
    '''
    generates an instance of a traveling salesman problem with nr_cities cities and nr_obj objectives. 
    This is done by drawing randomly p locations in the rectangle bounds x bounds of the cities, and taking the euclidian distances between two cities as cost vectors.
    Hence, this method yields an instance that satisfies objective-wise the triangle inequality.

    returns distance matric as np.ndarray of shape nr_cities x nr_cities x nr_obj 
            locations as np.ndarray of shape nr_cities x 2 
    '''

    np.random.seed(seed)
    locations = np.random.random_integers(bounds[0],bounds[1], size = [nr_cities,2,nr_obj])
    C = np.zeros([nr_cities,nr_cities,nr_obj])

    for i,j in itertools.combinations(range(nr_cities),2):
        #calculate eucl_dist
        eucl_dist = np.sqrt(np.sum(np.square(locations[i][:][:]-locations[j][:][:]),0))

        # store in matrix C
        C[i,j,:] = eucl_dist
        C[j,i,:] = eucl_dist

    return C,locations[:,:,0]

def write_traveling_salesman_instance(path:str, C):
    with open(path,"w") as file:
        for i,j in itertools.combinations(range(C.shape[0]),2):
            file.write(str(i) + " " + str(j) + " ")
            file.write(np.array2string(C[i,j,:])[1:-1])
            file.write('\n')


def read_traveling_salesman_instance(path:str):
    dist = {}
    n = 0
    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace('[','').replace(']','').replace(',','').replace('\n','')
            line = line.split()
            array = [ float(nr) for nr in line[2:] ]
            nr_obj = len(array)
            dist[int(line[0]),int(line[1])] = np.array(array)

            if int(line[0]) > n:
                n = int(line[0])

    
    n = n + 2

    C = np.zeros([n,n,nr_obj])

    for key, value in dist.items():
        C[key[0],key[1],:] = value
        C[key[1],key[0],:] = value
    return C

        

