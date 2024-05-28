# own imports
from molib.problems import traveling_salesman as tsp
from molib.algorithms.approximations import approximate_convex_pareto_set as acPs
from molib.core import Image
from molib.core.quality_indicators import get_epsilon_convex_indicator

import json
import os
import logging
import numpy as np
import datetime
import func_timeout


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level = logging.INFO)    
class TravelingSalesman_Christofides_Specs(tsp.TravelingSalesman_Christofides):
    def __init__(self, C):
        super().__init__(C)
        self.name = "Christofides"
        self.calls_weighted_sum = 0
        self.time_in_weighted_sum = 0

    def weighted_sum(self, weights: list) -> Image:
        self.calls_weighted_sum += 1   
        start = datetime.datetime.now()
        convexParetoSet = super().weighted_sum(weights)
        end = datetime.datetime.now()
        self.time_in_weighted_sum += (end-start).total_seconds() * 1000
        return convexParetoSet

def main(instance_name : str):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    C = tsp.read_traveling_salesman_instance(os.path.join(dir_path,'instances','TSP',instance_name))
    instance = {
        "name": instance_name[:-4],
        "path": os.path.join(dir_path,'instances','TSP',instance_name),
        "nr" : int(instance_name[:-4].split('-')[1]),
        "d" : C.shape[2],
        "n": C.shape[0],
        "cost matrix": C.tolist(),
        "Y_ESN" : np.loadtxt(os.path.join(dir_path,'outputs_references','TSP','ESN-' + instance_name[:-4] + '.tsp'),ndmin=2).tolist(),
    }
    logging.info(f"Run instance " + instance["name"])
        
    subroutines = [TravelingSalesman_Christofides_Specs]
    convex_algorithms = [(acPs.DandY,"DandY"),(acPs.GridFPTAA,"Grid"),(acPs.FPTOAA,"OAA")]
    epsilons = [0.5,0.25,0.1]    
    nr_runs = 10

    configurations = []
    for subroutine in subroutines:
        for alg in convex_algorithms:
            for eps in epsilons:
                for run in range(nr_runs):
                    configurations.append((instance,subroutine,alg,eps,run))


    from multiprocessing import Pool
    with Pool() as pool:
        instance["results"] = pool.map(get_convex_approximation_set, configurations)
    pool.join()
    
    with open(os.path.join(dir_path,'results','TSP',instance["name"] + '.json'), "w") as f:
        json.dump(instance,f, indent=4)

    return 
    
def get_convex_approximation_set(configuration):
    instance,subroutine,alg,eps,run = configuration

    problem = subroutine(np.array(instance["cost matrix"]))
    result = {
        "alpha" : problem.solution_quality_weighted_sum,
        "epsilon" : eps,
        "convex approximation algorithm": alg[1],
        "subroutine": problem.name,
        "run" : run
        }
    
    logging.info(f"Run {run} of instance " + instance["name"] + f" with algorithm {alg[1]} for epsilon={eps}")

    try:
        start = datetime.datetime.now()
        images = func_timeout.func_timeout(
            3600, 
            alg[0], 
            args=[problem,eps]
            )
        end = datetime.datetime.now()

        #get specs
        result["running time (s)"] = (end-start).total_seconds()
        result["images"] = np.asarray(list(images),).tolist()
        result["epsilon_convex_indicator"] = get_epsilon_convex_indicator(images=np.asarray(list(images)).tolist(),Y_ESN=instance["Y_ESN"],type="min",method="gurobi")

    except func_timeout.FunctionTimedOut:
        logging.info(f"...did not finish within {1}h")
        result["running time (s)"] = None
        result["images"] = None
        result["epsilon_convex_indicator"] = None
    return result


if __name__ == "__main__":
    instances = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'instances','TSP'))
    
    for instance in instances:
        main(instance)