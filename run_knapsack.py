from molib.problems import knapsack as kp
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
class Knapsack_Greedy_with_Specs(kp.Knapsack_Greedy):
    def __init__(self, profits: np.ndarray, weights: np.ndarray, capacity: int):
        super().__init__(profits, weights, capacity)
        self.name = 'Greedy'
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
    
    profits, weights,capacity = kp.read_knapsack_instance(os.path.join(dir_path,'instances','Knapsack',instance_name))
    instance = {
        "name": instance_name[:-4],
        "path": os.path.join(dir_path,'instances','Knapsack_small',instance_name),
        "type": instance_name.split('-')[1],
        "nr" : int(instance_name.split('-')[3]),
        "d" : len(profits[0]),
        "n": len(weights),
        "profits": profits.tolist(),
        "weights": weights.tolist(),
        "capacity": capacity,
        "Y_ESN" : np.loadtxt(os.path.join(dir_path,'outputs_references','Knapsack','ESN-' + instance_name[:-4] + '.txt'),ndmin=2).tolist(),
    }
    logging.info(f"Run instance " + instance["name"])
        
    subroutines = [Knapsack_Greedy_with_Specs]
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
    
    with open(os.path.join(dir_path,'results','Knapsack',instance["name"] + '.json'), "w") as f:
        json.dump(instance,f, indent=4)

    return 
    
def get_convex_approximation_set(configuration):
    instance,subroutine,alg,eps,run = configuration

    problem = subroutine(np.array(instance["profits"]),np.array(instance["weights"]),np.array(instance["capacity"]))
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
        result["epsilon_convex_indicator"] = get_epsilon_convex_indicator(images=np.asarray(list(images)).tolist(),Y_ESN=instance["Y_ESN"],type="max",method="gurobi")

    except func_timeout.FunctionTimedOut:
        logging.info(f"...did not finish within {1}h")
        result["running time (s)"] = None
        result["images"] = None
        result["epsilon_convex_indicator"] = None
    return result


if __name__ == "__main__":
    instances = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'instances','Knapsack'))
    
    for instance in instances:
        main(instance)