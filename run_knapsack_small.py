# own imports
from molib.problems import knapsack as kp
from molib.problems.image_set import ImageSet_withLexWS
from molib.core import Image
from molib.algorithms.exact_methods import outer_approximation
from molib.algorithms.approximations import approximate_convex_pareto_set as acPs
from molib.core import quality_indicators as qi


# regular imports
from multiprocessing import Pool
import os
import numpy as np
import func_timeout
import datetime
import pandas as pd

#subroutines    
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

# input and output path
input_path = os.path.join(os.path.realpath('.'),'instances','Knapsack_small')
output_path = os.path.join(os.path.realpath('.'),'outputs','Knapsack_small')
reference_set_path = os.path.join(os.path.realpath('.'),'outputs_reference','Knapsack_small')

# instances
#INSTANCES = ['3-Confliciting-' + str(n) + '-' + str(i) + '.dat' for n in [10,20,30] for i in range(1,6)] + ['3-Uniform-' + str(n) + '-' + str(i) + '.dat' for n in [10,20,30] for i in range(1,6)] 
instances = ['3-Confliciting-' + str(n) + '-' + str(i) + '-' + str(ub) + '.dat' for n in [10] for i in range(1,2) for ub in  list([ 0.01 * k for k in range(1,11)])] + ['3-Uniform-' + str(n) + '-' + str(i) + '-' + str(ub) + '.dat' for n in [10] for i in range(1,2) for ub in  list([ 0.01 * k for k in range(1,11)])]

# Constants
# TODO: Adapt to your needs!
RUNTIME_LIMIT = 3600  # seconds
MEMORY_LIMIT = 16  # GB
MEMORY_LIMIT *= 1024 * 1024 * 1024

def get_convex_approximation_set(configuration):
    instance,subroutine,alg,eps,run = configuration

    #read instance
    profits, weights,capacity = kp.read_knapsack_instance(os.path.join(input_path,instance))
    name = instance[:-4]
    problem = subroutine(profits,weights,capacity)
    alg_name = alg[1]
    n = len(weights)
    nr_obj = problem.nr_obj
    alpha = problem.solution_quality_weighted_sum
    th_approx = (1 + eps) * alpha

    
    print(instance,problem.name,alg_name,eps,run)
    try:
        start = datetime.datetime.now()
        images = func_timeout.func_timeout(
            RUNTIME_LIMIT, 
            alg[0], 
            args=[problem,eps]
            )
        end = datetime.datetime.now()

        #get specs
        running_time = (end-start).total_seconds() * 1000
        running_time_in_weighted_sum = problem.time_in_weighted_sum
        anzahl_aufrufe = problem.calls_weighted_sum
        anzahl_solutions = len(images) 
    except func_timeout.FunctionTimedOut:
        print(instance,problem.name,alg_name,eps,run,' did not finish for within ' + str(RUNTIME_LIMIT) + ' seconds, skip run')
        
        #create dummy specs
        image = Image([0,0,0])
        images = {image}
        running_time = 0
        running_time_in_weighted_sum = 0
        anzahl_aufrufe = 0
        anzahl_solutions = 0 

    # save convexPareto set with results
    save_name = 'ConvexApproxSet-' + problem.name + '-' + alg_name +'-eps-' + str(eps).replace('.','') + '-run-' + str(run) + '-' + name + '.txt' 
    np.savetxt(os.path.join(output_path,save_name), np.asarray(list(images)),
    fmt='%i',
    header = ' '.join( col + '-' + str(s) +" " for col,s in zip(['instance_type','instance_nr','n','d','alpha','subroutine','epsilon','running_time_(ms)','running_time_ws_(ms)','nr_calls','nr_solutions','th_approx'],
                                                                [name.split('-')[1],name.split('-')[3], n,nr_obj, str(alpha), problem.name, eps, running_time, running_time_in_weighted_sum,anzahl_aufrufe,anzahl_solutions,th_approx]))
    )  
    return

def evaluate_convex_approximation_set(instance,subroutine,alg,eps,run):
        # read instance
    name = instance[:-4]
    alg_name = alg[1]
    problem_name = 'Greedy'
    print(instance,problem_name,alg_name,eps,run)
    
    # load convex approxiamtion set
    save_name = 'ConvexApproxSet-' + problem_name + '-' + alg_name +'-eps-' + str(eps).replace('.','') + '-run-' + str(run) + '-' + name + '.txt' 
    images = np.loadtxt(os.path.join(output_path,save_name),skiprows=1)
    with open(os.path.join(output_path,save_name),"r") as file:
        specs = file.readline()
    

    # check if instance did not finish within time horizon
    if specs.split()[8].split('-')[1] == '0':
        specs = specs[2:-2] + ' real_approx-' + str(0)
        np.savetxt(os.path.join(output_path,save_name), np.atleast_2d(images),
        fmt='%i',
        header = specs)  
        return
    
    # determine convex indicator
    try:
        # load set including extreme supported nondomianted images
        reference_images = np.loadtxt(os.path.join(reference_set_path,'ESN-' + name + '.txt'))

        #determine epsilon convex indicator
        real_approx = qi.parametric_eps_indicator.get_epsilon_convex_indicator_GUROBI2(images,reference_images,type = "max")
    except:
        print('convex approximation error could not be determined!')
        real_approx = 0

    # save results with convex indicator
    specs = specs[2:-2] + ' real_approx-' + str(real_approx)
    np.savetxt(os.path.join(output_path,save_name), images,
    fmt='%i',
    header = specs) 
    return

def collect_data_of_runs(instance,subroutine,alg,eps,run):
    file_name = 'ConvexApproxSet-' + subroutine + '-' + alg + '-eps-' + str(eps).replace('.','') + '-run-' + str(run) + '-' + instance[:-4] + '.txt'
    print(file_name)
    try:
        with open(os.path.join(output_path,file_name),"r") as file:
            specs = file.readline().replace(',','').split()
            instance_type = specs[1].split('-')[1]
            instance_nr = specs[2].split('-')[1]
            n = int(specs[3].split('-')[1])
            d = int(specs[4].split('-')[1])
            alpha = float(specs[5].split('-')[1])
            subroutine = specs[6].split('-')[1]
            epsilon = float(specs[7].split('-')[1])

            running_time = float(specs[8].split('-')[1])
            running_time_ws = float(specs[9].split('-')[1])
            nr_calls =  int(specs[10].split('-')[1])
            nr_solutions = int(specs[11].split('-')[1])
            th_approx = alpha * (1 + eps)
            real_approx = float(specs[13].split('-')[1])
            Y_ESN = np.loadtxt(os.path.join(reference_set_path,'ESN-' + instance[-4] + '.txt'))
            if running_time < 0.00001:    
                #algorithm did not finish in time
                return instance_type,instance_nr,d,n,subroutine,alpha,alg, epsilon, th_approx, np.nan, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,np.nan
            else:
                return instance_type,instance_nr,d,n,subroutine,alpha,alg, epsilon, th_approx, running_time, running_time_ws, running_time_ws/running_time, nr_calls, nr_solutions,nr_solutions/nr_calls,nr_solutions / Y_ESN.shape[0], real_approx

    
    except:
        print('missing_files')
        return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,np.nan,np.nan,np.nan,np.nan


def main(number_of_processes = None):

    subroutines = [Knapsack_Greedy_with_Specs]
    convex_algorithms = [(acPs.DandY,"DandY"),(acPs.GridFPTAA,"Grid"),(acPs.FPTOAA,"OAA")]
    epsilons = [0.5,0.25,0.1]    
    nr_runs = 10

    print('Step 2: Determine set of convex approximation sets')
    configurations = []
    for instance in instances:
        for subroutine in subroutines:
            for alg in convex_algorithms:
                for eps in epsilons:
                    for run in range(nr_runs):
                        configurations.append((instance,subroutine,alg,eps,run))

    with Pool(number_of_processes) as pool:
        result = pool.map(get_convex_approximation_set, configurations)
    pool.join()

    print('Step 3: Determine epsilon convex indicator of determined convex approximation sets')
    for instance in instances:
        for subroutine in subroutines:
            for alg in convex_algorithms:
                for eps in epsilons:
                    for run in range(nr_runs):
                        try:
                            evaluate_convex_approximation_set(instance,subroutine,alg,eps,run)
                        except:
                            continue
    
    print('Step 4: Collect data of instances for evaluation')
    

    print('Step 5: Collect data of runs')
    results = []
    for instance in instances:
        for subroutine in subroutines:
            for alg in convex_algorithms:
                for eps in epsilons:
                    for run in range(nr_runs):
                        results.append(collect_data_of_runs(instance,subroutine,alg,eps,run))
    columns = ['instance_type','instance_nr','d', 'n', 'subroutine','alpha', "convex approximation algorithm", "epsilon","(1+eps)alpha",
           "running time (ms)","running time in subroutine (ms)", "ratio running time subroutine/ running time", 
           "nr calls subroutine","card(hatY)","ratio card(hatY) / nr calls","ratio card(hatY) / card(Y_ESN)",
           "convex indicator"]
    df = pd.DataFrame(data=results,columns = columns)
    df.to_csv("Knapsack_small_convex_approx_results.csv",index = False)

    print('Step 6: Evaluate')
    

if __name__ == "__main__":
    main()