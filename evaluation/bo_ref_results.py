from synthesis.ode_1_repo import ODE_1_Repository

import dill 

with open("results/20260125_171148/result_1_WL.pkl", 'rb') as f: 
    data = dill.load(f)
print(data)