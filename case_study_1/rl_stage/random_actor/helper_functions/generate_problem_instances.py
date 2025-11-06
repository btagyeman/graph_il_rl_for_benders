import numpy as np
from pyomo.environ import *
from spektral.data import Dataset
from helper_functions.initial_problem import solve_initial_problem

from tqdm import tqdm

class GenerateDataset(Dataset):
    def __init__(self,n_samples=100, **kwargs):
        self.w = []
        self.lbw = []
        self.ubw = []
        self.J = 0
        self.G = []
        self.lbg = []
        self.ubg = []
        self.discrete = []
        self.Guess = []
        self.a = [0, 0, 0, 0, 0, 0]
        self.b = [2, 2, 2, np.inf, np.inf, 3]
        self.y_guess = [1, 0, 0, 0, 0]
        self.x_guess = [1, 1, 1, 1, 1, 1]
        self.optimal_values = []
        self.optimal_solutions = []
        self.feasible_parameters = []
        self.optimal_cost = []
        self.n_samples = n_samples
        
        #original parameters
        self.U = 10
        self.coeff_y1 = 5
        self.coeff_y2 = 8
        self.coeff_y3 = 6
        self.coeff_y4 = 10
        self.coeff_y5 = 6
        self.rhs_logical_1 = 1
        self.rhs_logical_2 = 1
        self.rhs_y12 = 1
        self.rhs_y45 = 1

    def generate_parameters(self):
        self.coeff_y1 = np.random.randint(1, 40)
        self.coeff_y2 = np.random.randint(1, 40)
        self.coeff_y3 = np.random.randint(1, 40)
        self.coeff_y4 = np.random.randint(1, 40)
        self.coeff_y5 = np.random.randint(1, 8)
        # self.U = np.random.randint(2, 20)
        self.U = 10
        self.rhs_logical_1 = 1
        self.rhs_logical_2 = 1
        self.rhs_y12 = 1
        self.rhs_y45 = 1
        pass 

    def generate_feasible_parameters(self):
        print("Generating feasible parameters: Starting...")
        feasible_count = 0

        for _ in tqdm(range(self.n_samples), desc="Feasible sets"):
            self.generate_parameters()
            is_feasible = solve_initial_problem(self)
            if is_feasible:
                feasible_count += 1

        print(f"Generating feasible parameters: Completed with {feasible_count} feasible sets.")