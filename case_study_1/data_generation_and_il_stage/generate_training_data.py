import numpy as np
from pyomo.environ import *
from spektral.data import Dataset
import pickle
import os
from helper_functions.solve_initial_problem import solve_initial_problem
from helper_functions.solve_with_gbd import solve_gbd_problem
from helper_functions.graph_representation import create_graph_representation

class GenerateDataset(Dataset):
    def __init__(self,n_samples=100, **kwargs):
        self.n_samples=n_samples
        self.feasible_parameters=[]
        self.feasible_parameter_cur =[0]*10
        self.optimal_solutions=[]
        self.graph_features=None 
        self.cut_order=None 
        self.optimal_values=None #This is actually for the optimal values of the complicating variables
        self.initial_guesses = [[1,0,0,0,0]]
        
        self.n_variables = 5  # y1, y2, y3 , y4, y5
        self.x_guess = [1, 1, 1, 1, 1, 1]
        self.lbd = -1e5
        self.ubd = 1e5
        self.mu_guess = 0.6
        self.epsilon = 0.001

        
        super().__init__(**kwargs)
        pass 

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
        self.rhs_logical_1 = 1
        self.rhs_logical_2 = 1
        self.rhs_y12 = 1
        self.rhs_y45 = 1
        self.U = 10
        pass 

    def generate_feasible_parameters(self):
        print("Generating feasible parameters: Starting...")
        for _ in range(self.n_samples):
            print(f"Set {_+1}")
            self.generate_parameters()
            solve_initial_problem(self)
        pass
        print("Generating feasible parameters: Completed")

    def generate_graph_data(self):
        return  create_graph_representation(self)
    
    def read(self): # This method is automatically called by the constructor of Dataset class
        print("Generating GBD dataset: Starting...")
        entire_dataset = []
        self.generate_feasible_parameters()
        for i in range(len(self.feasible_parameters)):
            print(f"Solving GBD for feasible parameter set {i+1}")
            for initial_guess in self.initial_guesses:
                print(f"Using guess: {initial_guess}")
                self.feasible_parameter_cur=self.feasible_parameters[i]
                solve_gbd_problem(self,self.feasible_parameter_cur, self.ubd, self.lbd, self.mu_guess, self.x_guess, initial_guess, epsilon=self.epsilon)
                entire_dataset.extend(self.generate_graph_data())
                pass 
            print(f"Completed solving GBD for feasible parameter set {i+1}\n")
            pass 
        pass
        print("Generating GBD dataset: Completed\n")
        return entire_dataset 
    
    def generate_gbd_dataset(self):
        self.generate_feasible_parameters()
        for i in range(len(self.feasible_parameters)):
            for initial_guess in self.initial_guesses:
                self.feasible_parameter_cur=self.feasible_parameters[i]
                solve_gbd_problem(self,self.feasible_parameter_cur, self.ubd, self.lbd, self.mu_guess, self.x_guess, initial_guess, epsilon=self.epsilon)
                pass 
            pass 
        pass
        
os.makedirs("datasets", exist_ok=True)
print("Generating first dataset of 3000 samples...")
dataset_a = GenerateDataset(n_samples=3000)
with open("datasets/graph_gbd_dataset_3k.pkl", "wb") as f_a:
    pickle.dump(dataset_a.graphs, f_a)
    pass 
print("First dataset of 3000 samples generated and saved as   'datasets/graph_gbd_dataset_3k.pkl'")
