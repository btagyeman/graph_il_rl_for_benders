import numpy as np 
import scipy.sparse as sp
from spektral.data import Graph

def create_graph_representation(self):
    _graphs = []
    if len(self.graph_features['optimality_cuts'].values()) != 0 and len(self.graph_features['feasibility_cuts'].values()) != 0:
        cons_opt = list(self.graph_features['optimality_cuts'].values())
        cons_feas= list(self.graph_features['feasibility_cuts'].values())
        cons_total = len(cons_opt) + len(cons_feas)
        cons_combined=[0]*cons_total
        idx_feas=[]
        idx_opt=[]
        for i in range(len(self.cut_order)):
            if self.cut_order[i]=='feasibility_cut':
                idx_feas.append(i)
                pass 
            else:
                idx_opt.append(i)
                pass 
            pass 
        for i in range(len(idx_feas)):
            cons_combined[idx_feas[i]] = cons_feas[i]
            pass 
        for j in range(len(idx_opt)):
            cons_combined[idx_opt[j]] = cons_opt[j]
            pass
    elif len(self.graph_features['optimality_cuts'].values()) != 0 and len(self.graph_features['feasibility_cuts'].values()) == 0:
        # Only optimality cuts are present
        cons_opt=list(self.graph_features['optimality_cuts'].values())
        cons_total=len(cons_opt)
        cons_combined=cons_opt
        pass 
    elif len(self.graph_features['optimality_cuts'].values()) == 0 and len(self.graph_features['feasibility_cuts'].values()) != 0:
        # Only feasibility cuts are present
        cons_feas=list(self.graph_features['feasibility_cuts'].values())
        cons_total=len(cons_feas)
        cons_combined=cons_feas
        pass
    else:
        # No cuts present
        print("No cuts found in graph features. Returning None.")
        raise ValueError("No cuts found in graph features. Cannot create graph representation.")
    
    n_cons_set=np.arange(cons_total)+1 
    for cons in n_cons_set:
        n_nodes=self.n_variables + cons # number of nodes = number of variables + number of constraints
        A_matrix=np.zeros((cons, self.n_variables))
        b_vector=np.zeros(cons)

        for j in range(cons):
            A_matrix[j,:] = cons_combined[j][0:-1]  # Get coefficients for the first n_vars variables
            b_vector[j] = cons_combined[j][-1]  # Get the right-hand side value
            pass
        var_node_features = np.array(self.optimal_values[cons-1]).reshape(-1,1)  # Use the values of the complicating variables at the previous iter as node features 
        # var_node_features = np.zeros(self.n_variables).reshape(-1,1) # TO DO: CHANGE THIS IF NEEDED
        cons_node_features = b_vector.reshape(-1,1) 
        x=np.concatenate((var_node_features,cons_node_features)) # nodes features 
        edges = []
        edge_features = []
        for i in range(cons):
            for j in range(self.n_variables):
                if A_matrix[i,j]!=0:
                    var_node = j
                    cons_node = self.n_variables + i
                    edges.append([var_node,cons_node])
                    edges.append([cons_node,var_node])  # symmetric
                    edge_features.append([A_matrix[i,j]])
                    edge_features.append([A_matrix[i,j]])
        edges=np.array(edges).T
        a=sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n_nodes, n_nodes))
        e=np.array(edge_features) 
        y=np.array(self.optimal_values[cons])
        _graphs.append(Graph(x=x,a=a,e=e,y=y))
        pass
    return _graphs