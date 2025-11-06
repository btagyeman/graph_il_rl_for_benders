import numpy as np 
import scipy.sparse as sp
from spektral.data import Graph


def create_graph_representation(cut_order,graph_features,past_values, optimal_values,n_variables):
    if len(graph_features['optimality_cuts'].values()) != 0 and len(graph_features['feasibility_cuts'].values()) != 0:
        cons_opt = list(graph_features['optimality_cuts'].values())
        cons_feas= list(graph_features['feasibility_cuts'].values())
        cons_total = len(cons_opt) + len(cons_feas)
        cons_combined=[0]*cons_total
        idx_feas=[]
        idx_opt=[]
        for i in range(len(cut_order)):
            if cut_order[i] == 'feasibility_cut':
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
    elif len(graph_features['optimality_cuts'].values()) != 0 and len(graph_features['feasibility_cuts'].values()) == 0:
        # Only optimality cuts are present
        cons_opt = list(graph_features['optimality_cuts'].values())
        cons_total = len(cons_opt)
        cons_combined = cons_opt
        pass 
    elif len(graph_features['optimality_cuts'].values()) == 0 and len(graph_features['feasibility_cuts'].values()) != 0:
        # Only feasibility cuts are present
        cons_feas = list(graph_features['feasibility_cuts'].values())
        cons_total = len(cons_feas)
        cons_combined = cons_feas
        pass
    else:
        # No cuts present
        print("No cuts found in graph features. Returning None.")
        raise ValueError("No cuts found in graph features. Cannot create graph representation.")
    

    n_nodes = n_variables + cons_total # number of nodes = number of variables + number of constraints
    A_matrix = np.zeros((cons_total, n_variables))
    b_vector = np.zeros(cons_total)
    for j in range(cons_total):
        A_matrix[j,:] = cons_combined[j][0:-1]  # Get coefficients for the first n_vars variables
        b_vector[j] = cons_combined[j][-1]  # Get the right-hand side value
        pass
    var_node_features = np.array(past_values).reshape(-1,1)  # Use the values of the complicating variables at the previous iter as node features  
    cons_node_features = b_vector.reshape(-1,1) 
    x=np.concatenate((var_node_features, cons_node_features)) # nodes features 
    edges = []
    edge_features = []
    for i in range(cons_total):
        for j in range(n_variables):
            if A_matrix[i,j]!=0:
                var_node = j
                cons_node = n_variables + i
                edges.append([var_node,cons_node])
                edges.append([cons_node,var_node])  # symmetric
                edge_features.append([A_matrix[i,j]])
                edge_features.append([A_matrix[i,j]])
                pass 
            pass 
        pass 
    edges=np.array(edges).T
    a=sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n_nodes, n_nodes))
    e=np.array(edge_features) 
    y=np.array(optimal_values)
    return Graph(x=x, a=a, e=e, y=y)