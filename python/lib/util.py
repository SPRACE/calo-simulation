import numpy as np

def mean_component(data, axis):
    component_sum = np.sum(data, axis=axis)
    component_mean = np.mean(component_sum, axis=0)
    return component_mean

def mean_eta(data):
    return mean_component(data, axis=1)

def mean_phi(data):
    return mean_component(data, axis=2)
               
    
def sum_energy(data):
    if len(data.shape) == 2:
        return np.sum(data, axis=(0,1))
    else:
        return np.sum(data, axis=(1,2))