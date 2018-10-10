import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_hist(data):
    """ 
    """    
    fig = plt.figure()
    plt.imshow(data, norm=LogNorm())
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Energy [GeV]", rotation=270, fontsize=18, labelpad=25)
    plt.xlabel("ϕ", fontsize=16)
    plt.ylabel("η", fontsize=16)
    plt.show()
    
    
def plot_cumulative(data, xlabel="", ylabel="", legend=""):
    """
    """
    data_to_plot = []
    if not isinstance(data, tuple):
        data_to_plot.append(data)
    else:
        data_to_plot = list(data)
    
    for data in data_to_plot:
        plt.plot(np.log1p(data))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)    
        
        
    
    