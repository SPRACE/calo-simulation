import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_hist(data):
    """Function to plot a heatmap from particle data

    Args:
        data (numpy.ndarray): A 2D array containing the energy deposition at the calorimeter cells

    Returns:
        void: Plots the 2D histogram accordingly

    """    
    
    fig = plt.figure()
    plt.imshow(data, norm=LogNorm())
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Energy [GeV]", rotation=270, fontsize=18, labelpad=25)
    plt.xlabel("ϕ", fontsize=16)
    plt.ylabel("η", fontsize=16)
    plt.show()
    return fig
    
def plot_cumulative(data, xlabel="", ylabel="", legend=""):
    """Function to plot a line chart representing the eta and phi components

    Args:
        data (numpy.ndarray or tuple): A 2D array containing the energy deposition 
                                       at the calorimeter cells in eta or phi.
                                       In case of a tuple, several lines are printed.
        xlabel (string): The value to be printed at the X-axis label
        ylabel (string): The value to be printed at the Y-axis label
        legend (tuple): The set of values to be used at the legend

    Returns:
        void: Plots the line chart accordingly

    """    
    gen_plot(data=data, xlabel=xlabel, ylabel=ylabel, 
             legend=legend, plot_func=plt.plot)
    
def plot_energy_hist(data, bins=10, xlabel="", ylabel="", legend=""):
    """Function to plot a histogram representing the distribution of the summed energy

    Args:
        data (numpy.ndarray or tuple): A 2D array containing the sum of the energy deposited 
                                       at the calorimeter cells for several particles.
                                       In case of a tuple, several histograms are printed.
        bins (int): The number of bins to use at each histogram
        xlabel (string): The value to be printed at the X-axis label
        ylabel (string): The value to be printed at the Y-axis label
        legend (tuple): The set of values to be used at the legend

    Returns:
        void: Plots the histogram accordingly

    """    
    gen_plot(data=data, xlabel=xlabel, ylabel=ylabel, 
             legend=legend, plot_func=plt.hist, bins=bins, 
             histtype='step', linewidth=2.)
    
def gen_plot(data=np.zeros((0,1)), xlabel="", ylabel="", legend="", plot_func=plt.plot, **kwargs):
    """Generic Function to plot a chart

    Args:
        data (numpy.ndarray or tuple): A 2D array containing the energy deposited 
                                       at the calorimeter cells for several particles.
                                       In case of a tuple, several charts are printed.
        xlabel (string): The value to be printed at the X-axis label
        ylabel (string): The value to be printed at the Y-axis label
        legend (tuple): The set of values to be used at the legend
        plot_func (function): The function used to plot
        kwargs: a set of extra arguments used at the plot 

    Returns:
        void: Plots the chart accordingly

    """    
    data_to_plot = []
    if not isinstance(data, tuple):
        data_to_plot.append(data)
    else:
        data_to_plot = list(data)
        
    for data in data_to_plot:
        plot_func(data, **kwargs)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)    
    
    