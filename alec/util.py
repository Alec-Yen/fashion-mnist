import matplotlib.pyplot as plt
import numpy as np

"""
Author: Alec Yen
ECE 471: Introduction to Pattern Recognition

Purpose: Functions
This file includes function definitions for general utilities.
This is just for my convenience when using math functions 
or plotting lots of graphs.
"""

############################################################

"""
Purpose:
    Get Euclidean distance
Args:
    x1: point 1
    x2: point 2
Returns:
    Euclidean distance
"""
def dist (a,b,ax=1):
    return np.linalg.norm(a - b, axis=ax)


"""
Purpose:
    Plot multiple curves easily
    Write curve to figures/
Args:
    x: the x axis, can either be an array or a list of arrays
    curves: list of curves
    title: what to name the plot
    xlabel: what to name the x axis
    ylabel: what to name the y axis
    legend: what to have in the legend
    flag: 0 or 1:
        flag=0: each curve shares common x
        flag=1: each curve has different x
Returns:
    None
"""
# if flag = 1, plot different x for each curve
def plot_curves (x, curves, title, xlabel, ylabel, legend, flag):
    fig = plt.figure()
    line = []
    if flag == 0:
        for c in curves:
            temp, = plt.plot(x, c)
            line.append(temp)
    else:
        for index,c in enumerate(curves):
            temp, = plt.plot(x[index],c)
            line.append(temp)

    #plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(line,legend)

    plt.savefig("latex/figures/"+title+".png")

    fig.show()

