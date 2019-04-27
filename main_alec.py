import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

"""
Author: Alec Yen
ECE 471: Introduction to Pattern Recognition
Final Project
main_alec.py

Purpose: Main Program
Description
"""

# files with my function definitions
import alec.perfeval as pe
import alec.classify_s as cls
import alec.preprocessing as pp

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion')
print(data)
