#Xinsong Cui
#0028860023

import numpy as np 
import pandas as pd
import matplotlib.pyplot as pl

"""
vocab_size = [100, 500, 1000, 5000, 10000, 20000]
accuarcy = [51.16, 57.14, 62.46, 71.76, 73.09, 73.09]
df = pd.DataFrame({'data':vocab_size, 'depth':accuarcy})
pl.plot('data', 'depth', 'k--', data = df)
pl.title("Test accuracy vs vocabulary size")
pl.ylabel("Test Accuracy")
pl.xlabel("Vocabulary Size")
pl.show()

"""

num_iter = [1, 2, 5, 10, 20, 50]
accuarcy = [65.45, 51.16, 52.82, 73.09, 74.42, 76.08]
df = pd.DataFrame({'data':num_iter, 'depth':accuarcy})
pl.plot('data', 'depth', 'k--', data = df)
pl.title("Test accuracy vs number iteration")
pl.ylabel("Test Accuracy")
pl.xlabel("Number Iteration")
pl.show()

With bias:
Perceptron Results:
Accuracy: 77.74, Precision: 79.56, Recall: 73.64, F1: 76.49

Averaged Perceptron Results:
Accuracy: 76.08, Precision: 82.75, Recall: 64.86, F1: 72.72

Without:
Perceptron Results:
Accuracy: 76.41, Precision: 78.94, Recall: 70.94, F1: 74.73

Averaged Perceptron Results:
Accuracy: 79.40, Precision: 76.21, Recall: 84.45, F1: 80.12

