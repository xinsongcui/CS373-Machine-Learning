#Xinsong Cui
#0028860023

import numpy as np 
import pandas as pd
import matplotlib.pyplot as pl
"""
train_size = [40, 60 , 80, 100]
train_score = [0.9558, 0.9571, 0.9558, 0.9518]
test_score = [0.8022, 0.7761, 0.7799, 0.8097]
df = pd.DataFrame({'data':train_size, 'train':train_score, 'test':test_score})
pl.plot('data', 'train', 'm', data = df)
pl.plot('data', 'test', 'k--', data = df)
pl.title("Accuray vs training size")
pl.ylabel("Accuracy")
pl.xlabel("training size")
pl.show()

train_size = [40,50,60,70,80]
best_train_score = [0.8715, 0.8617, 0.9383, 0.8509, 0.8454]
best_test_score = [0.8246, 0.8358, 0.7612, 0.8134, 0.8246]
best_depth = [5, 5, 10, 5, 5]
best_valid_socre = [0.8000, 0.8240, 0.8000, 0.8000, 0.7920]
df = pd.DataFrame({'data':train_size, 'train':best_train_score, 'test':best_test_score})
pl.plot('data', 'train', 'm', data = df)
pl.plot('data', 'test', 'k--', data = df)
pl.title("Accuray vs training size")
pl.ylabel("Accuracy")
pl.xlabel("training size")
pl.show()

train_size = [40,50,60,70,80]
best_train_score = [0.8715, 0.8617, 0.9383, 0.8509, 0.8454]
best_test_score = [0.8246, 0.8358, 0.7612, 0.8134, 0.8246]
best_depth = [5, 5, 10, 5, 5]
best_valid_socre = [0.8000, 0.8240, 0.8000, 0.8000, 0.7920]
df = pd.DataFrame({'data':train_size, 'depth':best_depth})
pl.plot('data', 'depth', 'k--', data = df)
pl.title("optimal choice of depth vs training size")
pl.ylabel("depth")
pl.xlabel("training size")
pl.show()

train_size = [40, 50, 60 , 70, 80]
train_score = [0.9558, 0.9582, 0.9571, 0.9564, 0.9558]
test_score = [0.8022, 0.7649, 0.7761, 0.7649, 0.7799]
df = pd.DataFrame({'data':train_size, 'train':train_score, 'test':test_score})
pl.plot('data', 'train', 'm', data = df)
pl.plot('data', 'test', 'k--', data = df)
pl.title("Accuray vs training size")
pl.ylabel("Accuracy")
pl.xlabel("training size")
pl.show()

train_size = [40, 60 , 80, 100]
nodes = [121, 171, 221, 289]
df = pd.DataFrame({'node':nodes, 'train':train_size})
pl.plot('train', 'node', 'g', data = df)
pl.title("number of nodes vs training size")
pl.ylabel("nodes")
pl.xlabel("training size")
pl.show()
"""
train_size = [40, 50, 60 , 70, 80]
nodes = [93, 113, 135, 144, 161]
df = pd.DataFrame({'node':nodes, 'train':train_size})
pl.plot('train', 'node', 'g', data = df)
pl.title("number of nodes vs training size")
pl.ylabel("nodes")
pl.xlabel("training size")
pl.show()
