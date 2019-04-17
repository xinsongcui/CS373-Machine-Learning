import numpy as np 
import pandas as pd
import matplotlib.pyplot as pl
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
