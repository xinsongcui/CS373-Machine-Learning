#!/usr/bin/env python
# coding: utf-8
"""© 2019 Rajkumar Pujari All Rights Reserved

- Original Version

    Author: Rajkumar Pujari
    Last Modified: 03/12/2019

"""

from classifier import BinaryClassifier
import utils
from utils import get_feature_vectors
import numpy as np
import random



class Perceptron(BinaryClassifier):
	
	def __init__(self, args):
	#TO DO: Initialize parameters here
		#print(args);
		global f_dim, vocab_size, num_iter, lr, bin_feats, Bias, W;
		
		f_dim = args.f_dim;
		vocab_size = args.vocab_size;
		num_iter = args.num_iter;
		lr = args.lr;
		bin_feats =args.bin_feats;		
		Bias = 0;
		W = [0.] * vocab_size;

	def fit(self, train_data):
	#TO DO: Learn the parameters from the training data
		
		tr_size = len(train_data[0])
		indices = range(tr_size)
		random.seed(5) #this line is to ensure that you get the same shuffled order everytime
		random.shuffle(indices)
		train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
		
		feature_vec = get_feature_vectors(train_data[0], bin_feats);
		
		i = 0;
		j = 0;
		
		global Bias;
		global W;
		
		while i < num_iter:
			j = 0;
			while j <= len(feature_vec)-1:
				X = feature_vec[j];
				Y = train_data[1][j];
				
				if np.dot(X,W) + Bias <= 0:
				#if np.dot(X,W) <= 0:
					sign = -1;
				else: 
					sign = 1;
				
				if Y != sign :	
					k = 0;
					while k < vocab_size:
						W[k] = W[k] + lr*Y*X[k];
						k  = k + 1 ;
					
					Bias = Bias + lr*Y;	
				
				j = j + 1;
			i = i + 1; 	
		
	def predict(self, test_x):
	#TO DO: Compute and return the output for the given test inputs
		

		text_l = [0] * len(test_x);
		feature_vec = get_feature_vectors(test_x , bin_feats);	
		i = 0;	
		
		while i <= len(feature_vec)-1:
			X = feature_vec[i];
			
			if np.dot(X,W) + Bias <= 0:
				sign = -1;
			else:
				sign = 1; 
			text_l[i] = sign;
			i = i + 1;
		#print(text_l);
		return text_l;
		
class AveragedPerceptron(BinaryClassifier):
	def __init__(self, args):
        #TO DO: Initialize parameters here
		global f_dim, vocab_size, num_iter, lr, bin_feats, Bias, W;
		
		f_dim = args.f_dim;
		vocab_size = args.vocab_size;
		num_iter = args.num_iter;
		lr = args.lr;
		bin_feats =args.bin_feats;
		W = [0.] * vocab_size;
		Bias = 0;

	def fit(self, train_data):
        #TO DO: Learn the parameters from the training data

		tr_size = len(train_data[0])
		indices = range(tr_size)
		random.seed(5) #this line is to ensure that you get the same shuffled order everytime
		random.shuffle(indices)
		train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])

		feature_vec = get_feature_vectors(train_data[0], bin_feats);
		survival = 1;	

		global Bias;
		global W;
		i = 0;
		j = 0;

		Wtemp = [0.] * vocab_size;

		while i < num_iter:
		
			j = 0;
			while j <= len(feature_vec)-1:
				X = feature_vec[j];
				Y = train_data[1][j];
				
				if np.dot(X,W) + Bias <= 0:
				#if np.dot(X,W) <= 0:
					sign = -1;
				else: 
					sign = 1;
				
				if Y != sign :	
					k = 0;
					while k < vocab_size:
						W[k] = (W[k] + lr*Y*X[k] + survival*W[k])/(survival + 1);
						k = k + 1;
					survival = 1;
		
					Bias = Bias + lr*Y;	
				else:
					survival = survival + 1;
				j = j + 1;
			i = i + 1; 	
        
	def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs

		text_l = [0] * len(test_x);
		feature_vec = get_feature_vectors(test_x , bin_feats);	

		i = 0;	
		while i <= len(feature_vec)-1:
			X = feature_vec[i];
			
			if np.dot(X,W) + Bias <= 0:
				sign = -1;
			else:
				sign = 1; 
			text_l[i] = sign;
			i = i + 1;
		#print(text_l);
		return text_l;
		
