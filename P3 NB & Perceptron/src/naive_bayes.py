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
import math
import random

class NaiveBayes(BinaryClassifier):

	def __init__(self, args):
	#TO DO: Initialize parameters here
		global f_dim, vocab_size, num_iter, lr, bin_feats, W_plus, W_minus, Pc_plus, Pc_minus;
		f_dim = args.f_dim;
		vocab_size = 10000;
		num_iter = args.num_iter;
		lr = args.lr;
		bin_feats = args.bin_feats;
		W_plus = [0.] * vocab_size;	
		W_minus = [0.] * vocab_size;
		Pc_plus = 0.0;
		Pc_minus = 0.0;

	def fit(self, train_data):
	#TO DO: Learn the parameters from the training data
		global W_plus, W_minus, Pc_plus, Pc_minus;
	
		tr_size = len(train_data[0])
		indices = range(tr_size)
		random.seed(5) #this line is to ensure that you get the same shuffled order everytime
		random.shuffle(indices)
		train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])		
		
		feature_vec = get_feature_vectors(train_data[0], bin_feats);
		c_plus = train_data[1].count(1);
		c_minus = train_data[1].count(-1);
		Pc_plus = c_plus/c_plus+c_minus;
		Pc_minus = c_minus/c_plus+c_minus;
		total_plus = 0;
		total_minus = 0;
		
		i = 0;
		j = 0;
		while i <= len(feature_vec)-1 :
			count = 0;
			j = 0;
			#while j <= len(feature_vec[i])-1 :
			while j < vocab_size:
				count = count + feature_vec[i][j];
				j = j + 1;
			if train_data[1][i] == 1 :
				total_plus = total_plus + count;
			else:
				total_minus = total_minus + count;
			i = i + 1;

		i = 0;
		j = 0;
		while j < vocab_size :
			count_plus = 0;
			count_minus = 0;
			i = 0;
			while i <= len(feature_vec)-1 :
				if train_data[1][i] == 1 :
					count_plus = count_plus + feature_vec[i][j];
				else :
					count_minus = count_minus + feature_vec[i][j];
				i = i + 1;
			
			if count_plus == 0:
				W_plus[j] = math.log(1/(total_plus + vocab_size + 1));
			else:
				W_plus[j] = math.log(count_plus/total_plus);
			if count_minus == 0:
				W_minus[j] = math.log(1/(total_minus + vocab_size + 1));
			else:
				W_minus[j] = math.log(count_minus/total_minus);
			j = j + 1;
			

	def predict(self, test_x):
	#TO DO: Compute and return the output for the given test inputs
		test_l = [0] * len(test_x);
		feature_vec = get_feature_vectors(test_x, bin_feats);
		
		i = 0;
		j = 0;
		while i <= len(feature_vec)-1:
			P_plus = 0;
			P_minus = 0;
			j = 0;
			
			while j < vocab_size:
				if feature_vec[i][j] != 0:
					P_plus = P_plus + W_plus[j];
					P_minus = P_minus + W_minus[j];
				j = j + 1;
			
			if P_plus*Pc_plus > P_minus*Pc_minus:
				test_l[i] = 1;
			else:
				test_l[i] = -1;	
			i = i + 1;
		#print(test_l);
		return test_l;




