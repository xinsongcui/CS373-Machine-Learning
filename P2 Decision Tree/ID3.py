##############
# Name: Xinsong Cui
# email: cui102@purdue.edu
# Date: 3/3/2019

import pandas as pd
import numpy as np
import sys
import os
import copy

column = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Relatives', 'IsAlone']


def entropy(freqs):
    """ 
    entropy(p) = -SUM (Pi * log(Pi))
    >>> entropy([10.,10.])
    1.0
    >>> entropy([10.,0.])
    0
    >>> entropy([9.,3.])
    0.811278
    """
    all_freq = sum(freqs)
    entropy = 0 
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entropy += -prob * np.log2(prob)
    return entropy
    
def infor_gain(before_split_freqs, after_split_freqs):
	"""
	gain(D, A) = entropy(D) - SUM ( |Di| / |D| * entropy(Di) )
	>>> infor_gain([9,5], [[2,2],[4,2],[3,1]])
    	0.02922
    	"""
	gain = entropy(before_split_freqs)
	overall_size = sum(before_split_freqs)
	for freq in after_split_freqs:
		ratio = sum(freq) * 1.0 / overall_size
		gain -= ratio * entropy(freq)
	return gain
    
    
class Node(object):
	def __init__(self, isleaf, attribute, label, info_gain, middle):
		self.isleaf = isleaf
		self.attribute = attribute
		self.label = label
		self.info_gain =  info_gain
		self.middle= middle
		self.left = None
		self.right = None
		
class Tree(object):
	def __init__(self):
		self.nodes = 0
		#self.test_file = test_file
		#self.model = model
		pass

	def counting_Nodes(self, root):	
		if root.isleaf != 1:
			if root.left.isleaf != 1:
				self.nodes = self.nodes + 1
				self.counting_Nodes(root.left)
			if root.right.isleaf != 1:
				self.ndoes = self.nodes + 1
				self.counting_Nodes(root.right)
		self.nodes = self.nodes+1
		return

	def ID3_util(self, data, index):
		max_gain, max_gain_l, max_gain_r, middle_point = 0.0, None, None, None
		
		for j in np.unique(data[:, index]):
			l, r, sl, sr, dl, dr = [], [], 0, 0, 0, 0
			for a in data:
				if a[index] <= j:
					l.append(a)
				else:
					r.append(a)

			left_data = np.asarray(l)
			right_data = np.asarray(r)

			if len(left_data)>0 and len(right_data)>0:
				for k in left_data[:, -1]:
					if k == 1:
						sl = sl + 1
					else:
						dl = dl + 1
				for q in right_data[:, -1]:
					if q == 1:
						sr = sr + 1
					else:
						dr = dr + 1
				gain = infor_gain([(dl+dr),(sl+sr)],[[dl, sl], [dr, sr]]);
				if gain > max_gain:
					max_gain, max_gain_l, max_gain_r, middle_point = gain,left_data, right_data, j
					
		return 	max_gain, max_gain_l, max_gain_r, middle_point

	def ID3(self, root, data, column):
		attribute, label, info_gain, left_data, right_data, middle = None, None, 0.0, None, None, None
		for index in range(len(column)):
			g, l, r, m = self.ID3_util(data, index)
			if g > info_gain:
				info_gain = g
				attribute = column[index]
				left_data = l
				right_data = r 
				middle = m
		if info_gain == 0.0:
			root.info_gain = -1
		else:	
		 	root.isleaf, root.attribute, root.label, root.info_gain, root.middle = 0, attribute, label, info_gain, middle

		return root, left_data, right_data
	
	def build_vanilla(self, train_file, train_label, train_size):
		train_f = pd.read_csv(train_file, delimiter = ',', index_col=None, engine='python')
		train_l = pd.read_csv(train_label, delimiter = ',', index_col=None, engine='python')
		train = pd.concat([train_f, train_l], axis=1)
		train = train.as_matrix()
		data = copy.deepcopy(train)
		data = data[0:int(len(data) * int(train_size) /100), :]
		
		root = Node(0, None, None, 0.0, None)
		
		self.build_tree(root, data) 
		return root, data

	def build_depth(self, train_file, train_label, train_size, depth):
		train_f = pd.read_csv(train_file, delimiter = ',', index_col=None, engine='python')
		train_l = pd.read_csv(train_label, delimiter = ',', index_col=None, engine='python')
		train = pd.concat([train_f, train_l], axis=1)
		train = train.as_matrix()
		data = copy.deepcopy(train)
		data = data[0:int(len(data) * int(train_size) /100), :]
		
		root = Node(0, None, None, 0.0, None)
		
		self.build_tree_with_depth(root, data, depth) 
		return root, data

	def build_tree_with_depth(self, root, data, depth):
		if all(index == data[0,-1] for index in data[:,-1]):
			root.isleaf = 1
			root.label = data[0,-1]
			return 
		if depth == 0:
			root.isleaf = 1
			if((data[:,-1] == 0).sum() > (data[:,-1] == 1).sum()):
				root.label = 0
			else: 
				root.label = 1
			return

		tree, left, right = self.ID3(root, data, column)
		
		if tree.info_gain == -1:
			root.isleaf = 1
			root.label = 0
			return		
	
		tree.left = Node(0, None, None, 0.0, None) 
		tree.right = Node(0, None, None, 0.0, None) 
		self.build_tree_with_depth(tree.left, left, depth-1)
		self.build_tree_with_depth(tree.right, right, depth-1)

	def build_tree(self, root, data):
		#if data.shape[0] < 0:
		#	root.isleaf = 1
		#	self.label = 0
		#	return 
		if all(index == data[0,-1] for index in data[:,-1]):
			root.isleaf = 1
			root.label = data[0, -1]
			return 

		tree, left, right = self.ID3(root, data, column)

		if tree.info_gain == -1:
			root.isleaf = 1
			root.label = 0
			return 
		
		tree.left = Node(0, None, None, 0.0, None) 
		tree.right = Node(0, None, None, 0.0, None) 
		self.build_tree(tree.left, left)
		self.build_tree(tree.right, right)

	def test_tree(self, root, test_file, test_label, data):
		test_f = pd.read_csv(test_file, delimiter = ',', index_col=None, engine='python')
		test_l = pd.read_csv(test_label, delimiter = ',', index_col=None, engine='python')
		test_f= test_f.as_matrix()
		test_l = test_l.as_matrix()
		test_data = copy.deepcopy(test_f)
		test_label = copy.deepcopy(test_l)
		predict = list()
		predict_true = 0.0
		predict_false = 0.0
		
		for row in test_data:
			temp = root
			while temp.isleaf != 1:
				if row[self.find_index(temp.attribute)] <= temp.middle:
					temp = temp.left	
				else:
					temp = temp.right
					
			predict.append(temp.label)
	
		for index in range(len(test_data)):
			if test_label[index]  == predict[index]:
				predict_true += 1.0
			else:
				predict_false += 1.0
		score = predict_true/(predict_true + predict_false)
		return score

	def valid_tree(self, root, train_file, train_label, valid_size):
		train_f = pd.read_csv(train_file, delimiter = ',', index_col=None, engine='python')
		train_l = pd.read_csv(train_label, delimiter = ',', index_col=None, engine='python')
		train = pd.concat([train_f, train_l], axis=1)
		train = train.as_matrix()
		valid = copy.deepcopy(train)
		valid = valid[int(len(valid) * int(100-valid_size) /100):, :]
		predict = list()
		predict_true = 0.0
		predict_false = 0.0
		
	
		for row in valid:
			temp = root
			while temp.isleaf != 1:
				if row[self.find_index(temp.attribute)] <= temp.middle:
					temp = temp.left
				else:
					temp = temp.right
			predict.append(temp.label)

		for index in range(len(valid[:,-1])):
			if valid[index,-1] == predict[index]:
				predict_true += 1.0
			else:
				predict_false += 1.0
		score = predict_true/(predict_true + predict_false)
		return score
	def train_tree(self, root, data):
		predict = list()
		predict_true = 0.0
		predict_false = 0.0
		
	
		for row in data:
			temp = root
			while temp.isleaf != 1:
				if row[self.find_index(temp.attribute)] <= temp.middle:
					temp = temp.left
				else:
					temp = temp.right
			predict.append(temp.label)

		for index in range(len(data[:,-1])):
			if data[index,-1] == predict[index]:
				predict_true += 1.0
			else:
				predict_false += 1.0
		score = predict_true/(predict_true + predict_false)
		return score
			
	def find_index(self, attribute):
		for index in range(len(column)):
			if column[index] == attribute: 
				return index
			
	


if __name__ == "__main__":
	# parse arguments
	if len(sys.argv) >= 5:
		train_file = sys.argv[1]
		test_file = sys.argv[2]
		train_model = sys.argv[3]
		train_size = sys.argv[4]
	if len(sys.argv) >= 6:
		valid_size = sys.argv[5]
	if len(sys.argv) == 7:
		muti_func = sys.argv[6]

	train_label = train_file + "/titanic-train.label"
	train_file += "/titanic-train.data"
	test_label = test_file + "/titanic-test.label"
	test_file += "/titanic-test.data"

	if train_model == "vanilla":
		tree = Tree()
		root, data = tree.build_vanilla(train_file, train_label, train_size)
		tree.counting_Nodes(root)
		#print(tree.nodes)
		train_score = tree.train_tree(root, data)
		test_score = tree.test_tree(root, test_file, test_label, data)
		print("Train set accuracy: %.4f" % train_score)
		print("Test set accuracy: %.4f" % test_score)

	if train_model == "depth":
		tree = Tree()
		root, data = tree.build_depth(train_file, train_label, train_size, int(muti_func))		
		tree.counting_Nodes(root)
		#print(tree.nodes)
		train_score = tree.train_tree(root, data)
		valid_score = tree.valid_tree(root, train_file, train_label, int(valid_size))
		test_score = tree.test_tree(root, test_file, test_label, data)
		print("Train set accuracy: %.4f" % train_score)
		print("Validation set accuracy: %.4f" % valid_score)
		print("Test set accuracy: %.4f" % test_score)

	if train_model == "min_split":
		tree = Tree()
		root, data = tree.build_vanilla(train_file, train_label, train_size)	
		train_score = tree.train_tree(root, data)
		valid_score = tree.valid_tree(root, train_file, train_label, int(valid_size))
		test_score = tree.test_tree(root, test_file, test_label, data)
		print("Train set accuracy: %.4f" % train_score)
		print("Validation set accuracy: %.4f" % valid_score)
		print("Test set accuracy: %.4f" % test_score)

	if train_model == "prune":
		tree = Tree()
		root, data = tree.build_vanilla(train_file, train_label, train_size)
		train_score = tree.train_tree(root, data)
		test_score = tree.test_tree(root, test_file, test_label, data)
		print("Train set accuracy: %.4f" % train_score)
		print("Test set accuracy: %.4f" % test_score)
