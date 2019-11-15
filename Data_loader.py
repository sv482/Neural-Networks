# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:17:32 2019

@author: sushv
"""

import json
import random
import os as dir

dir.getcwd()
dir.chdir('C:/Users/sushv/Desktop/Assignments/NLP/Project 3/Project3FA19')

#Data Fetching 
def fetch_data():
	with open('training.json') as training_f:
		training = json.load(training_f)
	with open('validation.json') as valid_f:
		validation = json.load(valid_f)
	# If needed you can shrink the training and validation data to speed up somethings but this isn't always safe to do by setting k < 16000
	# k = #fill in
	# training = random.shuffle(training)
	# validation = random.shuffle(validation)
	# training, validation = training[:k], validation[:(k // 10)]
	tra = []
	val = []
	for elt in training:
		tra.append((elt["text"].split(),int(elt["stars"]-1)))
	for elt in validation:
		val.append((elt["text"].split(),int(elt["stars"]-1)))
	return tra, val


