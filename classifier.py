import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, f1_score	
from sklearn import svm
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import RTPmining

#Data = list of MSS, Omega = candidate list of lists, g = gap
def create_binary_matrix(Data, Omega, max_g):
	binary_matrix = np.zeros((len(Data),len(Omega)))
	for i in range(0,len(Data)):
		for j in range(0,len(Omega)):
				present = RTPmining.recent_temporal_pattern(Data[i], Omega[j], max_g)
				if(present):
					binary_matrix[i,j] = 1
				else:
					binary_matrix[i,j]= 0
	return binary_matrix

def learn_classifier(opts, train_data, train_labels, test_data, test_labels):
	if opt.classifier == 'svm':
		return learn_svm(train_data, train_labels, test_data, test_labels)
	elif opts.classifier == 'dt':
		return learn_decision_tree(train_data, train_labels, test_data, test_labels)
	elif opts.classifier == 'lr':
		return learn_logistic_regression(train_data, train_labels, test_data, test_labels)
	elif opts.classifier == 'nb':
		return learn_naive_bayes(train_data, train_labels, test_data, test_labels)
	elif opts.classifier == 'knn':
		return learn_knn(train_data, train_labels, test_data, test_labels)

def learn_svm(train_data, train_labels, test_data, test_labels):
	print ("SVM Learning........")
	clf = svm.SVC(probability=True)
	param_grid = {"gamma" : [1, 1e-1, 1e-2, 1e-3], "C" : [1, 10, 100]}
	clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, n_jobs=5)
	clf.fit(train_data, train_labels)
	print (clf.best_params_)
	# clf = clf.fit(train_data, train_labels)
	train_predicted = clf.predict(train_data)
	test_predicted_prob = clf.predict_proba(test_data)
	accuracy = accuracy_score(train_labels, train_predicted)
	print ("training accuracy:", accuracy)
	test_predicted = clf.predict(test_data)
	accuracy = accuracy_score(test_labels, test_predicted)
	print ("test accuracy:", accuracy)
	conf_matrix = metrics.confusion_matrix(test_labels,test_predicted)
	print (conf_matrix)
	return train_predicted, test_predicted, test_predicted_prob

def learn_decision_tree(train_data,train_labels,test_data,test_labels):
	print "DECISION TREE Learning........"
	clf = DecisionTreeClassifier().fit(train_data, train_labels)
	
	train_predicted = clf.predict(train_data)
	test_predicted_prob = clf.predict_proba(test_data)
	accuracy = accuracy_score(train_labels, train_predicted)
	print ("training accuracy:", accuracy)
	test_predicted = clf.predict(test_data)
	accuracy = accuracy_score(test_labels, test_predicted)
	print ("test accuracy:", accuracy)
	conf_matrix = metrics.confusion_matrix(test_labels,test_predicted)
	print (conf_matrix)
	return train_predicted, test_predicted, test_predicted_prob

def learn_logistic_regression(train_data,train_labels,test_data,test_labels):
	print "LOGISTIC REGRESSION Learning........"
	clf = linear_model.LogisticRegression().fit(train_data, train_labels)

	train_predicted = clf.predict(train_data)
	test_predicted_prob = clf.predict_proba(test_data)
	accuracy = accuracy_score(train_labels, train_predicted)
	print ("training accuracy:", accuracy)
	test_predicted = clf.predict(test_data)
	accuracy = accuracy_score(test_labels, test_predicted)
	print ("test accuracy:", accuracy)
	conf_matrix = metrics.confusion_matrix(test_labels,test_predicted)
	print (conf_matrix)
	return train_predicted, test_predicted, test_predicted_prob

def learn_naive_bayes(train_data,train_labels,test_data,test_labels):
	print "NAIVE BAYES Learning........"
	clf = KNeighborsClassifier(5).fit(train_data, train_labels)
	
	train_predicted = clf.predict(train_data)
	test_predicted_prob = clf.predict_proba(test_data)
	accuracy = accuracy_score(train_labels, train_predicted)
	print ("training accuracy:", accuracy)
	test_predicted = clf.predict(test_data)
	accuracy = accuracy_score(test_labels, test_predicted)
	print ("test accuracy:", accuracy)
	conf_matrix = metrics.confusion_matrix(test_labels,test_predicted)
	print (conf_matrix)
	return train_predicted, test_predicted, test_predicted_prob

def learn_knn(train_data,train_labels,test_data,test_labels):
	print "KNN Learning........"
	clf = GaussianNB().fit(train_data, train_labels)
	
	train_predicted = clf.predict(train_data)
	test_predicted_prob = clf.predict_proba(test_data)
	accuracy = accuracy_score(train_labels, train_predicted)
	print ("training accuracy:", accuracy)
	test_predicted = clf.predict(test_data)
	accuracy = accuracy_score(test_labels, test_predicted)
	print ("test accuracy:", accuracy)
	conf_matrix = metrics.confusion_matrix(test_labels,test_predicted)
	print (conf_matrix)
	return train_predicted, test_predicted, test_predicted_prob