import pandas as pd
import numpy as np
import pickle
import sys
import random
from sklearn.metrics import recall_score, roc_curve, auc, accuracy_score, confusion_matrix, precision_score, f1_score

import TemporalAbstraction
import RTPmining
import classifier
from Config import Options

def store_patterns(i, trainC1, trainC0, opts):

	C1_patterns = RTPmining.pattern_mining(trainC1, opts.max_g, opts.sup_pos*len(trainC1), opts)
	print("Total # patterns from positive:", len(C1_patterns))
	C0_patterns = RTPmining.pattern_mining(trainC0, opts.max_g, opts.sup_neg*len(trainC0), opts)
	print("Total # patterns from negative:", len(C0_patterns))
	
	############## Writing patterns to the files #################
	C1_pos_file = open(opts.patterns_path + 'C1_pos_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.txt','w')
	C0_neg_file = open(opts.patterns_path + 'C0_neg_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.txt','w')
	for p in C1_patterns:
		C1_pos_file.write(p.describe())
	for p in C0_patterns:
		C0_neg_file.write(p.describe())

	pos_fname = opts.patterns_path + 'C1_pos_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.pckl'
	neg_fname = opts.patterns_path + 'C0_neg_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.pckl'

	f = open(pos_fname, 'wb')
	pickle.dump(C1_patterns, f)
	f.close()
	f = open(neg_fname, 'wb')
	pickle.dump(C0_patterns, f)
	f.close()

	return C1_patterns, C0_patterns

def load_patterns(i, opts):
	pos_fname = opts.patterns_path + 'C1_pos_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.pckl'
	neg_fname = opts.patterns_path + 'C0_neg_'+opts.alignment+'_fold'+str(i)+'_'+str(opts.early_prediction)+'.pckl'
	f = open(pos_fname, 'rb')
	C1_patterns = pickle.load(f)
	f.close()
	f = open(neg_fname, 'rb')
	C0_patterns = pickle.load(f)
	f.close()
	return C1_patterns, C0_patterns

def random_subset(iterator, K):
	result = []
	N = 0
	for item in iterator:
		N += 1
		if len( result ) < K:
			result.append( item )
		else:
			s = int(random.random() * N)
			if s < K:
				result[ s ] = item
	return result


def make_MSS(pos_events, neg_events, opts):
	if opts.early_prediction > 0 and opts.alignment == 'right':
		pos_cut = pos_events[pos_events.EventTime - pos_events[opts.timestamp_variable] >= opts.early_prediction * 60]
		neg_cut = neg_events[neg_events.LastMinute - neg_events[opts.timestamp_variable] >= opts.early_prediction * 60]
		if opts.observation_window:
			pos_cut = pos_cut[pos_cut.EventTime - pos_cut[opts.timestamp_variable] <= 60 * (opts.observation_window + opts.early_prediction)]
			neg_cut = neg_cut[neg_cut.LastMinute - neg_cut[opts.timestamp_variable] <= 60 * (opts.observation_window + opts.early_prediction)]

	elif opts.observation_window and opts.alignment == 'left':
		pos_cut = pos_events[pos_events[opts.timestamp_variable] <= opts.early_prediction * 60]
		neg_cut = neg_events[neg_events[opts.timestamp_variable] <= opts.early_prediction * 60]
			
	if opts.settings == 'trunc':
		pos_events = pos_cut
		neg_events = neg_cut

	if len(neg_events[opts.unique_id_variable].unique()) > len(pos_events[opts.unique_id_variable].unique()):
		neg_id = random_subset(neg_events[opts.unique_id_variable].unique(), len(pos_events[opts.unique_id_variable].unique()))
		neg_events = neg_events[neg_events[opts.unique_id_variable].isin(neg_id)]
	if len(pos_events[opts.unique_id_variable].unique()) > len(neg_events[opts.unique_id_variable].unique()):
		pos_id = random_subset(pos_events[opts.unique_id_variable].unique(), len(neg_events[opts.unique_id_variable].unique()))
		pos_events = pos_events[pos_events[opts.unique_id_variable].isin(pos_id)]

	for f in opts.numerical_feat:
		pos_events.loc[:,f], neg_events.loc[:,f] = TemporalAbstraction.abstraction_alphabet(pos_events[f], neg_events[f])

	if opts.settings == 'entire':
		for f in opts.numerical_feat:
			pos_cut.loc[:,f], neg_cut.loc[:,f] = TemporalAbstraction.abstraction_alphabet(pos_cut[f], neg_cut[f])


	MSS_positive = []
	grouped = pos_events.groupby(opts.unique_id_variable)
	for name, group in grouped:
		group = group.sort_values([opts.timestamp_variable])
		MSS_positive.append(TemporalAbstraction.MultivariateStateSequence(group, opts))
	print(len(MSS_positive))

	MSS_negative = []
	grouped = neg_events.groupby(opts.unique_id_variable)
	for name, group in grouped:
		group = group.sort_values([opts.timestamp_variable])
		MSS_negative.append(TemporalAbstraction.MultivariateStateSequence(group, opts))
	print(len(MSS_negative))

	f = open('RTP_log/MSS_pos_'+opts.alignment+'_'+str(opts.early_prediction)+'.pckl', 'wb')
	pickle.dump(MSS_positive, f)
	f.close()
	f = open('RTP_log/MSS_neg_'+opts.alignment+'_'+str(opts.early_prediction)+'.pckl', 'wb')
	pickle.dump(MSS_negative, f)
	f.close()

	return MSS_positive, MSS_negative

def load_MSS(opts):
	f = open('RTP_log/MSS_pos_'+opts.alignment+'_'+str(opts.early_prediction)+'.pckl', 'rb')
	MSS_positive = pickle.load(f)
	f.close()
	f = open('RTP_log/MSS_neg_'+opts.alignment+'_'+str(opts.early_prediction)+'.pckl', 'rb')
	MSS_negative = pickle.load(f)
	f.close()

	return MSS_positive, MSS_negative

def pred_5fold(MSS_positive, MSS_negative, opts):
	if opts.alignment == 'right':
		if len(MSS_negative) > len(MSS_positive):
			MSS_negative = random_subset(MSS_negative, len(MSS_positive))

	print ("size of negative data:", len(MSS_negative))
	print ("size of negative data:", len(MSS_positive))

	C0_subset_size = len(MSS_negative)//opts.num_folds
	C1_subset_size = len(MSS_positive)//opts.num_folds
	
	test_pred, test_pred_prob, train_pred, test_labels = [], [],[], []
	for i in range(opts.num_folds):
		print( "***************** FOLD ", i+1, "*****************")
		trainC0 = MSS_negative[:i*C0_subset_size] + MSS_negative[(i+1)*C0_subset_size:]
		trainC1 = MSS_positive[:i*C1_subset_size] + MSS_positive[(i+1)*C1_subset_size:]

		if opts.settings == 'trunc':
			testC0 = MSS_negative[i*C0_subset_size:][:C0_subset_size]
			testC1 = MSS_positive[i*C1_subset_size:][:C1_subset_size]

		
		print ("Size of negative and negative training:", len(trainC1), len(trainC0))
		print( "Size of negative and negative test:", len(testC1), len(testC0))

		# ---- either store patterns or load the dumped ones
		C1_patterns, C0_patterns = store_patterns(i, trainC1, trainC0, opts)
		# C1_patterns, C0_patterns = load_patterns(i, opts)
		
		all_patterns = list(C1_patterns)
		for j in range(0,len(C0_patterns)):
			if not any((x == C0_patterns[j]) for x in all_patterns):
				all_patterns.append(C0_patterns[j])
		print ("number of all patterns:", len(all_patterns))

		train_data = list(trainC1)
		train_data.extend(trainC0)
		train_labels = list(np.ones(len(trainC1)))
		train_labels.extend(np.zeros(len(trainC0)))

		test_data = list(testC1)
		test_data.extend(testC0)
		test_labels = list(np.ones(len(testC1)))
		test_labels.extend(np.zeros(len(testC0)))

		train_binary_matrix = pd.DataFrame(classifier.create_binary_matrix(train_data, all_patterns, opts.max_g))
		test_binary_matrix = pd.DataFrame(classifier.create_binary_matrix(test_data, all_patterns, opts.max_g))
		
		trp, tsp, tsp_prob = classifier.learn_svm(train_binary_matrix, train_labels, test_binary_matrix, test_labels)
		print (len(tsp))
		test_labels.extend(test_labels)
		test_pred.extend(tsp)
		train_pred.extend(trp)
		for each in tsp_prob:
			test_pred_prob.append(each[1])
		print(len(tsp_prob))

	print(confusion_matrix(test_labels, test_pred))
	accuracy = accuracy_score(test_labels, test_pred)
	precision = precision_score(test_labels, test_pred)
	recall = recall_score(test_labels, test_pred)
	f_measure = f1_score(test_labels, test_pred)
	fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)
	roc_auc = auc(fpr, tpr)
	return accuracy, precision, recall, f_measure, roc_auc


if __name__ == "__main__":
	opts = Options()
	pos_events = pd.read_csv(opts.ts_pos_filepath)
	pos_events.loc[:,'EventTime'] = pos_events.ShockTime 
	neg_events = pd.read_csv(opts.ts_neg_filepath)
	# neg_events.loc[:,'LastMinute'] = neg_events.groupby(opts.unique_id_variable).tail(1)[opts.timestamp_variable]
	# neg_events.LastMinute = neg_events.groupby(opts.unique_id_variable)['LastMinute'].bfill()
	# MSS_pos, MSS_neg = make_MSS(pos_events, neg_events, opts)
	MSS_pos, MSS_neg = load_MSS(opts)
	accuracy, precision, recall, f_measure, auc = pred_5fold(MSS_pos, MSS_neg, opts)

	print([h, accuracy, precision, recall, f_measure, auc])






