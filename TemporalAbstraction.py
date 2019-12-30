#import mysql.connector
import pandas as pd
import numpy as np
import sys

class State:
	def __init__(self, feature, value):
		self.feature = feature
		self.value = value
	def describe(self):
		return "(" + self.feature + "," + str(self.value) + ")"
	def __eq__(self, other):
		if self.feature == other.feature and self.value == other.value:
			return True
		return False
	def __hash__(self):
		return hash((self.feature,self.value))

class StateInterval:
	def __init__(self, feature, value, start, end, opts):
		self.feature = feature
		self.value = value
		self.start = start
		self.end = end
		self.rels = opts.num_tp_rel
	def __gt__(self, state2):
		return self.start > state2.start
	def describe(self):
		return "(" + self.feature + "," + str(self.value) + "," + str(self.start) + "," + str(self.end) + ")"
	def find_relation(self, s2):
		if self.end < s2.start:
			return 'b'
		if self.rels == 2 and self.start <= s2.start and s2.start <= self.end:
			return 'c'
		# for 3 relations	
		if self.rels == 3 and self.start < s2.start and self.end < s2.end:
			return 'p'
		if self.rels == 3 and self.start <= s2.start and self.end >= s2.end:
			return 'c'

def abstraction_alphabet(f1, f0):
	'''
	Discritize the values for a feature based on the whole data
	'''
	lab_values = pd.concat([f1, f0])
	VL_range = np.percentile(lab_values[np.isfinite(lab_values)],10)
	L_range = np.percentile(lab_values[np.isfinite(lab_values)],25)
	N_range = np.percentile(lab_values[np.isfinite(lab_values)],75)
	H_range = np.percentile(lab_values[np.isfinite(lab_values)],90)
	VH_range = np.percentile(lab_values[np.isfinite(lab_values)],100)
	pos = pd.DataFrame(f1)
	pos.loc[f1<VL_range,pos.columns.values[0]] = "VL"
	pos.loc[(f1>=VL_range) & (f1<L_range),pos.columns.values[0]] = "L"
	pos.loc[(f1>=L_range) & (f1<N_range),pos.columns.values[0]] = "N"
	pos.loc[(f1>=N_range) & (f1<H_range),pos.columns.values[0]] = "H"
	pos.loc[(f1>=H_range) & (f1<=VH_range),pos.columns.values[0]] = "VH"
	neg = pd.DataFrame(f0)
	neg.loc[f0<VL_range,neg.columns.values[0]] = "VL"
	neg.loc[(f0>=VL_range) & (f0<L_range),neg.columns.values[0]] = "L"
	neg.loc[(f0>=L_range) & (f0<N_range),neg.columns.values[0]] = "N"
	neg.loc[(f0>=N_range) & (f0<H_range),neg.columns.values[0]] = "H"
	neg.loc[(f0>=H_range) & (f0<=VH_range),neg.columns.values[0]] = "VH"
	return pos, neg

def state_generation(abstracted_lab_values, feature, opts):
	'''
	Given the discritized values of a feature for a specific sample, generate the state intervals.
	Assumption: If the values are missing in between two readings of the same value, we consider the whole interval was maintaining the same value.
	'''
	state_intervals = []
	previous_value = np.nan
	state_start = np.nan
	state_end = np.nan
	for i,val in abstracted_lab_values.iterrows():
		if pd.notnull(val[feature]) and pd.isnull(previous_value):
			previous_value = val[feature]
			state_start = val[opts.timestamp_variable]
			state_end = val[opts.timestamp_variable]
		elif pd.notnull(val[feature]) and (val[feature]==previous_value):
			state_end = val[opts.timestamp_variable]
		elif pd.notnull(val[feature]) and (val[feature]!=previous_value):
			state_intervals.append(StateInterval(feature, previous_value, state_start, state_end, opts))
			previous_value = val[feature]
			state_start = val[opts.timestamp_variable]
			state_end = val[opts.timestamp_variable]
	if pd.notnull(previous_value) and pd.notnull(state_end) and pd.notnull(state_start):
		state_intervals.append(StateInterval(feature, previous_value, state_start, state_end, opts))
	return state_intervals

def MultivariateStateSequence(sequence_data, opts):
	'''
	Takes a sequence of data (for one sample) and returns the MSS
	'''
	MSS = []
	for f in opts.numerical_feat:
		MSS.extend(state_generation(sequence_data, f, opts))
	for f in opts.binary_feat:
		MSS.extend(state_generation(sequence_data, f, opts))
	for f in opts.categorical_feat:
		MSS.extend(state_generation(sequence_data, f, opts))
	MSS.sort(key=lambda x: x.start)
	return MSS

def state_find_matches(mss, state, fi):
	'''
	Given an MSS, find the index of state intervals with the same feature and value as of state, starting from fi index
	'''
	match = []
	for i in range (fi, len(mss)):
		if state.feature == mss[i].feature and state.value == mss[i].value:
			match.append(i)
	return match


def MSS_contains_Pattern(mss, p, i, fi, prev_match):
	'''
	A recursive function that determines whether a sequence contains a pattern or not, based on DEFINITION 2 from Batal et al.
	'''			
	if i >= len(p.states):
		return True, prev_match
	same_state_index = state_find_matches(mss, p.states[i], fi)
	for fi in same_state_index:
		flag = True
		for pv in range(0,len(prev_match)):
			if prev_match[pv].find_relation(mss[fi]) != p.relation[pv][i]:
				flag = False
				break
		if flag:
			prev_match.append(mss[fi])
			contains, seq = MSS_contains_Pattern(mss, p, i+1, 0, prev_match)
			if contains:
				return True, seq
			else:
				del prev_match[-1]
	return False, np.nan

def recent_state_interval(mss, j, g):
	'''
	Determines whether a state interval is recent or not, based on DEFINITION 3 from Batal et al.
	'''
	if mss[len(mss)-1].end - mss[j].end <= g:
		return True
	flag = False
	for k in range(j+1, len(mss)):
		if mss[j].feature == mss[k].feature:
			flag = True
	if not flag:
		return True
	return False

def get_index_in_sequence(mss, e):
	'''
	Finds the index of a state interval in an MSS
	'''
	for i in range(0,len(mss)):
		if mss[i] == e:
			return i
	return -1

def sequences_containing_state(RTPlist, new_s):
	'''
	Given a new state and a list of MSS's, determines which contain this state
	'''
	p_RTPlist = []
	for z in RTPlist:
		for e in z:
			if e.feature == new_s.feature and e.value == new_s.value:
				p_RTPlist.append(z)
				break
	return p_RTPlist

def find_all_frequent_states(D, support, opts):
	'''
	Given all MSS's of a group and their minimum support, finds all frequent states
	'''
	freq_states = []
	for f in opts.numerical_feat:
		for v in opts.num_categories:
			state = State(f, v)
			if len(sequences_containing_state(D, state)) >= support:
				freq_states.append(state)
	for f in opts.binary_feat:
		for v in (0,1):
			state = State(f, v)
			if len(sequences_containing_state(D, state)) >= support:
				freq_states.append(state)
	for f in opts.categorical_feat:
		for v in opts.categorical_feat[f]:
			state = State(f, v)
			if len(sequences_containing_state(D, state)) >= support:
				freq_states.append(state)
	return freq_states