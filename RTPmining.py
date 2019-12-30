import pandas as pd
import numpy as np

import TemporalAbstraction

class TemporalPattern:
	def __init__(self, states, relation, RTPlist, p_RTPlist):
		self.states = states
		self.relation = relation
		self.RTPlist = RTPlist
		self.p_RTPlist = p_RTPlist

	def describe(self):
		string = "*****************\n"
		string = string + "states:\n"
		for s in self.states:
			string = string + s.describe()
			string = string + "\n"
		array_str = "relation:\n"
		for row in self.relation:
			for e in row:
				array_str = array_str + e +" "
			array_str = array_str + "\n"
		string = string + array_str
		string = string + "\n"
		string = string + "length of the RTP List:" + str(len(self.RTPlist))
		string = string + "\n"
		string = string + "length of the possible RTP List:" + str(len(self.p_RTPlist))
		string = string + "\n"
		string = string + "*****************"
		string = string + "\n"
		return string

	def __eq__(self, other):
		if len(self.states) != len(other.states):
			return False
		if set(self.states) != set(other.states):
			return False
		k = len(self.states)
		mapping = {}
		for i in range(k):
			indices = [j for j in range(k) if self.states[i]==other.states[j]]
			for idx in indices:
				if idx not in mapping.values():
					mapping[i] = idx
					break
			if i not in mapping:
				return False
		for i in range(k):
			for j in range(i+1,k):
				if self.relation[i][j] != other.relation[min(mapping[i],mapping[j])][max(mapping[i],mapping[j])]:
					return False
		return True

	def __ne__(self, other):
		return (not self.__eq__(other))	

	def __hash__(self):
		return hash((self.states,self.relation))

def build_patterns(R, S, i):
	Pnew_set = []
	if i == len(R):
		P = TemporalPattern(S,R,[],[])
		# print P.describe()
		return [P]
	possible_relations = ['b','c','p']
	if S[0].feature == S[i].feature:
		possible_relations = ['b']
	for idx in range(1,i):
		if R[0][idx] == 'c' and R[idx][i] == 'p':
			if 'b' in possible_relations:
				possible_relations.remove('b')
		if R[0][idx] == 'c' and R[idx][i] == 'c':
			if 'b' in possible_relations:
				possible_relations.remove('b')
			if 'p' in possible_relations:
				possible_relations.remove('p')
		if R[0][idx] == 'p' and R[idx][i] == 'b':
			if 'c' in possible_relations:
				possible_relations.remove('c')
			if 'p' in possible_relations:
				possible_relations.remove('p')
		if R[0][idx] == 'p' and R[idx][i] == 'p':
			if 'c' in possible_relations:
				possible_relations.remove('c')
	for rel in possible_relations:
		R2 = []
		for j in range(0,len(R)):
			R2.append(list(R[j]))
		R2[0][i] = rel
		if rel != 'b':
			Pnew_set.extend(build_patterns(R2, S, i+1))
		else:
			P = TemporalPattern(S,R2,[],[])
			Pnew_set.append(P)
			# print P.describe()
	return Pnew_set

def extend_backward_three(RTP, new_state):               #WITH 3 RELATIONS: b,c,p
	'''
	An extension of the original algorithm to extarct patterns with 3 different relations rather than two.
	'''
	states = RTP.states
	relations = RTP.relation
	k = len(states)
	sPrime = []
	sPrime.append(new_state)
	sPrime[1:] = states
	rPrime = []
	row1 = []
	row1.append("o")
	for i in range(1,(k+1)):
		row1.append("b")
	rPrime.append(row1)
	for i in range(0,k):
		row = ["o"]
		row[1:] = relations[i]
		rPrime.append(row)
	pPrime = TemporalPattern(sPrime,rPrime,[],[])
	newRelation = []
	for i in range(0,len(pPrime.states)):
		newRelation.append(list(pPrime.relation[i]))
	C = [TemporalPattern(pPrime.states[:],newRelation,[],[])]
	X = build_patterns(rPrime,sPrime,1)
	if X != None:
		C.extend(X)
	return C

def extend_backward(RTP, new_state):
	'''
	Extend patterns backward in time by appending a new state to a (k-1)-RTP from previous round. Based on ALGORITHM 1 in Batal et al.
	'''
	states = RTP.states
	relations = RTP.relation
	k = len(states)
	sPrime = []
	sPrime.append(new_state)
	sPrime[1:] = states
	rPrime = []
	row1 = []
	row1.append("o")
	for i in range(1,(k+1)):
		row1.append("b")
	rPrime.append(row1)
	for i in range(0,k):
		row = ["o"]
		row[1:] = relations[i]
		rPrime.append(row)
	pPrime = TemporalPattern(sPrime,rPrime,[],[])
	newRelation = []
	for i in range(0,len(pPrime.states)):
		newRelation.append(list(pPrime.relation[i]))
	C = [TemporalPattern(pPrime.states[:],newRelation,[],[])]
	for i in range(1,k+1):
		if sPrime[0].feature == sPrime[i].feature:
			break
		else:
			rPrime[0][i] = "c"
			pPrime = TemporalPattern(sPrime,rPrime,[],[])
			if not any((x == pPrime) for x in C):
				newRelation = []
				for i in range(0,k+1):
					newRelation.append(list(pPrime.relation[i]))
				C.append(TemporalPattern(pPrime.states[:],newRelation,[],[]))
	return C

def recent_temporal_pattern(mss, p, g):
	'''
	Determines whether a pattern is RTP or not based on DEFINITION 4 in Batal et al.
	'''
	mapping = []
	contains, mapping = TemporalAbstraction.MSS_contains_Pattern(mss, p, 0, 0, mapping)
	if not contains:
		return False
	if not TemporalAbstraction.recent_state_interval(mss, TemporalAbstraction.get_index_in_sequence(mss, mapping[len(p.states)-1]), g):
		return False
	for i in range(0,len(mapping)-1):
		if mapping[i+1].start - mapping[i].end > g:
			return False
	return True

#  
def RTP_support(P, g):
	'''
	This function calculates the support of a recent temporal pattern using DEFINITION 6 in Batal et al.
	'''						
	RTPlist = []
	for Z in P.p_RTPlist:
		if recent_temporal_pattern(Z, P, g):
			RTPlist.append(Z)
	return RTPlist

def counting_phase(candidates, g, support):
	kRTP = []
	for C in candidates:
		C.RTPlist = RTP_support(C, g)
		if len(C.RTPlist) >= support:
			kRTP.append(C)
	return kRTP

def candidate_generation(D, kRTP, p_states, g, support):
	'''
	Candidate generation function based on ALGORITHM 2 in Batal et al.
	'''
	candidates = []
	for p in kRTP:
		for s in p_states:
			C = extend_backward(p, s)
			for q in range(0, len(C)):
				C[q].p_RTPlist = TemporalAbstraction.sequences_containing_state(p.RTPlist, s)
				if len(C[q].p_RTPlist) >= support:
					if not any((x == C[q]) for x in candidates):
						candidates.append(C[q])
	return candidates


def pattern_mining(D, g, support, opts):
	'''
	The main RTP mining procedure that takes MSS of a group and their corresponding parameters and extracts ALL RTPs
	'''
	one_RTP = []
	freq_states = TemporalAbstraction.find_all_frequent_states(D, support, opts)
	print("number of frequent states are:", len(freq_states))
	for s in freq_states:
		new_pattern = TemporalPattern([s],[['o']],[],[])
		RTPlist = []
		for Z in D:
			interval_matches = TemporalAbstraction.state_find_matches(Z, s, 0)
			if len(interval_matches) != 0:
				if TemporalAbstraction.recent_state_interval(Z, max(interval_matches), g):
					RTPlist.append(Z)
		if len(RTPlist) >= support:
			new_pattern.RTPlist = RTPlist
			new_pattern.p_RTPlist = RTPlist
			one_RTP.append(new_pattern)

	print("the number of one-RTPs:", len(one_RTP))
	K = max(len(z) for z in D)
	kRTP = one_RTP
	Omega = []
	Omega.extend(one_RTP)
	for k in range(1, K+1):
		candidates = candidate_generation(D, kRTP, freq_states, g, support)
		print ("------------------------length of the candidates for", k+1, "pattern is:", len(candidates))
		kRTP = counting_phase(candidates, g, support)
		print ("------------------------length of the kRTP for", k+1, "pattern is:", len(kRTP))
		if len(kRTP) == 0:
			break
		Omega.extend(kRTP)
	print ("number of all patterns found:", len(Omega))
	return Omega
