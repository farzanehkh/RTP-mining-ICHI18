class Options:
	def __init__(self):
		# model parameters
		self.max_g = 14
		self.sup_pos = 0.18
		self.sup_neg = 0.18

		# experiment settings
		self.early_prediction = 24
		self.observation_window = 5*24 
		self.alignment = 'right' 
		self.settings = 'trunc'
		self.num_folds = 5

		# directory settings
		self.ts_pos_filepath = '~/codes/Sampling/RTP_cchs_shock.csv'
		self.ts_neg_filepath = '~/codes/Sampling/RTP_cchs_nonshock.csv'
		self.res_path = 'results/'
		self.patterns_path = 'patterns/'

		'''
		Available Classifiers:
			SVM: 'svm'
			Decision Tree: 'dt'
			Naive Bayes: 'nb'
			Logistic Regression: 'lr'
			K-Nearest Neighbors: 'knn'
		'''
		self.classifier = 'svm'

		# number of temporal patterns: {2: before, o-occur, 3: before, overlap, contain}
		self.num_tp_rel = 2

		# dataset features specification
		self.numerical_feat = ['SystolicBP','DiastolicBP','HeartRate','RespiratoryRate','Temperature','PulseOx','BUN','Procalcitonin' ,\
								'WBC' ,'Bands' ,'Lactate' ,'Platelet','Creatinine','MAP','BiliRubin','CReactiveProtein','SedRate' ,'FIO2','OxygenFlow']
		self.num_categories = ['VL', 'L', 'N', 'H', 'VH']
		self.binary_feat = ['Observed_InfectionFlag', 'InflammationFlag', 'OrganFailure']
		self.categorical_feat = {'CurrentLocationTypeCode':['ED', 'NURSE', 'ICU', 'STEPDN']}
		self.unique_id_variable = 'VisitIdentifier'
		self.timestamp_variable = 'MinutesFromArrival'

