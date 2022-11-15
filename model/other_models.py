import os
import sys
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

class CalibratedRF(BaseEstimator):
	"""
	Calibrated classifier using a RandomForestClassifier classifier as the base
	"""

	def __init__(self, n_estimators = 200, max_features = 'log2' , max_depth = 20, min_samples_split = 2, min_samples_leaf = 2, method = "sigmoid",cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42), n_jobs = 10):
		super().__init__()
		
		self.n_estimators = n_estimators
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.max_depth = max_depth
		self.max_features = max_features
		self.method = method
		self.cv = cv
		self.model = None
		self.classes_ = [0,1]
		self.n_jobs = n_jobs

	def fit(self,X,y,sample_weight=None,**kwargs):
		"""Fit the calibrated model.

		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels

		sample_weight : 1d array-like
			Sample weights. If None, then samples are equally weighted.
		
		fit_params : dict
			Additional fit parameters

		Returns
		-------
		model : object
			Returns an instance of the calibrated model.
		"""
		self.model = CalibratedClassifierCV(base_estimator =RandomForestClassifier(n_estimators = self.n_estimators, max_features = self.max_features , max_depth = self.max_depth, min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf, random_state=42), method = self.method, cv = self.cv, n_jobs = self.n_jobs)
		self.model.fit(X,y,sample_weight,**kwargs)
		self._estimator_type = self.model._estimator_type
		return self.model

	def predict(self, X):
		"""
		Returns the predictons of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		C : 1d array-like
			Predictions
		"""
		return self.model.predict(X)

	def predict_proba(self, X):
		"""
		Returns the predicted probabilities of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 2d array-like
			Predicted probabilities
		"""
		return self.model.predict_proba(X)
	
	def decision_function(self, X):
		"""
		Returns the probabilities given input samples are positive
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 1d array-like
			Predicted probabilities
		"""
		return self.predict_proba(X)[:,1]

class CalibratedBinaryXGBoost(BaseEstimator):
	"""
	Calibrated classifier using a XGBClassifier classifier as the base
	"""

	def __init__(self, n_estimators = 100, max_depth = 20, learning_rate = 0.1, objective = 'binary:logistic', booster = 'gblinear', tree_method = 'hist', gamma = 0, eval_metric='auc', method = 'sigmoid', random_state = 42, cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42), n_jobs=10):
		super().__init__()

		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.learning_rate = learning_rate
		self.objective = objective
		self.booster = booster
		self.tree_method = tree_method
		self.gamma = gamma
		self.eval_metric = eval_metric
		self.method = method
		self.random_state = random_state
		self.cv = cv
		self.classes_ = [0,1]
		self.n_jobs = n_jobs

	def fit(self,X,y,sample_weight=None,**kwargs):
		"""Fit the calibrated model.

		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels

		sample_weight : 1d array-like
			Sample weights. If None, then samples are equally weighted.
		
		fit_params : dict
			Additional fit parameters

		Returns
		-------
		model : object
			Returns an instance of the calibrated model.
		"""

		self.model = CalibratedClassifierCV(base_estimator = XGBClassifier(n_estimators = self.n_estimators, max_depth = self.max_depth, learning_rate = self.learning_rate, objective = self.objective, booster = self.booster, tree_method = self.tree_method, eval_metric=self.eval_metric, verbosity=0, use_label_encoder=False),
													method = self.method, cv = self.cv, n_jobs = self.n_jobs)
		self.model.fit(X,y,sample_weight,**kwargs)
		return self.model

	def predict(self, X):
		"""
		Returns the predictons of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		C : 1d array-like
			Predictions
		"""
		return self.model.predict(X)

	def predict_proba(self, X):
		"""
		Returns the predicted probabilities of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 2d array-like
			Predicted probabilities
		"""
		return self.model.predict_proba(X)
	def decision_function(self, X):
		"""
		Returns the probabilities given input samples are positive
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 1d array-like
			Predicted probabilities
		"""
		return self.predict_proba(X)[:,1]


class CalibratedBinaryLog(BaseEstimator):
	"""
	Calibrated classifier using a LogisticRegression classifier as the base
	"""

	def __init__(self, penalty='l2', C=1.0, solver = 'lbfgs', multi_class = 'ovr', method = "isotonic", max_iter=2000, cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42), n_jobs=10):
		super().__init__()

		self.C = C
		self.penalty = penalty
		self.method = method
		self.max_iter = max_iter
		self.cv = cv
		self.solver = solver
		self.multi_class = multi_class
		self.model = None
		self.n_jobs = n_jobs

	def fit(self,X,y,sample_weight=None,**kwargs):
		"""Fit the calibrated model.

		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels

		sample_weight : 1d array-like
			Sample weights. If None, then samples are equally weighted.
		
		fit_params : dict
			Additional fit parameters

		Returns
		-------
		model : object
			Returns an instance of the calibrated model.
		"""

		self.model = CalibratedClassifierCV(base_estimator =LogisticRegression(C = self.C, penalty = self.penalty, solver = self.solver, multi_class = self.multi_class, max_iter = self.max_iter), method = self.method, cv = self.cv, n_jobs = self.n_jobs)
		self.model.fit(X,y,sample_weight,**kwargs)
		self._estimator_type = self.model._estimator_type
		self.classes_ = self.model.classes_

		return self.model

	def predict(self, X):
		"""
		Returns the predictons of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		C : 1d array-like
			Predictions
		"""
		return self.model.predict(X)

	def predict_proba(self, X):
		"""
		Returns the predicted probabilities of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 2d array-like
			Predicted probabilities
		"""
		return self.model.predict_proba(X)

class CalibratedBinaryKNN(BaseEstimator):
	"""
	Calibrated classifier using a KNeighborsClassifier classifier as the base
	"""

	def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', method = "isotonic",cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42), n_jobs=10):
		super().__init__()

		self.n_neighbors = n_neighbors
		self.p = p
		self.leaf_size = leaf_size
		self.weights = weights
		self.algorithm = algorithm
		self.method = method
		self.cv = cv
		self.model = None
		self.metric = metric

		self.n_jobs = n_jobs

	def fit(self,X,y,sample_weight=None,**kwargs):
		"""Fit the calibrated model.

		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels

		sample_weight : 1d array-like
			Sample weights. If None, then samples are equally weighted.
		
		fit_params : dict
			Additional fit parameters

		Returns
		-------
		model : object
			Returns an instance of the calibrated model.
		"""

		self.model = CalibratedClassifierCV(base_estimator =KNeighborsClassifier(n_neighbors = self.n_neighbors, p = self.p, leaf_size = self.leaf_size, weights = self.weights, algorithm = self.algorithm), method = self.method, cv = self.cv, n_jobs = self.n_jobs)
		self.model.fit(X,y,sample_weight,**kwargs)
		self._estimator_type = self.model._estimator_type
		self.classes_ = self.model.classes_

		return self.model

	def predict(self, X):
		"""
		Returns the predictons of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		C : 1d array-like
			Predictions
		"""
		return self.model.predict(X)

	def predict_proba(self, X):
		"""
		Returns the predicted probabilities of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 2d array-like
			Predicted probabilities
		"""
		return self.model.predict_proba(X)

class CalibratedBinarySVC(BaseEstimator):
	"""
	Calibrated classifier using an SVM classifier as the base
	"""

	def __init__(self, probability = True, C = 0.1, kernel = 'rbf', gamma = 'auto_deprecated', degree = 3, method = "isotonic", max_iter=1000, cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42), n_jobs=10):
		super().__init__()

		self.probability = probability
		self.C = C
		self.kernel = kernel
		self.gamma = gamma
		self.degree = degree

		self.method = method
		self.max_iter = max_iter
		self.cv = cv

		self.model = None

		self.n_jobs = n_jobs

	def fit(self,X,y,sample_weight=None,**kwargs):
		"""Fit the calibrated model.

		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels

		sample_weight : 1d array-like
			Sample weights. If None, then samples are equally weighted.
		
		fit_params : dict
			Additional fit parameters

		Returns
		-------
		model : object
			Returns an instance of the calibrated model.
		"""

		self.model = CalibratedClassifierCV(base_estimator =SVC(C=self.C, kernel = self.kernel, gamma = self.gamma, degree = self.degree, max_iter = self.max_iter, random_state = 42), method = self.method, cv = self.cv, n_jobs = self.n_jobs)
		self.model.fit(X,y,sample_weight,**kwargs)
		self._estimator_type = self.model._estimator_type
		self.classes_= self.model.classes_

		return self.model

	def predict(self, X):
		"""
		Returns the predictons of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		C : 1d array-like
			Predictions
		"""
		return self.model.predict(X)

	def predict_proba(self, X):
		"""
		Returns the predicted probabilities of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 2d array-like
			Predicted probabilities
		"""
		return self.model.predict_proba(X)

class CalibratedBinaryMLP(BaseEstimator):
	"""
	Calibrated classifier using a MLPClassifier classifier as the base
	"""

	def __init__(self, hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, learning_rate='constant', max_iter =1000, method = "isotonic",cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42), n_jobs=10):
		super().__init__()

		self.learning_rate = learning_rate
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation = activation
		self.solver = solver
		self.alpha = alpha
		self.max_iter = max_iter
		self.method = method
		self.cv = cv
		self.model = None
		self.classes_ = [0,1]

		self.n_jobs = n_jobs

	def fit(self,X,y,sample_weight=None,**kwargs):
		"""Fit the calibrated model.

		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels

		sample_weight : 1d array-like
			Sample weights. If None, then samples are equally weighted.
		
		fit_params : dict
			Additional fit parameters

		Returns
		-------
		model : object
			Returns an instance of the calibrated model.
		"""

		self.model = CalibratedClassifierCV(base_estimator =MLPClassifier(learning_rate = self.learning_rate, hidden_layer_sizes = self.hidden_layer_sizes, activation = self.activation, solver = self.solver, alpha = self.alpha, max_iter = self.max_iter, random_state = 42), method = self.method, cv = self.cv, n_jobs = self.n_jobs)
		self.model.fit(X,y,sample_weight,**kwargs)
		self._estimator_type = self.model._estimator_type

		return self.model

	def predict(self, X):
		"""
		Returns the predictons of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		C : 1d array-like
			Predictions
		"""
		return self.model.predict(X)

	def predict_proba(self, X):
		"""
		Returns the predicted probabilities of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 2d array-like
			Predicted probabilities
		"""
		return self.model.predict_proba(X)
	def decision_function(self, X):
		"""
		Returns the probabilities given input samples are positive
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 1d array-like
			Predicted probabilities
		"""
		return self.predict_proba(X)[:,1]

class CalibratedBinaryGBC(BaseEstimator):
	"""
	Calibrated classifier using a GradientBoostingClassifier classifier as the base
	"""

	def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,  min_samples_split=2, min_samples_leaf=1, max_depth=3, max_features='sqrt', method = "isotonic",cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42), n_jobs=10):
		super().__init__()
		
		self.loss = loss
		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.max_depth = max_depth
		self.max_features = max_features
		self.method = method
		self.cv = cv
		self.model = None

		self.n_jobs = n_jobs

	def fit(self,X,y,sample_weight=None,**kwargs):
		"""Fit the calibrated model.

		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels

		sample_weight : 1d array-like
			Sample weights. If None, then samples are equally weighted.
		
		fit_params : dict
			Additional fit parameters

		Returns
		-------
		model : object
			Returns an instance of the calibrated model.
		"""

		self.model = CalibratedClassifierCV(base_estimator =GradientBoostingClassifier(loss = self.loss, learning_rate = self.learning_rate, n_estimators = self.n_estimators, min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf, max_depth = self.max_depth, max_features = self.max_features, random_state = 42), method = self.method, cv = self.cv, n_jobs = self.n_jobs)
		self.model.fit(X,y,sample_weight,**kwargs)
		self._estimator_type = self.model._estimator_type
		self.classes_ = self.model.classes_

		return self.model

	def predict(self, X):
		"""
		Returns the predictons of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		C : 1d array-like
			Predictions
		"""
		return self.model.predict(X)

	def predict_proba(self, X):
		"""
		Returns the predicted probabilities of a given input
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 2d array-like
			Predicted probabilities
		"""
		return self.model.predict_proba(X)