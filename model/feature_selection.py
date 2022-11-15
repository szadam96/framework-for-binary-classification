import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from pathlib import Path
from typing import Union, List

from mrmr import mrmr_classif
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

class FeatureSelectionCV:
	"""
	A class that fits a model with hyperparameter optimization and feature selection using cross-validation. Serves as the inner cv in the nested cross-validation.
	
	Parameters
	----------
	estimator :
		The machine learning estimator the is to be fit
		
	hyper_params : dict
		Dictionary of possible hyperparameters passed to the grid search algorithm
		
	cv : 
		Cross-validation instance that is used for the grid search
		
	scoring : str
		The metric that is used for the grid search's evaluation
		
	step : int
		How many steps the number of selected features is reduced after each training
		
	min_features : int
		The number of minimum features that will be selected
	
	max_features : int
		The number of maximum selected features
		
	n_jobs : int
		Number of jobs to run in parallel.
	"""
	def __init__(self, estimator, hyper_params: dict, cv=None, scoring='roc_auc', step=1, min_features=10, max_features=20, score_threshold=0.95, n_jobs=10):
		self.estimator = estimator
		self.hyper_params = hyper_params
		self.cv = cv
		self.scoring = scoring
		self.step = step
		self.score_threshold = score_threshold
		self.min_features = min_features
		self.max_features = max_features
		self.n_jobs=n_jobs
	
	def calculate_feature_importances(self, X, y, K=None):
		"""
		Calculates the feature importances using the MRMr algorithm
		
		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels
			
		K : int
			Number of features to select
		
		Returns
		-------
		list of K features
		"""
		if K is None:
			K = X.shape[0]
		return mrmr_classif(X, y, K=K)
		
	def fit(self, X, y, **fit_params):
		"""
		Fit the model using grid search and feature selection
		
		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels
		
		fit_params : dict
			Additional fit parameters
		"""
		current_features = self.calculate_feature_importances(X, y, None)
		gs = GridSearchCV(self.estimator, self.hyper_params, scoring=self.scoring, cv = self.cv, error_score='raise', n_jobs=self.n_jobs)
		gs.fit(X, y, **fit_params)
		initial_score = gs.best_score_
		print(f'{X.shape[1]} features: {self.scoring}: {initial_score: .4f} +/- {gs.cv_results_["std_test_score"][gs.best_index_]: .4f}')
		if self.max_features > X.shape[1]:
			self.max_features = X.shape[1]
		
		self.best_params_ = gs.best_params_
		self.best_score_ = 0
		self.best_estimator_ = gs.best_estimator_
		self.best_features_ = X.columns
		
		self.scores = dict()
		self.scores[X.shape[1]] = initial_score
		
		self.score_stds = [gs.cv_results_['std_test_score'][gs.best_index_]]
		
		current_score = initial_score
		#current_features = None
		X_current = X.copy()
		for i in range(self.max_features, self.min_features-1, -self.step):
			current_features = current_features[:i]
			
			X_current = self.__transform(X_current, current_features)
			gs.fit(X_current, y, **fit_params)
			current_score = gs.best_score_
			current_std = gs.cv_results_['std_test_score'][gs.best_index_]
			print(f'{X_current.shape[1]} features: {self.scoring}: {current_score: .4f} +/- {current_std: .4f}')
			
			if current_score > self.best_score_ and current_score > initial_score * self.score_threshold:			
				self.best_params_ = gs.best_params_
				self.best_score_ = gs.best_score_
				self.best_estimator_ = gs.best_estimator_
				self.best_features_ = current_features
				
			self.scores[i] = current_score
			self.score_stds.append(current_std)
		
		if self.best_score_ == 0:
			self.best_score_ = initial_score
			
		
	def transform(self, X):
		"""
		Transforms the input array to only include the best selected features
		
		Parameters
		----------
		X : 2d array-like
			Array of features
		
		Returns
		-------
		Transormed array
		"""
		return self.__transform(X, self.best_features_)
		
	def fit_transform(self, X, y, **fit_params):
		"""
		Fits the model and transforms the input array
		
		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels
		
		fit_params : dict
			Additional fit parameters
		"""
		self.fit(X, y, **fit_params)
		return self.transform(X)	
	
	def __transform(self, X, features):
		return X[features]
