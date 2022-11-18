import warnings
import numpy as np

from model.feature_selection import FeatureSelectionCV
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib.pyplot as plt

from utils.utils import mean_confidence_interval, scores_with_optimal_cutoff
warnings.filterwarnings("ignore")

class CrossValidatedModel:
	"""
	A class that implements a machine learning model trained with grid search and nested cross-validation
	
	Attributes
	----------
	base_model :
		the machine learning model that is trained
		
	cv : StratifiedKFold
		cross-validation instance that is used for both of the inner and outer vaildation of the nested cross-validation
		
	fit_models : 
		the ml models that have been fit with grid search on the different folds the cv

	do_feature_selection: bool
		controls wehther or not the training will include feature selection

	feature_selection_params: dict
		keyword arguments for the feature selection
		
	feature_selectors : list(FeatureSelectionCV)
		the FeatrueSelectionCV instance corresponding to each of the models in fit_models
		
	best_features : set(str)
		the union of the best features selected by the feature selectors
		
	best_hyperparams : list(dict)
		the best hyperparameters choosen by the grid search algorithms corresponding to the models in fit_params
		
	param_grid : dict
		dictionary of possible hyperparameters passed to the grid search algorithm
		
	random_state : int
		random seed to make the cross-validation deterministic
		
	n_jobs : int
		Number of jobs to run in parallel.		
	"""
	def __init__(self, base_model, param_grid, random_state, weighted=False, do_feature_selection=False, feature_selection_params=None, n_jobs=10, cv=5):
		self.base_model = base_model
		self.cv = StratifiedKFold(n_splits=cv, shuffle=True,random_state=random_state)
		self.fit_models = []
		self.feature_selectors = []
		self.best_features = set()
		self.best_hyperparams = []
		self.param_grid = param_grid
		self.random_state = random_state
		self.n_jobs = n_jobs
		self.weighted = weighted
		self.do_feature_selection = do_feature_selection
		self.feature_selection_params = feature_selection_params

	def fit_gs(self, X, y):
		"""
		fit a model on each of the cv folds using grid search and feature selection

		Parameters:
		-----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels
		"""
		if self.do_feature_selection:
			self.__feature_selection_gs(X, y)
		else:
			self.__do_grid_serach(X, y)
	
	def cross_validate(self, X, y, confidence=0.95):
		"""
		Validate each fit models on the corresponding test set
		
		Parameters
		----------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels
		
		confidence: float
			Confidence value used for the confidence interval calculation
			
		Returns
		-------
		fig_roc : Figure
			Figure of the ROC curve for each fold and their means
			
		fig_pr : Figure
			Figure of the PR curve for each fold and their means
			
		out : dict
			Dictionary of the evaluation metrics
		"""
		if len(self.fit_models) == 0:
			self.fit_gs(X, y)
		tprs = []
		mean_fpr = np.linspace(0, 1, 100)
		
		precs = []
		mean_recall = np.linspace(1, 0, 100)
		
		aucs = []
		aps = []
		
		log_losses = []
		briers = []
		accs = []
		bas = []
		f1s = []
		
		fig_roc, ax_roc = plt.subplots()
		fig_pr, ax_pr = plt.subplots()
		for i, (train, test) in enumerate(self.cv.split(X, y)):
			if len(self.feature_selectors) > 0:
				X_test = self.feature_selectors[i].transform(X.iloc[test])
			else:
				X_test = X.iloc[test]
				
			clf = self.fit_models[i]
			
			y_proba = clf.predict_proba(X_test)[:,1]
			ax_roc, viz_roc = self.__plot_roc_fold(y.iloc[test], y_proba, ax_roc, i)
				
			ax_pr, viz_pr = self.__plot_pr_fold(y.iloc[test], y_proba, ax_pr, i)

			fpr, tpr, thresholds = roc_curve(y.iloc[test], y_proba)
			interp_tpr = np.interp(mean_fpr, fpr, tpr)
			interp_tpr[0] = 0.0
			tprs.append(interp_tpr)
			
			interp_prec = np.interp(mean_recall, np.flip(viz_pr.recall), np.flip(viz_pr.precision))
			precs.append(interp_prec)
			
			aucs.append(viz_roc.roc_auc)
			aps.append(viz_pr.average_precision)

			log_loss_score, brier_score, acc, f1, ba = scores_with_optimal_cutoff(tpr, fpr, thresholds, y.iloc[test], y_proba)
			log_losses.append(log_loss_score)
			briers.append(brier_score)
			accs.append(acc)
			f1s.append(f1)
			bas.append(ba)

		ax_roc = self.__plot_mean_curve(ax_roc, tprs, mean_fpr, aucs, curve='roc')
			
		ax_pr = self.__plot_mean_curve(ax_pr, precs, mean_recall, aps, curve='pr')

		logloss_mean, logloss_lower, logloss_upper = mean_confidence_interval(log_losses, confidence=confidence)
		brier_mean, brier_lower, brier_upper = mean_confidence_interval(briers, confidence)
		acc_mean, acc_lower, acc_upper = mean_confidence_interval(accs, confidence)
		f1_mean, f1_lower, f1_upper = mean_confidence_interval(f1s, confidence)
		auc_mean, auc_lower, auc_upper = mean_confidence_interval(aucs, confidence)
		ba_mean, ba_lower, ba_upper = mean_confidence_interval(bas, confidence)
		ap_mean, ap_lower, ap_upper = mean_confidence_interval(aps, confidence)
		
		out = {}
		out['auc_mean'] = auc_mean
		out['auc_lower'] = auc_lower
		out['auc_upper'] = auc_upper
		out['logloss_mean'] = logloss_mean
		out['logloss_lower'] = logloss_lower
		out['logloss_upper'] = logloss_upper
		out['brier_loss_mean'] = brier_mean
		out['brier_loss_lower'] = brier_lower
		out['brier_loss_upper'] = brier_upper
		out['accuracy_mean'] = acc_mean
		out['accuracy_lower'] = acc_lower
		out['accuracy_upper'] = acc_upper
		out['balanced_accuracy_mean'] = ba_mean
		out['balanced_accuracy_lower'] = ba_lower
		out['balanced_accuracy_upper'] = ba_upper
		out['f1_score_mean'] = f1_mean
		out['f1_score_lower'] = f1_lower
		out['f1_score_upper'] = f1_upper
		out['average_precision_mean'] = ap_mean
		out['average_precision_lower'] = ap_lower
		out['average_precision_upper'] = ap_upper

		return fig_roc, fig_pr, out

	def predict_proba(self, X):
		"""
		Calculates the predicted probabilities using all the fit models
		
		Parameters
		----------
		X : 2d array-like
			Array of features
			
		Returns
		-------
		probas : 1d array-like
			Predicted probabilities
		"""
		assert len(self.fit_models) != 0, 'Model have not yet been trained!'
		probas = []
		for i, clf in enumerate(self.fit_models):
			if len(self.feature_selectors) > 0:
				X_transformed = self.feature_selectors[i].transform(X)
			else:
				X_transformed = X
			probas.append(clf.predict_proba(X_transformed))
		return np.mean(probas, axis=0)

	def __plot_mean_curve(self, ax, ys, mean_x, scores, confidence=0.95, curve='roc'):
		mean_score = np.mean(scores)
		std_score = np.std(scores)
		mean_y = np.mean(ys, axis=0)
		if curve == 'roc':
			ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
			label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_score, std_score)
			mean_y[-1] = 1.0
		elif curve == 'prc' or curve == 'pr':
			label=r"Mean PRC (AP = %0.2f $\pm$ %0.2f)" % (mean_score, std_score)
		else:
			raise ValueError(f'{curve} curve is not supported!')

		ax.plot(
			mean_x,
			mean_y,
			color="b",
			label=label,
			lw=2,
			alpha=0.8
		)

		_, y_lower, y_upper = mean_confidence_interval(ys, confidence=confidence)
		ax.fill_between(
			mean_x,
			y_lower,
			y_upper,
			color="grey",
			alpha=0.2,
			label=f'Confidence interval: {confidence}',
		)
		ax.set(
			xlim=[-0.05, 1.05],
			ylim=[-0.05, 1.05],
		)
		ax.legend(loc="lower right")
		return ax
		
	def __plot_roc_fold(self, y_true, y_pred, ax, fold):
		viz = RocCurveDisplay.from_predictions(
				y_true,
				y_pred,
				name="ROC fold {}".format(fold),
				alpha=0.3,
				lw=1,
				ax=ax,
			)
		return ax, viz
		
	def __plot_pr_fold(self, y_true, y_pred, ax, fold):
		viz = PrecisionRecallDisplay.from_predictions(
				y_true,
				y_pred,
				name="PRC fold {}".format(fold),
				alpha=0.3,
				lw=1,
				ax=ax,
			)
		return ax, viz
	
	def __feature_selection_gs(self, X, y):
		for i, (train, test) in enumerate(self.cv.split(X, y)):
			fs = FeatureSelectionCV(self.base_model,
									self.param_grid,
									scoring='roc_auc',
									cv=self.cv,
									n_jobs=self.n_jobs,
									**self.feature_selection_params)
			w_train = None
			if self.weighted:
				w_train = compute_sample_weight('balanced', y.iloc[train])

			fs.fit(X.iloc[train], y.iloc[train], classifier__sample_weight=w_train)
			print(f'fold_{i} {len(fs.best_features_)} features: roc_auc: {fs.best_score_: .4f}')
			self.feature_selectors.append(fs)
			self.fit_models.append(fs.best_estimator_)
			self.best_hyperparams.append(fs.best_params_)
			self.best_features.update(fs.best_features_)
	
	def __do_grid_serach(self, X, y):
		for i, (train, test) in enumerate(self.cv.split(X, y)):
			gs = GridSearchCV(self.base_model,
							  self.param_grid,
							  scoring='roc_auc',
							  cv=self.cv, error_score='raise',
							  n_jobs=self.n_jobs)

			w_train = None
			if self.weighted:
				w_train = compute_sample_weight('balanced', y.iloc[train])

			gs.fit(X.iloc[train], y.iloc[train], classifier__sample_weight=w_train)
			self.fit_models.append(gs.best_estimator_)
			print(f'fold_{i}: gs_roc: {gs.best_score_}')