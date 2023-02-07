from pathlib import Path
import dill
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss, brier_score_loss, f1_score

import numpy as np
import scipy
import yaml

from model.other_models import CalibratedBinaryGBC, CalibratedBinaryKNN, CalibratedBinaryLog, CalibratedBinaryMLP, CalibratedBinarySVC, CalibratedBinaryXGBoost, CalibratedRF

def load_data(data_path: str, target_column: str):
	"""
	Loads the data into pandas DataFrame, and separates them into input features and target labels
	
	Parameters
	----------
	data_path : str
		Path to the csv file containing the data
		
	target_column : str
		The name of the column in the dataset that is to be predicted by the model
		
	Returns
	-------
	X : 2d array-like
			Array of features

	y : 1d array-like
		Array of labels
	"""
	data = pd.read_csv(data_path)
	y = data[target_column]
	X = data.drop(columns=[target_column])
	return X, y

def load_model(model_path: str):
	"""
	Loads a fitted machine learning model
	
	Parameters
	----------
	model_path : str
		Path to the model
	
	Returns
	-------
	model
		The loaded model instance
	"""
	return dill.load(open(model_path, 'rb'))


def mean_confidence_interval(data, confidence=0.95):
	"""
	Calculates the mean and confidence interval of a data
	
	Parameters
	----------
	data : array-like
		The data that is used for the calculation
	
	confidence : float
		The confidence value that is used for the confidenc interval calculation
		
	Returns
	-------
	mean : float
		The mean of the data
	
	lower : float
		The lower value of the confidence interval
		
	upper : float
		The upper value of the confidence interval
	"""
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h
	
def scores_with_optimal_cutoff(tpr, fpr, thresholds, y_true, y_proba):
	"""
	Calculates multiple metrics based on the optimal cutoff of the predicted probabilities
	
	Parameters
	----------
	tpr : 1d array-like
		True positive rates
	
	fpr : 1d array-like
		False positive rates
		
	thresholds : 1d array-like
		Threshold values
	
	y_true : 1d array-like
		True labels
	
	y_proba : 1d array-like
		Predicted probabilities
		
	Returns
	-------
	log_loss_score : float
		Logloss score
		
	brier_score : float
		Brier score
		
	acc : float
		Accuracy score
		
	f1 : float
		F1 score
	
	ba : float
		Balanced accuracy score
	"""
	optimal_idx = np.argmax(tpr - fpr)
	optimal_threshold = thresholds[optimal_idx]
	th_mask = y_proba >= optimal_threshold
	y_pred = np.zeros(y_true.shape, np.int64)
	y_pred[th_mask] = 1
	acc = accuracy_score(y_true,y_pred)
	f1 = f1_score(y_true, y_pred)
	ba = balanced_accuracy_score(y_true, y_pred)
	if np.max(y_proba > 1):
		y_proba = y_pred
	log_loss_score = log_loss(y_true,y_proba)
	brier_score = brier_score_loss(y_true,y_proba)

	return log_loss_score, brier_score, acc, f1, ba

def get_param_grid_from_config(param_grid: dict, model_name: str):
	"""
	Creates a hyperparameter grid that can be passed to the grid search algorithm for hyperparameter optimization
	
	Parameters
	----------
	param_grid_path : str
		Path to the yaml file that defines the possible hyperparameter for the model. If None a default one is defined
		
	Returns
	-------
	param_grid: dict
		Dictionary that can be passed to GridSearchCV
	"""
	if param_grid is not None:
		param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
	else:
		with open('utils/default_hyperparams.yaml' , 'r') as f:
			default_params = yaml.safe_load(f)
			param_grid = {f'classifier__{k}': v for k, v in default_params[model_name].items()}
		
	return param_grid

def get_model(model_name, cv, n_jobs):
	"""
	Initiates a calibrate binary classifier

	Parameters
	----------
	model_name: str
		Name of the classifier
	
	cv: StratifiedKFold
		Cross-validation instance

	n_jobs: int
		Number of job running paralelly

	Returns
	-------
	model: object
		Calibrated binary classifier
	"""
	if model_name == "randomforest":
		return CalibratedRF(cv=cv, n_jobs=n_jobs)
	if model_name == "svc":
		return CalibratedBinarySVC(cv=cv, n_jobs=n_jobs)
	if model_name == "log_l1":
		return CalibratedBinaryLog(penalty='l1', cv=cv, n_jobs=n_jobs)
	if model_name == "log_l2":
		return CalibratedBinaryLog(penalty='l2', cv=cv, n_jobs=n_jobs)
	if model_name == "knn":
		return CalibratedBinaryKNN(cv=cv, n_jobs=n_jobs)
	if model_name == "mlp":
		return CalibratedBinaryMLP(cv=cv, n_jobs=n_jobs)
	if model_name == "gbc":
		return CalibratedBinaryGBC(cv=cv, n_jobs=n_jobs)
	if model_name == 'xgboost':
		return CalibratedBinaryXGBoost(cv = cv, n_jobs=n_jobs)
	raise ValueError(f'model {model_name} is not supported!')

def get_config(config_path: str):
	"""
	Creates the configurations for the training

	Parameters
	----------
	config_path: str
		Path to the yaml file containing the configurations

	Returns
	-------
	results: dict
		The reaulting configurations
	"""
	config_path = Path(config_path)
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	result = dict()

	result['cv'] = config.get('cv', 5)
	result['random_state'] = config.get('random_state', 42)
	result['n_jobs'] = config.get('n_jobs', 10)
	result['do_feature_selection'] = config.get('do_feature_selection', False)
	
	param_grid = config.get('hyperparameters', None)
	model_name = config.get('model', None)
	model = get_model(model_name, result['cv'], result['n_jobs'])
	result['model_name'] = model_name
	result['base_model'] = model
	result['param_grid'] = get_param_grid_from_config(param_grid, model_name)

	preproc = config['preprocess']
	result['balancing_method'] = preproc.get('balancing_method', None)
	result['weighted'] = False
	if result['balancing_method'] == 'weighted':
		result['weighted'] = True

	result['preprocess'] = dict()
	result['preprocess']['smote'] = False
	if result['balancing_method'] == 'smote':
		result['preprocess']['smote'] = True
	result['preprocess']['target_column'] = preproc['target_column']
	result['preprocess']['normalizer'] = preproc.get('normalizer', None)
	result['preprocess']['drop_threshold'] = preproc.get('drop_threshold', 0.3)
	result['preprocess']['categorical_impute'] = preproc.get('categorical_impute', 'external')
	result['preprocess']['real_impute'] = preproc.get('real_impute', 'external')

	if result['do_feature_selection']:
		result['feature_selection_params'] = config['feature_selection_params']

	return result

def simplify_cross_val_result(df: pd.DataFrame):
	'''
	This function simplifies the DataFrame that is created by the CrossValidatedModel's cross_validate function
	to make it more humanly readable.

	Parameters
	----------
	df: DataFrame
		the resulting data frame after the cross-validated training

	Returns
	-------
	res_df: DataFrame
		the simplified data frame
	'''
	df = df.T
	metrics = ['auc', 'logloss', 'brier_loss', 'accuracy', 'balanced_accuracy', 'f1_score', 'average_precision']
	res_df = pd.DataFrame(columns=['mean', '+/-'], index=metrics)
	model = df.columns[0]
	for m in metrics:
		mean = df.loc[f'{m}_mean', model]
		pm = df.loc[f'{m}_upper', model] - mean
		res_df.loc[m, 'mean'] = mean
		res_df.loc[m, '+/-'] = pm

	return res_df