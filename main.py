import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import dill
from sys import argv
from sklearn.manifold import MDS
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, roc_curve

from argparse import ArgumentParser
import warnings
from pathlib import Path

from bio_data.bio_data_preprocess import BioDataPreprocess
from model.cross_validated_model import CrossValidatedModel
import shap

from utils.utils import scores_with_optimal_cutoff, load_data, load_model, get_config, simplify_cross_val_result
warnings.filterwarnings("ignore")

def train_and_evaluate_model(data_path: str, config_path: str = None, target_folder: str = '.', calculate_feature_importances: bool = False):
	"""
	Trains and evaluates a new model using a given dataset
	
	Parameters
	----------
	data_path : str
		Path to the csv file containing the data
		
	config_path : str
		Path to the config file
		
	param_grid_path : str
		Path to the yaml file that defines the possible hyperparameter for the model. If None a default one is defined
		
	target_folder : str
		Path to the folder wher the results will be saved

	calculate_feature_importances: bool
		If true the SHAPley feature importances will be calculated
	"""	
	target_folder = Path(target_folder)
	config_path = Path(config_path)
	os.makedirs(target_folder, exist_ok=True)
	shutil.copy(config_path, target_folder / config_path.name)
	
	data = pd.read_csv(data_path)
	config = get_config(config_path)
	param_grid = config['param_grid']
	#param_grid = get_param_grid_from_config(param_grid_path=param_grid_path)
	X, y, pipeline = BioDataPreprocess(data,
										base_model=config['base_model'],
										random_state=config['random_state'],
										**config['preprocess']).prerocess_and_create_pipeline()
	model = CrossValidatedModel(pipeline, param_grid,
								random_state=config['random_state'],
								do_feature_selection=config['do_feature_selection'],
								feature_selection_params=config.get('feature_selection_params', None))
	model.fit_gs(X, y)
	dill.dump(model, open(target_folder / 'model.pickle', 'wb'))
	
	fig_roc, fig_pr, out = model.cross_validate(X, y)

	result = pd.DataFrame(out, index=[config.get('model_name', 0)])
	result = simplify_cross_val_result(result)
	result.to_csv(target_folder / 'cross_val_result.csv', index=True)
	fig_pr.savefig(target_folder / 'test_result_pr_curve.png',bbox_inches='tight')
	plt.clf()
	fig_roc.savefig(target_folder / 'test_result_roc_curve.png',bbox_inches='tight')
	plt.clf()

	if calculate_feature_importances:
		feature_importanes(model, X, target_folder)
	
	return result

def evaluate_model(model_path: str, data_path: str, target_column: str, target_folder: str = '.', calculate_feature_importances: bool = False):
	"""
	Evaluates a trained model on a given dataset
	
	Parameters
	----------
	model_path : str
		Path to the model
	
	data_path : str
		Path to the csv file containing the data
		
	target_column : str
		The name of the column in the dataset that is to be predicted by the model
		
	target_folder : str
		Path to the folder wher the results will be saved
	
	calculate_feature_importances: bool
		If true the SHAPley feature importances will be calculated
	"""
	target_folder = Path(target_folder)
	model = load_model(Path(model_path))
	X, y = load_data(Path(data_path), target_column)
	fig_roc, ax_roc = plt.subplots()
	fig_pr, ax_pr = plt.subplots()
	y_proba = model.predict_proba(X)[:,1]
	viz_roc = RocCurveDisplay.from_predictions(
				y,
				y_proba,
				alpha=1,
				lw=2,
				ax=ax_roc,
			)

	viz_pr = PrecisionRecallDisplay.from_predictions(
				y,
				y_proba,
				alpha=1,
				lw=2,
				ax=ax_pr,
			)
	ax_roc.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
	fpr, tpr, thresholds = roc_curve(y, y_proba)
	auc = viz_roc.roc_auc
	ap = viz_pr.average_precision
	log_loss_score, brier_score, acc, f1, ba = scores_with_optimal_cutoff(tpr, fpr, thresholds, y, y_proba)
	out = {}
	out['auc'] = auc
	out['average_precision'] = ap
	out['logloss'] = log_loss_score
	out['brier_loss'] = brier_score
	out['accuracy'] = acc
	out['balanced_accuracy'] = ba
	out['f1_score'] = f1

	os.makedirs(target_folder, exist_ok=True)

	result = pd.DataFrame(out, index=[0]).T
	result.to_csv(target_folder / 'test_result.csv', index=True)
	fig_pr.savefig(target_folder / 'test_result_pr_curve.png',bbox_inches='tight')
	fig_roc.savefig(target_folder / 'test_result_roc_curve.png',bbox_inches='tight')
	plt.clf()

	if calculate_feature_importances:
		feature_importanes(model, X, target_folder)

	return result
	
def predict_proba(model_path: str, data_path: str, target_folder: str= '.', save_to_file: bool=True):
	"""
	Loads a model and calculates the predicted probabilities of a given input.
	
	Parameters
	----------
	model_path : str
		Path to the model
	
	data_path : str
		Path to the csv file containing the data
		
	target_folder : str
		Path to the folder wher the results will be saved if save_to_file is True
		
	save_to_file : bool
		If True then the porbabilities will be saved to a csv file
		
	Returns
	-------
	y_proba : 1d array-like
		Predicted probabilities
	"""
	model = load_model(Path(model_path))
	X = pd.read_csv(data_path)
	y_proba = model.predict_proba(X)[:,1]
	if save_to_file:
		target_folder = Path(target_folder)
		os.makedirs(target_folder, exist_ok=True)
		pd.DataFrame(y_proba).to_csv(target_folder / 'predicted_probailities.csv', index=False, header=False)
		
	return y_proba

def feature_importanes(model, X, target_folder):
	'''
	Calculates feature importances using SHAP

	Parameters
	----------
	model: CrossValidatedModel
		the trained machine learning model
	
	X : 2d array-like
		Array of input features
	
	target_folder: str
		Path to the output folder the results will be saved
	'''
	def predict_proba_1(X_in):
		X_in = pd.DataFrame(X_in, columns=X.columns)
		return model.predict_proba(X_in)[:,1]
		
	explainer  = shap.explainers.Sampling(predict_proba_1, X)
	shap_values = explainer(X)
	
	mean_shap = np.mean(shap_values.abs.values, axis=0)
	
	features = shap_values.feature_names
	importances = pd.DataFrame({'col_name': features, 'mean_abs_shap_values': mean_shap})
	
	importances.to_csv(target_folder / 'feature_importances.csv')

	shap.summary_plot(shap_values, X, show=False)
	plt.savefig(target_folder / 'feature_importances_plot.png',bbox_inches='tight')
	plt.clf()

def detect_outliers(data_path: str, target_folder: str= '.'):
	"""
	Detects possible outliers in the dataset

	Parameters
	----------	
	data_path : str
		Path to the csv file containing the data
		
	target_folder : str
		Path to the folder where the results will be saved
	"""
	target_folder = Path(target_folder)
	os.makedirs(target_folder, exist_ok=True)
	
	data = pd.read_csv(data_path)
	X, detector = BioDataPreprocess(data, base_model=None,
									target_column=None).detect_outliers()
	pred = detector.fit_predict(X)
	pred = np.argwhere(pred==-1).flatten()

	detector.steps[-1] = ('pca', MDS(n_components=2))
	X_2d = detector.fit_transform(X)
	plt.scatter(X_2d[:,0], X_2d[:,1], label='Inliers')
	plt.scatter(X_2d[pred][:,0], X_2d[pred][:,1], c='red', label='Outliers')
	plt.xlabel('Dimension 1')
	plt.ylabel('Dimension 2')
	plt.legend()
	plt.savefig(target_folder / 'outliers_visualized.png',bbox_inches='tight')
	plt.clf()

	outliers = data.iloc[pred]
	outliers.to_csv(target_folder / 'possible_outliers.csv')

	return outliers