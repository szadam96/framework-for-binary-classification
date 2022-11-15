import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.impute import SimpleImputer, IterativeImputer

def get_normalizer(normalizer_name: str):
	"""
	Initiates a normalizer instance

	Parameters
	----------
	normalizer_name: str
		Name of the normalizer

	Returns
	-------
	normalizer: object
		Normalizer object
	"""
	if normalizer_name == 'standard':
		return StandardScaler()
	if normalizer_name == 'l1':
		return Normalizer(norm='l1')
	if normalizer_name == 'l2':
		return Normalizer(norm='l2')
	if normalizer_name == 'minmax':
		return MinMaxScaler()
	if normalizer_name == 'robust':
		return RobustScaler()
	if normalizer_name is None:
		return FunctionTransformer()
	raise ValueError(f'Normalizer {normalizer_name} is not supported')

def get_real_imputer(imputer_name, random_state=None):
	"""
	Initiates an imputer for real variables

	Parameter
	---------
	imputer_name: str
		Name of the mipute method
	
	random_state: int
		Random seed to make the iterative imputer deterministic

	Returns
	-------
	imputer: object
		Imputer object
	"""
	if imputer_name == 'iterative':
		return IterativeImputer(max_iter=1000, random_state=random_state)
	if imputer_name == 'external':
		return SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
	raise ValueError(f'Imputer {imputer_name} is not suppoerted')

def get_categorical_imputer(imputer_name):
	"""
	Initiates an imputer for categorical variables

	Parameter
	---------
	imputer_name: str
		Name of the mipute method
	
	random_state: int
		Random seed to make the iterative imputer deterministic

	Returns
	-------
	imputer: object
		Imputer object
	"""
	if imputer_name == 'mean':
		return SimpleImputer(missing_values=np.nan, strategy='mean')
	if imputer_name == 'external':
		return SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
	raise ValueError(f'Imputer {imputer_name} is not suppoerted')