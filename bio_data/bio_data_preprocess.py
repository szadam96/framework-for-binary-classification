import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

from utils.preprocess_utils import get_categorical_imputer, get_normalizer, get_real_imputer

class BioDataPreprocess:
	"""
	A class that defines the preprocessing steps to be done on the input data and creates the trining pipeline
	
	Attributes
	----------
	data : DataFrame
		The input data used for the training

	target_column : str
		The name of the column in the dataset that is to be predicted by the model

	base_model 
		The macine learning model used for the last step of the pipeline

	drop_threshold: float
		The ratio of missing data above which the row or column will be dropped

	normalizer: str
		Name of the normalizer method

	categorical_impute: str
		Name of the impute method for categorical variables

	real_impute: str
		Name of the impute method for real variables

	random_state : int
		Random seed to make the iterative imputer deterministic
	"""
	def __init__(self, data: pd.DataFrame,
				 target_column: str,
				 base_model,
				 smote: bool=False,
				 drop_threshold: int=0.3,
				 normalizer: str=None,
				 categorical_impute: str='most_frequent',
				 real_impute: str='mean', 
				 random_state=42):
		self.data = data
		self.target_column = target_column
		self.base_model = base_model
		self.random_state = random_state
		self.smote = smote
		self.drop_threshold = drop_threshold
		self.normalizer = normalizer
		self.categorical_impute = categorical_impute
		self.real_impute = real_impute
		self.lof = LocalOutlierFactor()
	
	def detect_outliers(self):
		"""
		Detects outliers in the dataset

		Returns
		-------
		outlier_detector: LocalOutlierFactor
		"""
		X = self.data
		real_columns = [col for col in X if len(X[col].dropna().unique()) > 10]
		categorical_columns = [col for col in X if len(X[col].dropna().unique()) <= 10]

		outlier_transformer = ColumnTransformer(
            [
                ('categorical', self.__preprocess_categorical_columns(), lambda df: [c for c in df.columns if c in categorical_columns]),
                ('real', self.__preprocess_real_columns(False), lambda df: [c for c in df.columns if c in real_columns]),
            ]
        )
		outlier_detector = Pipeline(steps=[('imputer', outlier_transformer), ('lof', self.lof)])

		return X, outlier_detector

	def prerocess_and_create_pipeline(self):
		"""
		Separates the data into input features and label, and creates the trainning pipeline
		
		Returns
		-------
		X : 2d array-like
			Array of features

		y : 1d array-like
			Array of labels
			
		pipeline : Pipeline
			the trining pipeline
		"""
		X = self.data
		X = X.loc[X.isna().mean(axis=1) < self.drop_threshold, X.isna().mean(axis=0) < self.drop_threshold]
		y = X[self.target_column]
		X = X.drop([self.target_column], axis=1)

		
		real_columns = [col for col in X if len(X[col].dropna().unique()) > 10]
		categorical_columns = [col for col in X if len(X[col].dropna().unique()) <= 10]
		
		col_transformer = ColumnTransformer(
            [
                ('categorical', self.__preprocess_categorical_columns(), lambda df: [c for c in df.columns if c in categorical_columns]),
                ('real', self.__preprocess_real_columns(), lambda df: [c for c in df.columns if c in real_columns]),
            ]
        )
		preprc_steps = [
			('preprocessor', col_transformer),
			('classifier', self.base_model)
		]
		if self.smote:
			preprc_steps.insert(1, ('smote', SMOTE(random_state=self.random_state)))


		pipeline = Pipeline(steps=preprc_steps)
		return X, y, pipeline
	
	def __preprocess_real_columns(self, normalize=True):
		imputer = get_real_imputer(self.real_impute, self.random_state)
		normalizer = get_normalizer(self.normalizer) if normalize else get_normalizer(None)
		return Pipeline(
			steps=[
				('imputer', imputer),
				('normalizer', normalizer)
			]
		)
	def __preprocess_categorical_columns(self):
		imputer = get_categorical_imputer(self.categorical_impute)
		return Pipeline(steps=[('imputer', imputer)])
		