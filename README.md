# ML framework for binary classification
This framework makes it possible to train and evalute multiple machine learning model for binary classification for biomedical data. The supported models are:
- Random Forest
- Support Vector Machine
- Logistic Regression
- K-Nearest Neighbor Classifier
- Multi-Layer Perceptron
- Gradient Boosting Classifier
- XGBoost
## Installation
The installation is done using conda with the following command:
```
conda env create -f environment.yml
```
## Usage
### Training
You can train and evaluating a new model from scratch on your training data using the following command:
```
python --data PATH_TO_DATA_CSV --target_folder TARGET_FOLDER --config_path PATH_TO_CONFIG_YAML [--calculate_feature_importances] train
```
Where ```PATH_TO_DATA_CSV``` is the path to the training data in a csv format, ```TARGET_FOLDER``` is the name of the folder the results will be saved, and ```PATH_TO_CONFIG_YAML``` is the yaml file containing the cofigurations of the training. An example config file has been provided. The feature importances of the model can be calculated using the SHAP library by using the ```--calculate_feature_importances``` flag.
### Evaluation
The evaluation of a trained model on an external dataset can ben done using the following command:
```
python --data PATH_TO_DATA_CSV --model_path MODEL_PATH --target_column TARGET_COLUMN --target_folder TARGET_FOLDER [--calculate_feature_importances] evaluate
```
Where ```PATH_TO_DATA_CSV``` is the path to the external dataset in a csv format, ```MODEL_PATH``` is the path to the trained and saved model that is to be evaluated, ```TARGET_COLUMN``` is the name of the column in the dataset that is to be predicted, and ```TARGET_FOLDER``` is the name of the folder the results will be saved. The feature importances of the model can be calculated using the SHAP library by using the ```--calculate_feature_importances``` flag.
### Prediction
A trained model can be used for prediction using the following command:
```
python --data PATH_TO_DATA_CSV --model_path MODEL_PATH --target_folder TARGET_FOLDER predict_proba
```
Where ```PATH_TO_DATA_CSV``` is the path to the external dataset in a csv format, ```MODEL_PATH``` is the path to the trained and saved model that is to be used for prediction, and ```TARGET_FOLDER``` is the name of the folder the predicted probabilities will be saved.

### Outlier detection
To run the outlier detection use the following command:
```
python --data PATH_TO_DATA_CSV --target_folder TARGET_FOLDER detect_outliers
```
Where ```PATH_TO_DATA_CSV``` is the path to the external dataset in a csv format, and ```TARGET_FOLDER``` is the name of the folder the possible outliers will be saved.