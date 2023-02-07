from main import train_and_evaluate_model, evaluate_model, predict_proba, detect_outliers
from pathlib import Path
from argparse import ArgumentParser
from sys import argv


def main():
	parser = ArgumentParser()
	action_choices = ['evaluate', 'train', 'predict_proba', 'detect_outliers']
	parser.add_argument('action', help='evaluate a trained model or train a new one using your own dataset', choices=action_choices)
	parser.add_argument('--data', help='path to the csv file of the dataset', required=True, type=str)
	parser.add_argument('--target_column', help='name of the column that contains the mortalities in the dataset', required=('evaluate' in argv), type=str)
	parser.add_argument('--target_folder', help='folder where the results will be saved', required=False, default='.', type=str)
	parser.add_argument('--model_path', help='path to the trained model', required=('evaluate' in argv))
	parser.add_argument('--config_path', help='path to the yaml file that contains the configurations for the training', default=None, required=('train' in argv))
	parser.add_argument('--calculate_feature_importances', help='if this flag is set sHAP feature importances will be calculated for th model', action='store_true')
	args = parser.parse_args()

	if args.action == 'train':
		train_and_evaluate_model(args.data, args.config_path, Path(args.target_folder), args.calculate_feature_importances)
	elif args.action == 'evaluate':
		evaluate_model(args.model_path, args.data, args.target_column, Path(args.target_folder), args.calculate_feature_importances)
	elif args.action == 'predict_proba':
		predict_proba(args.model_path, args.data, Path(args.target_folder), save_to_file=True)
	elif args.action == 'detect_outliers':
		detect_outliers(args.data, Path(args.target_folder))
		
if __name__ == '__main__':
	main()