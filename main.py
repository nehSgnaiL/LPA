from pipeline import run_model
from args import default_args
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model with specified parameters.')
    parser.add_argument('--downstream', type=bool, default=True, help='Whether to run the downstream task.')
    parser.add_argument('--pre_model_name', type=str, default='LPA', help='Name of the prediction model.')
    parser.add_argument('--activity_type', type=str, default='A6', help='Activity integration strategies.')
    parser.add_argument('--act_emb_size', type=int, default=5, help='Size of the activity embedding.')
    parser.add_argument('--embed_name', type=str, default='downstream', help='Type of the embedding.')
    parser.add_argument('--dataset', type=str, default='data_sample', help='Name of the dataset to use.')
    parser.add_argument('--max_epoch', type=int, default=30, help='Maximum number of epochs for training.')
    parser.add_argument('--config_file', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--saved_model', type=bool, default=True, help='Whether to save the model after training.')
    parser.add_argument('--train', type=bool, default=True, help='Whether to train the model.')

    args = vars(parser.parse_args())

    for key in args.keys():
        if key not in default_args.keys():
            print(f'error: unexpected key {key} in args.')
    for key in default_args.keys():
        if key not in args.keys():
            args[key] = default_args[key]

    other_args = {key: val for key, val in args.items() if key not in [
        'task', 'downstream', 'dataset', 'config_file', 'saved_model', 'train'] and
                  val is not None}
    run_model(task=args.get('task'), model_name=args.get('downstream'), dataset_name=args.get('dataset'),
              config_file=args.get('config_file'), saved_model=args.get('saved_model'),
              train=args.get('train'), other_args=other_args)
