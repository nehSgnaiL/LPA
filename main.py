from pipeline import run_model
from args import default_args


if __name__ == '__main__':
    args = {
        'downstream': True,
        'pre_model_name': 'LPA',
        'activity_type': 'A6',  # "None" "A3" "A6" "L6"
        'act_emb_size': 5,
        'embed_name': 'downstream',
        'dataset': 'data_sample',
        'max_epoch': 30,
        'config_file': None,
        'saved_model': True,
        'train': True,
    }

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
