import os
import random
import numpy as np

from utils import ConfigParser, set_random_seed, get_logger
from dataset import TrajectoryDataset
from embed.static import StaticEmbed, DownstreamEmbed
from downstream.LPA import LPA

from executor import TrajectoryExecutor


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): downstream name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the downstream
        train(bool): whether to train the downstream
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name,
                          config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)
    activity_type = config['activity_type']
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 100000)
        config['exp_id'] = exp_id
    logger = get_logger(config)
    embed_name = config.get('embed_name', 'downstream')
    logger.info('Begin pipeline, model_name={}, dataset_name={}, exp_id={}, embed_name={}, activity_type={}'.
                format(str(model_name), str(dataset_name), str(exp_id), embed_name, activity_type))
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    dataset = TrajectoryDataset(config)
    train_model(dataset, config)


def train_model(dataset, param):
    device = param.get('device', 'cuda:0')

    task_name = param.get('task', 'traj_loc_pred')
    embed_name = param.get('embed_name', 'downstream')
    embed_size = int(param.get('act_emb_size', 2))

    max_seq_len = dataset.max_seq_len  # max input sequence length

    activity_type = param.get('activity_type', 'None')

    if activity_type in ['L6']:
        vocab_size = dataset.encoder.num_loc  # statics
    else:
        vocab_size = dataset.encoder.num_act  # statics

    embed_mat = np.random.uniform(low=-0.5 / embed_size, high=0.5 / embed_size, size=(vocab_size, embed_size))
    embed_layer = StaticEmbed(embed_mat)
    if embed_name == 'downstream':
        embed_layer = DownstreamEmbed(vocab_size, embed_size)
    if task_name == 'traj_loc_pred':
        pre_model_name = param.get('pre_model_name', 'LPA')
        dataset_name = param['dataset']
        model_name = param['pre_model_name']
        exp_id = param['exp_id']
        train = param['train']
        saved_model = param['saved_model']
        activity_type = param["activity_type"]
        embed_name = param['embed_name']
        folder_name = '{}_{}_{}_{}'.format(exp_id, model_name, embed_name, activity_type, dataset_name)  #
        model_cache_file = './cache/model_cache/{}/model_cache/{}_final.m'.format(folder_name, folder_name)

        if pre_model_name == 'LPA':
            data_feature = dataset.data_feature
            train_data, valid_data, test_data = dataset.get_data_loaders()
            model = LPA(config=param, data_feature=data_feature, embed_layer=embed_layer)
            executor = TrajectoryExecutor(config=param, model=model, data_feature=data_feature)
            if train or not os.path.exists(model_cache_file):
                executor.train(train_data, valid_data)
                if saved_model:
                    executor.save_model(model_cache_file)
            else:
                executor.load_model(model_cache_file)
            executor.evaluate(test_data)


if __name__ == '__main__':
    pass
