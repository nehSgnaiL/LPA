import os
import json
import torch


class ConfigParser(object):
    """
    use to parse the user defined parameters and use these to modify the
    pipeline's parameter setting.
    值得注意的是，目前各阶段的参数是放置于同一个 dict 中的，因此需要编程时保证命名空间不冲突。
    config 优先级：命令行 > config file > default config
    """

    def __init__(self, task, model, dataset, config_file=None,
                 saved_model=True, train=True, other_args=None, hyper_config_dict=None):
        """
        Args:
            task, downstream, dataset (str): 用户在命令行必须指明的三个参数
            config_file (str): 配置文件的文件名，将在项目根目录下进行搜索
            other_args (dict): 通过命令行传入的其他参数
        """
        self.config = {}
        self._parse_external_config(task, model, dataset, saved_model, train, other_args, hyper_config_dict)
        self._parse_config_file(config_file)
        self._init_device()

    def _parse_external_config(self, task, model, dataset,
                               saved_model=True, train=True, other_args=None, hyper_config_dict=None):
        if task is None:
            raise ValueError('the parameter task should not be None!')
        if model is None:
            raise ValueError('the parameter downstream should not be None!')
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        # 目前暂定这三个参数必须由用户指定
        self.config['task'] = task
        self.config['downstream'] = model
        self.config['dataset'] = dataset
        self.config['saved_model'] = saved_model
        self.config['train'] = False if task == 'map_matching' else train
        if other_args is not None:
            for key in other_args:
                self.config[key] = other_args[key]
        if hyper_config_dict is not None:
            # 超参数调整时传入的待调整的参数，优先级低于命令行参数
            for key in hyper_config_dict:
                self.config[key] = hyper_config_dict[key]

    def _parse_config_file(self, config_file):
        if config_file is not None:
            if os.path.exists('./{}.json'.format(config_file)):
                with open('./{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)
        if use_gpu:
            torch.cuda.set_device(gpu_id)
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if torch.cuda.is_available() and use_gpu else "cpu")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    # 支持迭代操作
    def __iter__(self):
        return self.config.__iter__()