import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, data):
        """
        data: 必须是一个 list
        """
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Batch(object):

    def __init__(self, feature_name):
        """Summary of class here

        Args:
            feature_name (dict): key is the corresponding feature's name, and
                the value is the feature's data type
        """
        self.data = {}
        self.feature_name = feature_name
        for key in feature_name:
            self.data[key] = []

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def append(self, item):
        """
        append a new item into the batch

        Args:
            item (list): 一组输入，跟feature_name的顺序一致，feature_name即是这一组输入的名字
        """
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i, key in enumerate(self.feature_name):
            self.data[key].append(item[i])

    def to_tensor(self, device):
        """
        将数据self.data转移到device上

        Args:
            device(torch.device): GPU/CPU设备
        """
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = torch.LongTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'float':
                self.data[key] = torch.FloatTensor(np.array(self.data[key])).to(device)
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))

    def to_ndarray(self):
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = np.array(self.data[key])
            elif self.feature_name[key] == 'float':
                self.data[key] = np.array(self.data[key])
            else:
                raise TypeError(
                    'Batch to_ndarray, only support int, float but you give {}'.format(self.feature_name[key]))


class BatchPAD(Batch):

    def __init__(self, feature_name, pad_item=None, pad_max_len=None):
        """Summary of class here

        Args:
            feature_name (dict): key is the corresponding feature's name, and
                the value is the feature's data type
            pad_item (dict): key is the feature name, and value is the padding
                value. We will just padding the feature in pad_item
            pad_max_len (dict): key is the feature name, and value is the max
                length of padded feature. use this parameter to truncate the
                feature.
        """
        super().__init__(feature_name=feature_name)
        # 默认是根据 batch 中每个特征最长的长度来补齐，如果某个特征的长度超过了 pad_max_len 则进行剪切
        self.pad_len = {}
        self.origin_len = {}  # 用于得知补齐前轨迹的原始长度
        self.pad_max_len = pad_max_len if pad_max_len is not None else {}
        self.pad_item = pad_item if pad_item is not None else {}
        for key in feature_name:
            self.data[key] = []
            if key in self.pad_item:
                self.pad_len[key] = 0
                self.origin_len[key] = []

    def append(self, item):
        """
        append a new item into the batch

        Args:
            item (list): 一组输入，跟feature_name的顺序一致，feature_name即是这一组输入的名字
        """
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i, key in enumerate(self.feature_name):
            # 需保证 item 每个特征的顺序与初始化时传入的 feature_name 中特征的顺序一致
            self.data[key].append(item[i])
            if key in self.pad_item:
                self.origin_len[key].append(len(item[i]))
                if self.pad_len[key] < len(item[i]):
                    # 保持 pad_len 是最大的
                    self.pad_len[key] = len(item[i])

    def padding(self):
        """
        只提供对一维数组的特征进行补齐
        """
        for key in self.pad_item:
            # 只对在 pad_item 中的特征进行补齐
            if key not in self.data:
                raise KeyError('when pad a batch, raise this error!')
            max_len = self.pad_len[key]
            if key in self.pad_max_len:
                max_len = min(self.pad_max_len[key], max_len)
            for i in range(len(self.data[key])):
                if len(self.data[key][i]) < max_len:
                    self.data[key][i] += [self.pad_item[key]] * \
                        (max_len - len(self.data[key][i]))
                else:
                    # 截取的原则是，抛弃前面的点
                    # 因为是时间序列嘛
                    self.data[key][i] = self.data[key][i][-max_len:]
                    # 对于剪切了的，我们没办法还原，但至少不要使他出错
                    self.origin_len[key][i] = max_len

    def get_origin_len(self, key):
        return self.origin_len[key]

    def to_tensor(self, device):
        """
        将数据self.data转移到device上

        Args:
            device(torch.device): GPU/CPU设备
        """
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = torch.LongTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'float':
                self.data[key] = torch.FloatTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'array of int':
                for i in range(len(self.data[key])):
                    for j in range(len(self.data[key][i])):
                        try:
                            self.data[key][i][j] = torch.LongTensor(np.array(self.data[key][i][j])).to(device)
                        except TypeError:
                            print('device is ', device)
                            print(key, self.feature_name[key])
                            exit()
            # ------------------------------------------
            elif self.feature_name[key] == 'array of float':
                for i in range(len(self.data[key])):
                    for j in range(len(self.data[key][i])):
                        try:
                            self.data[key][i][j] = torch.FloatTensor(np.array(self.data[key][i][j])).to(device)
                        except TypeError:
                            print('device is ', device)
                            print(key, self.feature_name[key])
                            print(self.data[key][i][j])
                            exit()
            # ------------------------------------------
            elif self.feature_name[key] == 'no_pad_int':
                for i in range(len(self.data[key])):
                    self.data[key][i] = torch.LongTensor(np.array(self.data[key][i])).to(device)
            elif self.feature_name[key] == 'no_pad_float':
                for i in range(len(self.data[key])):
                    self.data[key][i] = torch.FloatTensor(np.array(self.data[key][i])).to(device)
            elif self.feature_name[key] == 'no_tensor':
                pass
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))


def generate_dataloader(train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, shuffle=True,
                        pad_with_last_sample=False):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        shuffle(bool): shuffle
        pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    if pad_with_last_sample:
        num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
        train_data = np.concatenate([train_data, data_padding], axis=0)
        num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
        eval_data = np.concatenate([eval_data, data_padding], axis=0)
        num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
        data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
        test_data = np.concatenate([test_data, data_padding], axis=0)

    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        batch = Batch(feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=False)
    return train_dataloader, eval_dataloader, test_dataloader


def generate_dataloader_pad(train_data, eval_data, test_data, feature_name,
                            batch_size, num_workers, pad_item=None,
                            pad_max_len=None, shuffle=True):
    """
    create dataloader(train/test/eval)

    Args:
        train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
        feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
        batch_size(int): batch_size
        num_workers(int): num_workers
        pad_item(dict): 用于将不定长的特征补齐到一样的长度，每个特征名作为 key，若某特征名不在该 dict 内则不进行补齐。
        pad_max_len(dict): 用于截取不定长的特征，对于过长的特征进行剪切
        shuffle(bool): shuffle

    Returns:
        tuple: tuple contains:
            train_dataloader: Dataloader composed of Batch (class) \n
            eval_dataloader: Dataloader composed of Batch (class) \n
            test_dataloader: Dataloader composed of Batch (class)
    """
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    test_dataset = ListDataset(test_data)

    def collator(indices):
        batch = BatchPAD(feature_name, pad_item, pad_max_len)
        for item in indices:
            batch.append(copy.deepcopy(item))
        batch.padding()
        return batch

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collator,
                                  shuffle=shuffle)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collator,
                                 shuffle=shuffle)
    return train_dataloader, eval_dataloader, test_dataloader