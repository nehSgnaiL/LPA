import os
import json
import time

import pandas as pd

from utils import top_k, top_k_geo
from logging import getLogger

allowed_metrics = ['Precision', 'Recall', 'F1', 'MRR', 'MAP', 'NDCG', 'GeoDis']


class TrajectoryEvaluator:

    def __init__(self, config):
        self.metrics = config['metrics']  # 评估指标, 是一个 list
        self.config = config
        self.topk = config['topk']
        self.result = {}
        # 兼容全样本评估与负样本评估
        self.evaluate_method = config['evaluate_method']
        self.intermediate_result = {
            'geo_diff': 0.0,
            'total': 0,
            'hit': 0,
            'rank': 0.0,
            'dcg': 0.0
        }
        # --------------------------
        # alter top_k to top_k list
        if not isinstance(self.topk, list):
            self.topk = [self.topk]
        self.result = {}
        self.intermediate_result = [{
            'geo_diff': 0.0,
            'total': 0,
            'hit': 0,
            'rank': 0.0,
            'dcg': 0.0
        } for k in self.topk]
        self.record_hit_table = pd.DataFrame()
        # --------------------------
        self._check_config()
        self._logger = getLogger()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for i in self.metrics:
            if i not in allowed_metrics:
                raise ValueError('the metric is not allowed in \
                    TrajLocPredEvaluator')

    def collect(self, batch):
        """
        Args:
            batch (dict): contains three keys: uid, loc_true, and loc_pred.
            uid (list): 来自于 batch 中的 uid，通过索引可以确定 loc_true 与 loc_pred
                中每一行（元素）是哪个用户的一次输入。
            loc_true (list): 期望地点(target)，来自于 batch 中的 target。
                对于负样本评估，loc_pred 中第一个点是 target 的置信度，后面的都是负样本的
            loc_pred (matrix): 实际上模型的输出，batch_size * output_dim.
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        for k_index, topk in enumerate(self.topk):
            location_decoder = batch['location_decoder']
            geo_dict = batch['geo_dict']
            hit, rank, dcg, geo_diff, hit_dict, predict_dict = top_k_geo(batch, topk, location_decoder, geo_dict)
            total = len(batch['loc_true'])
            self.intermediate_result[k_index]['geo_diff'] += geo_diff
            self.intermediate_result[k_index]['total'] += total
            self.intermediate_result[k_index]['hit'] += hit
            self.intermediate_result[k_index]['rank'] += rank
            self.intermediate_result[k_index]['dcg'] += dcg
            dyna_hit_key = 'hit@{}'.format(topk)
            predict_key = 'predict@1'
            for user_id in hit_dict.keys():
                self.record_hit_table.loc[user_id, predict_key] = predict_dict[user_id]
                self.record_hit_table.loc[user_id, dyna_hit_key] = hit_dict[user_id]

    def evaluate(self):
        for k_index, topk in enumerate(self.topk):
            precision_key = 'Precision@{}'.format(topk)
            precision = self.intermediate_result[k_index]['hit'] / (
                    self.intermediate_result[k_index]['total'] * topk)
            if 'Precision' in self.metrics:
                self.result[precision_key] = precision
            # recall is used to valid in the trainning, so must exit
            recall_key = 'Recall@{}'.format(topk)
            recall = self.intermediate_result[k_index]['hit'] \
                     / self.intermediate_result[k_index]['total']
            self.result[recall_key] = recall
            if 'F1' in self.metrics:
                f1_key = 'F1@{}'.format(topk)
                if precision + recall == 0:
                    self.result[f1_key] = 0.0
                else:
                    self.result[f1_key] = (2 * precision * recall) / (precision +
                                                                      recall)
            if 'MRR' in self.metrics:
                mrr_key = 'MRR@{}'.format(topk)
                self.result[mrr_key] = self.intermediate_result[k_index]['rank'] \
                                       / self.intermediate_result[k_index]['total']
            if 'MAP' in self.metrics:
                map_key = 'MAP@{}'.format(topk)
                self.result[map_key] = self.intermediate_result[k_index]['rank'] \
                                       / self.intermediate_result[k_index]['total']
            if 'NDCG' in self.metrics:
                ndcg_key = 'NDCG@{}'.format(topk)
                self.result[ndcg_key] = self.intermediate_result[k_index]['dcg'] \
                                        / self.intermediate_result[k_index]['total']
            if 'GeoDis' in self.metrics:
                geo_key = 'GeoDis@{}'.format(topk)
                self.result[geo_key] = self.intermediate_result[k_index]['geo_diff'] \
                                       / self.intermediate_result[k_index]['total']

        return self.result

    def save_result(self, save_path, filename=None):
        self.evaluate()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if filename is None:
            # 使用时间戳
            filename = time.strftime(
                "%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        self._logger.info('evaluate result is {}'.format(json.dumps(self.result, indent=1)))
        with open(os.path.join(save_path, '{}.json'.format(filename)), 'w') \
                as f:
            json.dump(self.result, f)
        self.record_hit_table.index.name = 'target_dyna_id'
        self.record_hit_table.to_csv(os.path.join(save_path, '{}.csv'.format(filename)),
                                     index=True, header=True, encoding='utf-8')

    def clear(self):
        self.result = {}
        self.intermediate_result = {
            'geo_diff': 0.0,
            'total': 0,
            'hit': 0,
            'rank': 0.0,
            'dcg': 0.0
        }
        # --------------------------
        # alter top_k to top_k list
        if not isinstance(self.topk, list):
            self.topk = [self.topk]
        self.result = {}
        self.intermediate_result = [{
            'geo_diff': 0.0,
            'total': 0,
            'hit': 0,
            'rank': 0.0,
            'dcg': 0.0
        } for k in self.topk]
        self.record_hit_table = pd.DataFrame()
        # --------------------------
