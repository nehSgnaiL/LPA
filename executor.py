import torch
import torch.optim as optim
import numpy as np
import os
from logging import getLogger
import pandas as pd
import json

from tqdm import tqdm

from evaluator import TrajectoryEvaluator


class TrajectoryExecutor:

    def __init__(self, config, model, data_feature):
        self.evaluator = TrajectoryEvaluator(config)
        self.metrics = 'Recall@1'
        self.config = config
        self.model = model.to(self.config['device'])
        self.exp_id = self.config.get('exp_id', None)

        model_name = self.config['pre_model_name']
        embed_name = self.config['embed_name']
        activity_type = self.config['activity_type']
        dataset_name = self.config['dataset']
        self.folder_name = '{}_{}_{}_{}'.format(self.exp_id, model_name, embed_name, activity_type, dataset_name)  #
        self.tmp_path = './cache/model_cache/{}/checkpoint/'.format(self.folder_name)
        self.cache_dir = './cache/model_cache/{}/model_cache/'.format(self.folder_name)
        self.evaluate_res_dir = './cache/model_cache/{}/evaluate_cache/'.format(self.folder_name)

        self.loss_func = None
        self._logger = getLogger()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        # ------------------
        # key type may change during load json
        self.location_decoder = {int(k): int(v) for k, v in data_feature['location_decoder'].items()}
        self.activity_decoder = data_feature['activity_decoder']

        dataset = self.config.get('dataset', '')
        data_path = './data/{}/'.format(dataset)
        geo_file = self.config.get('geo_file', dataset)
        geo_file_path = os.path.join(data_path, '{}.geo'.format(geo_file))
        # load data according to config
        geo_point = pd.read_csv(geo_file_path)
        geo_point['coordinate'] = geo_point['coordinate'].apply(lambda x: json.loads(x))  # turn json string to list
        self.geo_dict = pd.Series(data=geo_point['coordinate'].tolist(), index=geo_point['geo_id']).to_dict()
        self.model_file_path_list = []
        # ------------------

    def train(self, train_dataloader, eval_dataloader):
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        metrics = {}
        metrics['accuracy'] = []
        metrics['loss'] = []
        lr = self.config['learning_rate']
        for epoch in range(self.config['max_epoch']):
            self._logger.info('start train')
            self.model, avg_loss = self.run(train_dataloader, self.model,
                                            self.config['learning_rate'], self.config['clip'])
            self._logger.info('==>Train Epoch:{:4d} Loss:{:.5f} learning_rate:{}'.format(
                epoch, avg_loss, lr))
            # eval stage
            self._logger.info('start evaluate')
            avg_eval_acc, avg_eval_loss = self._valid_epoch(eval_dataloader, self.model)
            self._logger.info('==>Eval Acc:{:.5f} Eval Loss:{:.5f}'.format(avg_eval_acc, avg_eval_loss))
            metrics['accuracy'].append(avg_eval_acc)
            metrics['loss'].append(avg_eval_loss)
            if self.config['hyper_tune']:
                pass
            else:
                save_name_tmp = "{}_ep_{}.m".format(self.folder_name, epoch)  #
                self.model_file_path_list.append(self.tmp_path + save_name_tmp)  # record tmp downstream
                torch.save(self.model.state_dict(), self.tmp_path + save_name_tmp)
            self.scheduler.step(avg_eval_acc)
            # scheduler 会根据 avg_eval_acc 减小学习率
            # 若当前学习率小于特定值，则 early stop
            lr = self.optimizer.param_groups[0]['lr']
            if lr < self.config['early_stop_lr']:
                break
        if not self.config['hyper_tune'] and self.config['load_best_epoch']:
            best = np.argmax(metrics['accuracy'])
            load_name_tmp = "{}_ep_{}.m".format(self.folder_name, best)  #
            self.model.load_state_dict(
                torch.load(self.tmp_path + load_name_tmp))
        # 删除之前创建的临时文件夹
        # for rt, dirs, files in os.walk(self.tmp_path):
        #     for name in files:
        #         remove_path = os.path.join(rt, name)
        #         os.remove(remove_path)
        # os.rmdir(self.tmp_path)
        for file_path in self.model_file_path_list:  #
            os.remove(file_path)  #

    def load_model(self, cache_name):
        model_state, optimizer_state = torch.load(cache_name)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def save_model(self, cache_name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # save optimizer when load epoch to train
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def evaluate(self, test_dataloader):
        self.model.train(False)
        self.evaluator.clear()
        run_bar = tqdm(test_dataloader)
        for batch in run_bar:
            batch.to_tensor(device=self.config['device'])
            scores = self.model.predict(batch)
            if self.config['evaluate_method'] == 'popularity':
                evaluate_input = {
                    'location_decoder': self.location_decoder,  #
                    'geo_dict': self.geo_dict,  #
                    'target_dyna_id': batch['target_dyna_id'].tolist(),  #
                    'uid': batch['uid'].tolist(),
                    'loc_true': batch['target_loc'].tolist(),
                    'loc_pred': scores.tolist()
                }
            else:
                # negative sample
                # loc_true is always 0
                loc_true = [0] * self.config['batch_size']
                evaluate_input = {
                    'location_decoder': self.location_decoder,  #
                    'geo_dict': self.geo_dict,  #
                    'target_dyna_id': batch['target_dyna_id'].tolist(),  #
                    'uid': batch['uid'].tolist(),
                    'loc_true': loc_true,
                    'loc_pred': scores.tolist()
                }
            self.evaluator.collect(evaluate_input)
        self.evaluator.save_result(self.evaluate_res_dir)
        # save the downstream object
        torch.save(self.model, os.path.join(self.cache_dir, 'evaluated_downstream.pt'))

    def run(self, data_loader, model, lr, clip):
        model.train(True)
        if self.config['debug']:
            torch.autograd.set_detect_anomaly(True)
        total_loss = []
        loss_func = self.loss_func or model.calculate_loss
        run_bar = tqdm(data_loader)
        for batch in run_bar:
            # one batch, one step
            self.optimizer.zero_grad()
            batch.to_tensor(device=self.config['device'])
            loss = loss_func(batch)
            if self.config['debug']:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()
            total_loss.append(loss.data.cpu().numpy().tolist())
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            except:
                pass
            self.optimizer.step()
            run_bar.set_description('Loss: {:.3f}'.format(loss.item()))
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return model, avg_loss

    def _valid_epoch(self, data_loader, model):
        model.train(False)
        self.evaluator.clear()
        total_loss = []
        loss_func = self.loss_func or model.calculate_loss
        run_bar = tqdm(data_loader)
        for batch in run_bar:
            batch.to_tensor(device=self.config['device'])
            scores = model.predict(batch)
            loss = loss_func(batch)
            total_loss.append(loss.data.cpu().numpy().tolist())
            if self.config['evaluate_method'] == 'popularity':
                evaluate_input = {
                    'location_decoder': self.location_decoder,  #
                    'geo_dict': self.geo_dict,  #
                    'target_dyna_id': batch['target_dyna_id'].tolist(),  #
                    'uid': batch['uid'].tolist(),
                    'loc_true': batch['target_loc'].tolist(),
                    'loc_pred': scores.tolist()
                }
            else:
                # negative sample
                # loc_true is always 0
                loc_true = [0] * self.config['batch_size']
                evaluate_input = {
                    'location_decoder': self.location_decoder,  #
                    'geo_dict': self.geo_dict,  #
                    'target_dyna_id': batch['target_dyna_id'].tolist(),  #
                    'uid': batch['uid'].tolist(),
                    'loc_true': loc_true,
                    'loc_pred': scores.tolist()
                }
            self.evaluator.collect(evaluate_input)
        avg_acc = self.evaluator.evaluate()[self.metrics]  # 随便选一个就行
        avg_loss = np.mean(total_loss, dtype=np.float64)
        return avg_acc, avg_loss

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        if self.config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                   weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'],
                                        weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.config['learning_rate'],
                                            weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config['learning_rate'],
                                            weight_decay=self.config['L2'])
        elif self.config['optimizer'] == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.config['learning_rate'])
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                                   weight_decay=self.config['L2'])
        return optimizer

    def _build_scheduler(self):
        """
        目前就固定的 scheduler 吧
        """
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max',
                                                         patience=self.config['lr_step'],
                                                         factor=self.config['lr_decay'],
                                                         threshold=self.config['schedule_threshold'])
        return scheduler
