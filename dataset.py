import math
import os
import itertools
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import pickle

from utils import parse_time, cal_timeoff, generate_dataloader_pad

parameter_list = ['dataset', 'min_session_len', 'min_sessions', "max_session_len", 'cut_method', 'window_size',
                  'min_checkins', 'activity_type']
all_parameter_list = ['dataset', 'min_session_len', 'min_sessions', "max_session_len", 'cut_method', 'window_size',
                      'min_checkins',
                      'geo_file', 'dyna_file']


class TrajectoryDataset:
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './data/{}/'.format(self.dataset)
        self.his_k_length = self.config.get('his_k_length', 5)
        self.cache_file_folder = './cache/dataset_cache/'
        self.data_cut_cache = './cache/dataset_cache/traj_cut'
        self.data_gen_cache = './cache/dataset_cache/traj_gen'
        self.data_enc_cache = './cache/dataset_cache/traj_enc'
        for param in parameter_list:
            self.data_cut_cache += '_' + str(self.config[param])
            self.data_gen_cache += '_' + str(self.config[param])
            self.data_enc_cache += '_' + str(self.config[param])
        self.data_cut_cache += '.json'
        self.data_gen_cache += '.json'
        self.data_enc_cache += '.pkl'

        self.encoder = TrajectoryEncoder(config=config)
        self.encoded_data = None
        self.generated_data = None
        self.datasets = None

        self.max_seq_len = 0

        self.get_train_sets()

        self.feature_dict = {}
        self.feature_max_len = {}
        self.pad_item = {}
        self.data_feature = {}
        self.gen_data_feature()


    def get_train_sets(self):
        """
        轨迹比较特殊，原子文件中存储的并不是轨迹而是一个一个点，因此需要先对轨迹进行切割
        """
        print('get_train_sets...')
        if self.datasets is not None:
            return self.datasets
        if self.generated_data is None:
            if os.path.exists(self.data_gen_cache):
                # load cache
                with open(self.data_gen_cache, 'r') as f:
                    self.generated_data = json.load(f)

                with open(self.data_enc_cache, "rb") as f:
                    self.encoder = pickle.load(f)

            else:
                if os.path.exists(self.data_cut_cache):
                    f = open(self.data_cut_cache, 'r')
                    cut_data = json.load(f)
                    f.close()
                else:
                    cut_data = self.cutter_filter()
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.data_cut_cache, 'w') as f:
                        json.dump(cut_data, f)
                print('finish cut data.')
                self.encoded_data = self.encoder.encode(cut_data)
                self.generated_data = self.generate_dataset()
                if self.config['cache_dataset']:
                    if not os.path.exists(self.cache_file_folder):
                        os.makedirs(self.cache_file_folder)
                    with open(self.data_gen_cache, 'w') as f:
                        json.dump(self.generated_data, f)
                    with open(self.data_enc_cache, "wb") as f:
                        pickle.dump(self.encoder, f)

        # user 来划，以及按轨迹数来划。
        train_data, eval_data, test_data = self.divide_dataset()
        self.datasets = (train_data, eval_data, test_data)
        return self.datasets


    def cutter_filter(self):
        """
        list(
            history_loc(list[int]), history_tim(list[int]), history_act(list[int]),
            current_loc(list[int]), current_tim(list[int]), current_act(list[int]),
            ...,
            ...,
            target_dyna_id(int), target_loc(int), target_tim(int), target_act(int), uid(int)
        )
        """
        # load data according to config
        dyna_file_path = os.path.join(
            self.data_path, '{}.dyna'.format(self.dyna_file))
        print(f'reading {dyna_file_path}')
        traj = pd.read_csv(dyna_file_path)
        # filter inactive poi
        group_location = traj.groupby('location').count()
        filter_location = group_location[group_location['time'] >= self.config['min_checkins']]
        location_index = filter_location.index.tolist()
        traj = traj[traj['location'].isin(location_index)]

        user_set = pd.unique(traj['entity_id'])
        res = {}
        min_session_len = self.config['min_session_len']
        max_session_len = self.config['max_session_len']
        min_sessions = self.config['min_sessions']
        window_size = self.config['window_size']
        cut_method = self.config['cut_method']
        if cut_method == 'time_interval':
            # 按照时间窗口进行切割
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in enumerate(usr_traj):
                    now_time = parse_time(row[2])
                    if index == 0:  # 第一条
                        session.append(row.tolist())
                        prev_time = now_time
                    else:
                        time_off = cal_timeoff(now_time, prev_time)  # 遇上一条时间差
                        if window_size > time_off >= 0 and len(session) < max_session_len:  # 在时间差内，并且session未饱和
                            session.append(row.tolist())
                        else:  # 在时间差外，或者session饱和
                            if len(session) >= min_session_len:  # 之前session长度符合最小记录
                                sessions.append(session)
                            session = [row.tolist()]
                    prev_time = now_time
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        elif cut_method == 'same_date':
            # 将同一天的 check-in 划为一条轨迹
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                prev_date = None
                for index, row in enumerate(usr_traj):
                    now_time = parse_time(row[2])
                    now_date = now_time.day
                    if index == 0:
                        session.append(row.tolist())
                    else:
                        if prev_date == now_date and len(session) < max_session_len:
                            # 还是同一天
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = [row.tolist()]
                    prev_date = now_date
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        else:
            # cut by fix window_len used by STAN
            if max_session_len != window_size:
                raise ValueError('the fixed length window is not equal to max_session_len')
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in enumerate(usr_traj):
                    if len(session) < window_size:
                        session.append(row.tolist())
                    else:
                        sessions.append(session)
                        session = [row.tolist()]
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        return res

    def get_data_loaders(self):
        train_dataset, eval_dataset, test_dataset = self.get_train_sets()
        train_loader, eval_loader, test_loader = generate_dataloader_pad(train_dataset, eval_dataset, test_dataset,
                                                                         self.feature_dict,
                                                                         self.config['batch_size'],
                                                                         self.config['num_workers'], self.pad_item,
                                                                         self.feature_max_len)
        return train_loader, eval_loader, test_loader

    def gen_data_feature(self):
        # Eating out,Personal affairs,Recreation,Shopping,Home,Work
        self.feature_dict = {
            'history_loc': 'int', 'history_tim': 'int', 'history_act': 'int',

            'current_loc': 'int', 'current_tim': 'int', 'current_act': 'int',  # array of int

            'history_p_e': 'float', 'history_p_p': 'float', 'history_p_r': 'float', 'history_p_s': 'float',
            'history_p_h': 'float', 'history_p_w': 'float',

            'current_p_e': 'float', 'current_p_p': 'float', 'current_p_r': 'float', 'current_p_s': 'float',
            'current_p_h': 'float', 'current_p_w': 'float',

            'target_dyna_id': 'int', 'target_loc': 'int', 'target_tim': 'int', 'target_act': 'int', 'uid': 'int',
        }

        self.pad_item = {
            'current_loc': self.encoder.num_loc,
            'history_loc': self.encoder.num_loc,
            'current_tim': self.encoder.num_tim,
            'history_tim': self.encoder.num_tim,
            'current_act': self.encoder.num_act,
            'history_act': self.encoder.num_act,

            'history_p_e': 0,
            'history_p_p': 0,
            'history_p_r': 0,
            'history_p_s': 0,
            'history_p_h': 0,
            'history_p_w': 0,

            'current_p_e': 0,
            'current_p_p': 0,
            'current_p_r': 0,
            'current_p_s': 0,
            'current_p_h': 0,
            'current_p_w': 0,

        }
        self.data_feature = {
            'encode_columns': ['history_loc', 'history_tim', 'history_act',

                               'current_loc', 'current_tim', 'current_act',

                               'history_p_e', 'history_p_p', 'history_p_r', 'history_p_s',
                               'history_p_h', 'history_p_w',

                               'current_p_e', 'current_p_p', 'current_p_r', 'current_p_s',
                               'current_p_h', 'current_p_w',

                               'target_dyna_id', 'target_loc', 'target_tim', 'target_act', 'uid'],
            'loc_pad': self.encoder.num_loc,
            'tim_pad': self.encoder.num_tim,
            'act_pad': self.encoder.num_act,
            'loc_size': self.encoder.num_loc + 1,
            'tim_size': self.encoder.num_tim + 1,
            'act_size': self.encoder.num_act + 1,
            'uid_size': self.encoder.num_uid,
            'location_decoder': self.encoder.loc_de,
            'activity_decoder': self.encoder.act_de,
        }


    def get_pretrain_set(self):
        """
        @input:
            list(
                history_loc(list[int]), history_tim(list[int]), history_act(list[int]),
                ...,
                current_loc(list[int]), current_tim(list[int]), current_act(list[int]),
                ...,
                target_dyna_id(int), target_loc(int), target_tim(int), target_act(int), uid(int)
            )
        @return:
            list(
                history_loc(list[int]), history_tim(list[int]), history_act(list[int]), uid(int)
            )
        """
        train_data, eval_data, test_data = self.get_train_sets()
        pretrain_data = [[record[0], record[1], record[2], record[-1]] for record in train_data]
        pretrain_data.sort()
        pretrain_data_drop_duplicates = list(k for k, _ in itertools.groupby(pretrain_data))
        return pretrain_data_drop_duplicates


    def generate_dataset(self):
        res_en = self.encoded_data
        if res_en is None:
            return None

        data_set = {}
        for uid in tqdm(res_en.keys(), desc="generating dataset..."):
            trajectories = res_en[uid]
            record_set = []
            history_loc = []
            history_tim = []
            history_act = []

            history_p_e = []
            history_p_p = []
            history_p_r = []
            history_p_s = []
            history_p_h = []
            history_p_w = []
            for tra_idx, trajectory in enumerate(trajectories):
                # omit: type
                dyna_id_session, _, tim_session, user_idx_session, loc_session, pri_act_session, sec_act_session, eat_session, affair_session, recreation_session, shop_session, home_session, work_session = map(
                    list, zip(
                        *trajectory))
                if tra_idx != 0:
                    for i in range(len(loc_session) - 1):
                        target_dyna_id = dyna_id_session[i + 1]
                        target_loc = loc_session[i + 1]
                        target_tim = tim_session[i + 1]
                        target_act = sec_act_session[i + 1]
                        target_uid = user_idx_session[i + 1]

                        current_loc = loc_session[:i + 1]
                        current_tim = tim_session[:i + 1]
                        current_act = sec_act_session[:i + 1]

                        current_p_e = eat_session[:i + 1]
                        current_p_p = affair_session[:i + 1]
                        current_p_r = recreation_session[:i + 1]
                        current_p_s = shop_session[:i + 1]
                        current_p_h = home_session[:i + 1]
                        current_p_w = work_session[:i + 1]

                        trace = [
                            history_loc.copy(), history_tim.copy(), history_act.copy(),
                            current_loc.copy(), current_tim.copy(), current_act.copy(),

                            history_p_e.copy(), history_p_p.copy(), history_p_r.copy(), history_p_s.copy(),
                            history_p_h.copy(), history_p_w.copy(),
                            current_p_e.copy(), current_p_p.copy(), current_p_r.copy(), current_p_s.copy(),
                            current_p_h.copy(), current_p_w.copy(),

                            target_dyna_id, target_loc, target_tim, target_act, target_uid,
                        ]
                        record_set.append(trace)
                # important fix for history trajectory
                history_loc += loc_session
                history_tim += tim_session
                history_act += sec_act_session
                history_p_e += eat_session
                history_p_p += affair_session
                history_p_r += recreation_session
                history_p_s += shop_session
                history_p_h += home_session
                history_p_w += work_session
            data_set[uid] = record_set
        return data_set

    def divide_dataset(self):
        train_rate = self.config['train_rate']
        eval_rate = self.config['eval_rate']
        data_set = self.generated_data.copy()
        his_k_length = self.his_k_length

        max_seq_len = 0
        train_data = []
        eval_data = []
        test_data = []
        for uid in tqdm(data_set.keys(), desc="dividing data"):
            user_dataset = data_set[uid]
            for record in user_dataset:
                # history_loc, history_tim, history_act, current_loc, current_tim, current_act,
                # history_p_e, history_p_p, history_p_r, history_p_s, history_p_h, history_p_w,
                # current_p_e, current_p_p, current_p_r, current_p_s, current_p_h, current_p_w,
                # target_dyna_id, target_loc, target_tim, target_act, target_uid,
                record[0] = record[0][:his_k_length]
                record[1] = record[1][:his_k_length]
                record[2] = record[2][:his_k_length]
                record[6] = record[6][:his_k_length]
                record[7] = record[7][:his_k_length]
                record[8] = record[8][:his_k_length]
                record[9] = record[9][:his_k_length]
                record[10] = record[10][:his_k_length]
                record[11] = record[11][:his_k_length]

                his_len = len(record[0])
                cur_len = len(record[3])
                seq_len = his_len if his_len > cur_len else cur_len
                max_seq_len = seq_len if seq_len > max_seq_len else max_seq_len
            set_len = len(user_dataset)
            # 根据 traj_len 来划分 train eval test
            train_num = math.ceil(set_len * train_rate)  # 浮点数向上取整
            eval_num = math.ceil(set_len * (train_rate + eval_rate))
            train_data += user_dataset[:train_num]
            eval_data += user_dataset[train_num:eval_num]
            test_data += user_dataset[eval_num:]
        self.max_seq_len = max_seq_len
        print(f'max_seq_len:{max_seq_len}, his_k_length:{his_k_length}')
        return train_data, eval_data, test_data


class TrajectoryEncoder:
    def __init__(self, config):
        # dyna_id,type,time,entity_id,location,activity,
        # Eating out,Personal affairs,Recreation,Shopping,Home,Work
        # #encode: time,entity_id,location,activity,
        # #origin: dyna_id, Eating out,Personal affairs,Recreation,Shopping,Home,Work
        # #useless: type
        self.uid_en = {}
        self.uid_de = {}
        self.num_uid = 0

        self.loc_en = {}
        self.loc_de = {}
        self.num_loc = 0

        self.num_tim = 48

        self.act_type = config.get('activity_type', 'A3')
        self.act_en = {}
        self.act_de = {}
        self.num_act = 0

    def uid_encoder(self, uid_str):
        if uid_str not in self.uid_en.keys():
            self.uid_en[uid_str] = self.num_uid
            self.uid_de[self.num_uid] = uid_str
            self.num_uid += 1
        return self.uid_en[uid_str]

    def loc_encoder(self, loc_str):
        if loc_str not in self.loc_en.keys():
            self.loc_en[loc_str] = self.num_loc
            self.loc_de[self.num_loc] = loc_str
            self.num_loc += 1
        return self.loc_en[loc_str]

    def act_encoder(self, act_str):
        if self.act_type == 'A3':
            act_str = act_str if act_str in ['Home', 'Work'] else 'Others'
        elif self.act_type == 'A6':
            pass
        if act_str not in self.act_en.keys():
            self.act_en[act_str] = self.num_act
            self.act_de[self.num_act] = act_str
            self.num_act += 1
        return self.act_en[act_str]

    def tim_encoder(self, time_str):
        time = parse_time(time_str)
        if time.weekday() in [0, 1, 2, 3, 4]:
            return time.hour
        else:
            return time.hour + 24

    def encode(self, res: dict):
        res_en = res.copy()
        for trajectories in tqdm(res_en.values(), desc="encoding trajectory"):
            for trajectory in trajectories:
                for point in trajectory:
                    # dyna_id,type,time,entity_id,location,primary_activity,secondary_activity,Eating out,Personal affairs,Recreation,Shopping,Home,Work
                    # [2546363, 'trajectory', '2020-10-16T16:36:00Z', 1852471, 424559, 'Work', 0.0, 0.0, 0.0, 0.0, 0, 1]
                    point[2] = self.tim_encoder(time_str=point[2])
                    point[3] = self.uid_encoder(uid_str=str(point[3]))
                    point[4] = self.loc_encoder(loc_str=str(point[4]))
                    point[6] = self.act_encoder(act_str=point[6])
        return res_en


if __name__ == '__main__':
    from args import default_args

    # raw_df = pd.read_csv('./data/gz/activity_20201012_20201018.csv')
    # id, start_time, end_time, centroid_x, centroid_y, regrid_id,
    # activity,index_df,secondary_activity,Eating out,Personal affairs,Recreation,Shopping,ACT,day
    dataset = TrajectoryDataset(default_args)
    dataset.datasets = [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1], [2], [3], 0, 0, 0, 0, 0],
                        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1], [2], [3], 0, 0, 0, 0, 0],
                        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1], [2], [3], 0, 0, 0, 0, 0],
                        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1], [2], [3], 0, 0, 0, 0, 0], ], None, None
    dataset.get_pretrain_set()
    # history_loc.copy(), history_tim.copy(), history_act.copy(),
    #                             current_loc.copy(), current_tim.copy(), current_act.copy(),
    #                             target_dyna_id, target_loc, target_tim, target_act, uid
