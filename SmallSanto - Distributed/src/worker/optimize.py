import os
from datetime import datetime
from time import sleep
import dropbox

import keras.backend as K
import numpy as np
from keras.optimizers import SGD

from agent.model import GameModel, objective_function_for_policy, objective_function_for_value
from config import Config
from src.lib import tf_util
#from lib.tf_util import *
from src.lib.data_helper import get_game_data_filenames, read_game_data_from_file, get_next_generation_model_dirs
from src.lib.model_helpler import load_best_model_weight
from env.game_env import GameEnv, Player

def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.59)
    return OptimizeWorker(config).start()

class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: GameModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.optimizer = None
        self.dbx = None
    def start(self):
        auth_token = 'UlBTypwXWYAAAAAAAAAAEP6hKysZi9cQKGZTmMu128TYEEig00w3b3mJ--b_6phN'
        self.dbx = dropbox.Dropbox(auth_token)
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        last_load_data_step = last_save_step = total_steps = self.config.trainer.start_total_steps
        min_data_size_to_learn = 5000
        self.load_play_data()

        while True:
            if self.dataset_size < min_data_size_to_learn:
                print("dataset_size=",self.dataset_size," is less than ",min_data_size_to_learn)
                sleep(300)
                self.load_play_data()
                continue
            self.update_learning_rate(total_steps)
            steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            total_steps += steps
            if last_save_step + self.config.trainer.save_model_steps < total_steps:
                self.save_current_model()
                last_save_step = total_steps
            if last_load_data_step + self.config.trainer.load_data_steps < total_steps:
                self.load_play_data()
                last_load_data_step = total_steps

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, z_ary = self.dataset
        self.model.model.fit(state_ary, [policy_ary, z_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs, verbose=1)
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4

        if total_steps < 100000:
            lr = 1e-2
        elif total_steps < 500000:
            lr = 1e-3
        elif total_steps < 900000:
            lr = 1e-4
        else:
            lr = 2.5e-5  # means (1e-4 / 4): the paper batch size=2048, ours is 512.
        K.set_value(self.optimizer.lr, lr)
        print("total step=",total_steps,", set learning rate to ",lr)

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

        # Saving File to Drop Box
        with open(config_path, 'rb') as f:
            data = f.read()
        res = self.dbx.files_upload(data, '/model/next_generation/'+model_dir+'/'+next_generation_model_config_filename, dropbox.files.WriteMode.add, mute=True)
        with open(weight_path, 'rb') as f:
            data = f.read()
        res = self.dbx.files_upload(data, '/model/next_generation/'+model_dir+'/'+next_generation_model_weight_filename, dropbox.files.WriteMode.add, mute=True)

    def collect_all_loaded_data(self):
        state_ary_list, policy_ary_list, z_ary_list = [], [], []
        for s_ary, p_ary, z_ary_ in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            z_ary_list.append(z_ary_)

        state_ary = np.concatenate(state_ary_list)
        policy_ary = np.concatenate(policy_ary_list)
        z_ary = np.concatenate(z_ary_list)
        return state_ary, policy_ary, z_ary

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])

    def load_model(self):            
        for entry in self.dbx.files_list_folder('/model').entries:
            md, res = self.dbx.files_download('/model/'+entry.name)
            with open('SantoriniAZ-Dist/SmallSanto - Distributed/data/model/'+entry.name, 'wb') as f:  
            #with open('./data/model/'+entry.name, 'wb') as f:  
                f.write(res.content)

        from agent.model import GameModel
        model = GameModel(self.config)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            print("loading best model")
            if not load_best_model_weight(model):
                print("Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            print("loading latest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model

    def load_play_data(self):
        for entry in self.dbx.files_list_folder('/play_data').entries:
            md, res = self.dbx.files_download('/play_data/'+entry.name)
            with open('SantoriniAZ-Dist/SmallSanto - Distributed/data/play_data/'+entry.name, 'wb') as f:  
            #with open('./data/play_data/'+entry.name, 'wb') as f:  
                f.write(res.content)
        filenames = get_game_data_filenames(self.config.resource)
        
        updated = False
        for filename in filenames:
            if filename in self.loaded_filenames:
                continue
            self.load_data_from_file(filename)
            updated = True

        for filename in (self.loaded_filenames - set(filenames)):
            self.unload_data_of_file(filename)
            updated = True

        if updated:
            print("updating training dataset")
            self.dataset = self.collect_all_loaded_data()

    def load_data_from_file(self, filename):
        try:
            print("loading data from ",filename)
            data = read_game_data_from_file(filename)
            self.loaded_data[filename] = self.convert_to_training_data(data)
            self.loaded_filenames.add(filename)
        except Exception as e:
            print(str(e))

    def unload_data_of_file(self, filename):
        print("removing data about ",filename," from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]

    @staticmethod
    def convert_to_training_data(data):
        state_list = []
        policy_list = []
        z_list = []
        for state, policy, z in data:
            board = list(state)
            board = np.reshape(board, (3, 3))
            env = GameEnv().update(board)

            black_ary, white_ary = env.black_and_white_plane()
            state = [black_ary, white_ary] if env.player_turn() == Player.black else [white_ary, black_ary]

            state_list.append(state)
            policy_list.append(policy)
            z_list.append(z)

        return np.array(state_list), np.array(policy_list), np.array(z_list)
