import os, time
from random import random
from datetime import datetime
import dropbox

import keras.backend as K
import numpy as np
from keras.optimizers import SGD

from agent.model import GameModel, objective_function_for_policy, objective_function_for_value
from agent.player import GamePlayer

from config import Config
from src.lib import tf_util
from src.lib.data_helper import get_game_data_filenames, write_game_data_to_file, read_game_data_from_file, get_next_generation_model_dirs
from src.lib.model_helpler import save_as_best_model, load_best_model_weight
from env.game_env import GameEnv, Player, Winner

def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.59)
    return EvolverWorker(config).start()

class EvolverWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: GameModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.optimizer = None
        self.dbx = None
        self.version = 0 # Change to dynamic lookup from Drop Box Files
        self.env = GameEnv()
        self.raw_timestamp=None
        self.best_is_white = True
        self.play_files_per_generation = 15 # each file includes 25 games so each generation adds 375 games
        self.max_play_files = 300
        
        # at final there are alawys 7500 games to look at from previous 20 generations
        self.min_play_files_to_learn = 0
        self.play_files_on_dropbox = 0
    def start(self):
        auth_token = 'UlBTypwXWYAAAAAAAAAAEP6hKysZi9cQKGZTmMu128TYEEig00w3b3mJ--b_6phN'
        self.dbx = dropbox.Dropbox(auth_token)
        
        self.version = len(self.dbx.files_list_folder('/model/HistoryVersion').entries)
        print('\nThe Strongest Version found is: ',self.version,'\n')
        
        while True:
            self.model = self.load_model()
            self.compile_model()
            
            self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
            self.min_play_files_to_learn = min((self.version + 1) * self.play_files_per_generation, self.max_play_files)   
            while self.play_files_on_dropbox < self.min_play_files_to_learn:
                print('\nPlay Files Found:',self.play_files_on_dropbox,'of required',self.min_play_files_to_learn,'files. Started Self-Playing...\n')
                self.self_play()
                self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
            print('\nPlay Files Found:',self.play_files_on_dropbox,'of required',self.min_play_files_to_learn,'files. Training files sufficient for Learning!\n')
            self.load_play_data()
            self.raw_timestamp=self.dbx.files_get_metadata('/model/model_best_weight.h5').client_modified
        
            RetrainSuccessful = False
            while(RetrainSuccessful == False):
                # Training
                self.training()
                # Evaluating
                self.best_model = self.load_best_model()
                RetrainSuccessful = self.evaluate()
                if(self.raw_timestamp!=self.dbx.files_get_metadata('/model/model_best_weight.h5').client_modified):
                    # Other Evolvers in Distribution already got a successful competition - cease this current eval.
                    time.sleep(20)
                    #self.remove_all_play_data()
                    self.version = len(self.dbx.files_list_folder('/model/HistoryVersion').entries)
                    print('\nThe Strongest Version found is: ',self.version,'\n')
                    
                    # Also remove the oldest 15 files from dropbox
                    localfiles = get_game_data_filenames(self.config.resource)
                    localfilenames = []
                    for a in range(len(localfiles)):
                        localfilenames.append(localfiles[a][-32:])
                    dbfiles = []
                    for entry in self.dbx.files_list_folder('/play_data').entries:
                        dbfiles.append(entry.name)
                    localfiles_to_remove = set(localfilenames) - set(dbfiles)
                    print('Removing',len(localfiles_to_remove),'files from local drive')
                    for file in localfiles_to_remove:
                        print('Removing local play_data file',file)
                        path = os.path.join(self.config.resource.play_data_dir,file)
                        os.remove(path)
                    break
            self.dataset = None
                
    def self_play(self):
        self.buffer = []
        idx = 1

        for _ in range(self.config.play_data.nb_game_in_file):
            self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
            self.min_play_files_to_learn = min((self.version + 1) * self.play_files_per_generation, self.max_play_files) 
            if(self.play_files_on_dropbox >= self.min_play_files_to_learn):
                print('Training files sufficient for Learning, ending Self-Play...')
                break
            start_time = time.time()            
            env = self.self_play_game(idx)
            end_time = time.time()
            print("Game",idx," Time=",(end_time - start_time)," sec, Turn=", env.turn, env.observation, env.winner)
            idx += 1

    def load_model(self):            
        for entry in self.dbx.files_list_folder('/model').entries:
            if(entry.name!='HistoryVersion' and entry.name!='next_generation'):
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

    def compile_model(self):
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def training(self):
        #self.compile_model()
        last_load_data_step = last_save_step = total_steps = self.config.trainer.start_total_steps
        #self.update_learning_rate(total_steps)
        steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
        total_steps += steps
        self.save_current_model()
        last_save_step = total_steps

    def load_play_data(self):
        for entry in self.dbx.files_list_folder('/play_data').entries:
            md, res = self.dbx.files_download('/play_data/'+entry.name)
            with open('SantoriniAZ-Dist/SmallSanto - Distributed/data/play_data/'+entry.name, 'wb') as f:  
            #with open('./data/play_data/'+entry.name, 'wb') as f:  
                f.write(res.content)
        filenames = get_game_data_filenames(self.config.resource)
        
        updated = False
        for filename in filenames:
            #if filename in self.loaded_filenames:
            #    continue
            if filename not in self.loaded_filenames:
                self.load_data_from_file(filename)
            updated = True

        for filename in (self.loaded_filenames - set(filenames)):
            self.unload_data_of_file(filename)

        if updated:
            print("Updated Play Data.\n")
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

    @staticmethod
    def convert_to_training_data(data):
        state_list = []
        policy_list = []
        z_list = []
        for state, policy, z in data:
            board = list(state)
            board = np.reshape(board, (3, 3))
            env = GameEnv().update(board)

            white_ary, black_ary, block_ary, turn_ary = env.black_and_white_plane()
            #state = [black_ary, white_ary, block_ary, turn_ary] if env.player_turn() == Player.black else [white_ary, black_ary, block_ary, turn_ary]
            state = [white_ary, black_ary, block_ary, turn_ary]

            state_list.append(state)
            policy_list.append(policy)
            z_list.append(z)

        return np.array(state_list), np.array(policy_list), np.array(z_list)

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, z_ary = self.dataset
        self.model.model.fit(state_ary, [policy_ary, z_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs, verbose=1)
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def load_best_model(self):
        model = GameModel(self.config)
        load_best_model_weight(model)
        return model

    def evaluate(self):
        ng_model, model_dir = self.load_next_generation_model()
        print("start evaluate model", model_dir)
        ng_is_great = self.evaluate_model(ng_model)
        if ng_is_great:
            print("New Model become best model:", model_dir)
            save_as_best_model(ng_model)
            self.best_model = ng_model

            # Save to Drop Box inside History Version folder & save as best model in /model folder
            self.version = self.version+1
            with open('SantoriniAZ-Dist/SmallSanto - Distributed/data/model/model_best_weight.h5', 'rb') as f:
            #with open('./data/model/model_best_weight.h5', 'rb') as f:
                data = f.read()
            res = self.dbx.files_upload(data, '/model/HistoryVersion/Version'+"{0:0>4}".format(self.version) + '.h5', dropbox.files.WriteMode.add, mute=True)
            res = self.dbx.files_upload(data, '/model/model_best_weight.h5', dropbox.files.WriteMode.overwrite, mute=True)

            # Remove the oldest 15 files if files is already 300
            list = []
            for entry in self.dbx.files_list_folder('/play_data').entries:
                list.append(entry)
            if(len(list)==300):
                for i in range(14,-1,-1): #Remove the oldest 15 files in both DropBox and Local
                    print('Removing Dropbox play_data file',i,list[i].name)
                    self.dbx.files_delete('/play_data/'+list[i].name)
                  
                    print('Removing local play_data file',list[i].name)
                    path = os.path.join(self.config.resource.play_data_dir,list[i].name)
                    os.remove(path)
        self.remove_model(model_dir)
        return ng_is_great

    def load_next_generation_model(self):
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
            print("There is no next generation model to evaluate")
            sleep(600)
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = GameModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

    def evaluate_model(self, ng_model):
        results = []
        winning_rate = 0
        for game_idx in range(1,self.config.eval.game_num+1):
            ng_win, white_is_best = self.play_game(self.best_model, ng_model)
            if ng_win is not None:
                results.append(ng_win)
                winning_rate = sum(results) / len(results)
            if(ng_win==1 and white_is_best):
                print('Game', game_idx,': Wins with Black.  Winning rate ',winning_rate)
            elif(ng_win==1 and not white_is_best):
                print('Game', game_idx,': Wins with White.  Winning rate ',winning_rate)
            elif(ng_win==0 and white_is_best):
                print('Game', game_idx,': Loses with Black.  Winning rate ',winning_rate)
            elif(ng_win==0 and not white_is_best):
                print('Game', game_idx,': Loses with White.  Winning rate ',winning_rate)
            if results.count(0) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                print("Lose count reach", results.count(0)," so give up challenge\n")
                break
            if results.count(1) >= self.config.eval.game_num * self.config.eval.replace_rate:
                print("Win count reach", results.count(1)," so change best model\n")
                break

        winning_rate = sum(results) / len(results)
        return winning_rate >= self.config.eval.replace_rate

    def play_game(self, best_model, ng_model):
        env = GameEnv().reset()

        if(self.raw_timestamp!=self.dbx.files_get_metadata('/model/model_best_weight.h5').client_modified):
            print('A newer model version is available - giving up this match')
            ng_win = 0
            self.best_is_white= True
            return ng_win, self.best_is_white
    
        best_player = GamePlayer(self.config, best_model, play_config=self.config.eval.play_config)
        ng_player = GamePlayer(self.config, ng_model, play_config=self.config.eval.play_config)
        self.best_is_white = not self.best_is_white
        if not self.best_is_white:
            black, white = best_player, ng_player
        else:
            black, white = ng_player, best_player

        env.reset()
        while not env.done:
            if env.player_turn() == Player.black:
                action = black.action(env.board)
            else:
                action = white.action(env.board)
            env.step(action)

        ng_win = None
        if env.winner == Winner.white:
            if self.best_is_white:
                ng_win = 0
            else:
                ng_win = 1
        elif env.winner == Winner.black:
            if self.best_is_white:
                ng_win = 1
            else:
                ng_win = 0
        return ng_win, self.best_is_white

    def remove_model(self, model_dir):
        rc = self.config.resource
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        os.remove(config_path)
        os.remove(weight_path)
        os.rmdir(model_dir)

    def self_play_game(self, idx):
        self.env.reset()
        self.black = GamePlayer(self.config, self.model)
        self.white = GamePlayer(self.config, self.model)
        while not self.env.done:
            if self.env.player_turn() == Player.black:
                action = self.black.action(self.env.board)
            else:
                action = self.white.action(self.env.board)
            self.env.step(action)
            
        self.finish_game()
        self.save_play_data(write=idx % self.config.play_data.nb_game_in_file == 0)
        self.remove_play_data()
        return self.env

    def finish_game(self):
        if self.env.winner == Winner.black:
            black_win = 1
        elif self.env.winner == Winner.white:
            black_win = -1
        else:
            black_win = 0
        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

    def save_play_data(self, write=True):
        data = self.black.moves + self.white.moves
        self.buffer += data

        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        filename = rc.play_data_filename_tmpl % game_id
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        print("save play data to ",path)
        write_game_data_to_file(path, self.buffer)
        
        # Saving File to Drop Box
        self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
        self.min_play_files_to_learn = min((self.version + 1) * self.play_files_per_generation, 300)  
        if self.play_files_on_dropbox < self.min_play_files_to_learn:            
            with open(path, 'rb') as f:
                data = f.read()
            res = self.dbx.files_upload(data, '/play_data/'+filename, dropbox.files.WriteMode.add, mute=True)
        #print('uploaded as', res.name.encode('utf8'))

        self.buffer = []

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])

    def remove_all_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        for i in range(len(files)):
            os.remove(files[i])
