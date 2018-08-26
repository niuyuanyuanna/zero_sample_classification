from easydict import EasyDict as edict
from config.config_dataset import  *
import time


config = edict()

config.root_path = '/home/nyy/tianchi'
config.exp = '/home/nyy/tianchi/temp'
config.data_path = config.root_path

now_date = time.strftime('%Y-%m-%d', time.localtime(time, time()))
now_time = time.strftime('%H_%M_%S', time.localtime(time.time()))
config.log_path = os.path.join(config.exp, now_date, now_time + '_logs')
config.model_path = os.path.join(config.exp, now_date, now_time + '_models')

# add dataset
train_data_list = ['DatasetA_train']
test_data_list = ['DatasetA_test']
config.dataset = edict()
config.dataset.b_g_r_mean = [109, 122, 130]
config.dataset.b_g_r_std = [76, 72, 74]
config.dataset.input_resolution = [64, 64]
add_dataset_params(config, train_data_list, test_data_list)

# train detail
config.train = edict()
config.train.split_val = 0.2
config.train.aug_strategy = edict()
config.train.aug_strategy.normalize = True
config.train.aug_strategy.flip = True
config.train.aug_strategy.random_rotate = True
config.train.aug_strategy.ramdom_crop = False
config.train.aug_strategy.ramdom_color = False
config.train.max_rotate_angle = 20

# model params
config.support_network = ['ResNet']
config.network = "ResNet#50"
assert config.network.split('#')[0] in config.support_network
config.out_classes = [24, 300]
config.epoch = 60
config.batch_size = 60
config.data_loader_num_workers = 8
config.num_gpu = 1

# optimizer params
config.learn_rate = 0.00125 * config.num_gpu
config.momentum = 0.0
config.weightDecay = 0.0
config.alpha = 0.99
config.epsilon = 1e-8
config.load_model_path = None

config.sample_test = None
config.DEBUG = False