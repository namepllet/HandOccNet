import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    # HO3D, DEX_YCB
    trainset = 'HO3D'
    testset = 'DEX_YCB'
    
    ## input, output
    input_img_shape = (256,256) 
    
    ## training config
    if trainset == 'HO3D':
        lr_dec_epoch = [10*i for i in range(1,7)]
        end_epoch = 70
        lr = 1e-4
        lr_dec_factor = 0.7
    elif trainset == 'DEX_YCB':
        lr_dec_epoch = [i for i in range(1,25)]
        end_epoch = 25
        lr = 1e-4
        lr_dec_factor = 0.9
    train_batch_size = 16 # per GPU
    lambda_mano_verts = 1e4
    lambda_mano_joints = 1e4
    lambda_mano_pose = 10
    lambda_mano_shape = 0.1
    lambda_joints_img = 100
    ckpt_freq = 10

    ## testing config
    test_batch_size = 64

    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
