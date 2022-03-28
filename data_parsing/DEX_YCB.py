import json
import os
import os.path as osp
import numpy as np
import cv2
import copy
import torch
from tqdm import tqdm
import yaml

import sys
sys.path.insert(0, '../common')
sys.path.insert(0, '../main')
from config import cfg
# vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
# render_mesh(img, mesh, face, cam_param):
# vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

from utils.vis import vis_mesh, render_mesh, vis_keypoints_with_skeleton, vis_3d_skeleton
from utils.mano import MANO

mano = MANO()
import sys
sys.path.insert(0, cfg.mano_path)
import manopth
from manopth.manolayer import ManoLayer
manolayer_left = ManoLayer(mano_root=osp.join(cfg.mano_path, 'mano', 'models'), flat_hand_mean=False, use_pca=True, side='left', ncomps=45)
manolayer_right = ManoLayer(mano_root=osp.join(cfg.mano_path, 'mano', 'models'), flat_hand_mean=False, use_pca=True, side='right', ncomps=45)

_SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

_SERIALS = [
    '836212060125',
    '839512060362',
    '840412060917',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]

def parse_data(split, setup='s0'):
    _split = split
    _setup = setup

    _data_dir = "../data/DEX_YCB/data"
    _calib_dir = os.path.join(_data_dir, "calibration")
    
    _color_format = "color_{:06d}.jpg"
    _label_format = "labels_{:06d}.npz"
    _h = 480
    _w = 640

    save_data = {'images':[], 'annotations': []}

    if _setup == 's0':
        if _split == 'train':
            subject_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 != 4]
        if _split == 'val':
            subject_ind = [0, 1]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]
        if _split == 'test':
            subject_ind = [2, 3, 4, 5, 6, 7, 8, 9]
            serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]
            sequence_ind = [i for i in range(100) if i % 5 == 4]

    _subjects = [_SUBJECTS[i] for i in subject_ind]
    _serials = [_SERIALS[i] for i in serial_ind]
    _intrinsics = []

    for s in _serials:
        intr_file = os.path.join(_calib_dir, "intrinsics","{}_{}x{}.yml".format(s, _w, _h))
        with open(intr_file, 'r') as f:
            intr = yaml.load(f, Loader=yaml.FullLoader)
        intr = intr['color']
        _intrinsics.append(intr)

    _sequences = []
    _mapping = []
    _mano_side = []
    _mano_betas = []
    
    offset = 0
    for n in _subjects:
        seq = sorted(os.listdir(os.path.join(_data_dir, n)))
        seq = [os.path.join(n, s) for s in seq]
        assert len(seq) == 100
        seq = [seq[i] for i in sequence_ind]
        _sequences += seq
        for i, q in enumerate(seq):
            meta_file = os.path.join(_data_dir, q, "meta.yml")
            with open(meta_file, 'r') as f:
                meta = yaml.load(f, Loader=yaml.FullLoader)
            
            c = np.arange(len(_serials))
            f = np.arange(meta['num_frames'])
            f, c = np.meshgrid(f, c)
            c = c.ravel()
            f = f.ravel()
            s = (offset + i) * np.ones_like(c)

            m = np.vstack((s, c, f)).T
            _mapping.append(m)
            _mano_side.append(meta['mano_sides'][0])
            mano_calib_file = os.path.join(_data_dir, "calibration",
                                        "mano_{}".format(meta['mano_calib'][0]),
                                        "mano.yml")
            with open(mano_calib_file, 'r') as f:
                mano_calib = yaml.load(f, Loader=yaml.FullLoader)
            _mano_betas.append(mano_calib['betas'])
        offset += len(seq)
    _mapping = np.vstack(_mapping)
    
    for i,m in enumerate(_mapping):
        s,c,f = m
        d = os.path.join(_data_dir, _sequences[s], _serials[c])
        label = np.load(os.path.join(d, _label_format.format(f)))
        if np.all(label['pose_m']==0) and np.all(label['joint_3d']==-1) and np.all(label['joint_2d']==-1):
            continue
        
        img_filename = os.path.join(_sequences[s], _serials[c], _color_format.format(f))
        save_data['images'].append({'id': i, 'file_name': img_filename,
                                    'width': 640, 'height': 480})
        
        joints_img = label['joint_2d'][0]
        joints_coord_cam = label['joint_3d'][0] # meter
        cam_param = {'focal': [float(_intrinsics[c]['fx']), float(_intrinsics[c]['fy'])], 'princpt': [float(_intrinsics[c]['ppx']), float(_intrinsics[c]['ppy'])]}
        
        hand_type = _mano_side[s]
        mano_pose = label['pose_m'][0,:48]
        th_mano_pose = torch.FloatTensor(mano_pose).view(1,-1)
        selected_comps = manolayer_left.th_selected_comps if hand_type == 'left' else manolayer_right.th_selected_comps
        th_mano_pose[:,3:] = th_mano_pose[:,3:].mm(selected_comps)
        mano_pose = th_mano_pose.numpy()[0]
        mano_shape = np.array(_mano_betas[s], dtype=np.float32)
        #mano_trans = label['pose_m'][0,48:]
        mano_param = {"pose": mano_pose, "shape": mano_shape}#, "trans": mano_trans}
        
        save_data['annotations'].append({'id': i, 'image_id': i, 'joints_coord_cam': joints_coord_cam.tolist(), 'joints_img': joints_img.tolist(),
                                        'cam_param': cam_param, 'mano_param': {k:v.tolist() for k,v in mano_param.items()}, "hand_type": hand_type})

        vis=False
        if vis and hand_type=='left':
            img = cv2.imread(osp.join(_data_dir, img_filename))
            img = vis_keypoints_with_skeleton(img, np.concatenate([joints_img,np.ones([21,1])],1).T, mano.skeleton, kp_thresh=-1000)
            cv2.imwrite('joints_img{}.png'.format(i), img)
            import pdb;pdb.set_trace()

        vis=False
        if vis and hand_type=='left':
            img = cv2.imread(osp.join(_data_dir, img_filename))
            cv2.imwrite('img{}.png'.format(i),img)
            vis_3d_skeleton(joints_coord_cam, np.ones([21,1]), mano.skeleton,filename='kpts3d_{}.png'.format(i))
            import pdb;pdb.set_trace()

        vis=False
        if vis and hand_type=='left':
            img = cv2.imread(osp.join(_data_dir, img_filename))
            #cv2.imwrite('img{}.png'.format(i),img)
            if hand_type == 'right':
                verts, joints = mano.layer_dex_right(torch.FloatTensor(mano_param['pose']).view(1,-1), torch.FloatTensor(mano_param['shape']).view(1,-1))#, torch.FloatTensor(mano_param['trans']).view(1,-1))
            else:
                verts, joints = mano.layer_dex_left(torch.FloatTensor(mano_param['pose']).view(1,-1), torch.FloatTensor(mano_param['shape']).view(1,-1))#, torch.FloatTensor(mano_param['trans']).view(1,-1))
            verts = verts[0].numpy()
            verts = verts - joints[0][0].numpy() + joints_coord_cam[0]*1000
            if hand_type == 'right':
                img = render_mesh(img, verts/1000, mano.face, cam_param)
            else:
                img = render_mesh(img, verts/1000, mano.face_left, cam_param)
            cv2.imwrite('my_left{}.png'.format(i),img)

            import pdb;pdb.set_trace()
    
    with open(osp.join(_data_dir, "annotations", "DEX_YCB_{}_{}_data.json".format(setup, split)), "w") as f:
        json.dump(save_data, f)
    import pdb;pdb.set_trace()
        
        
if __name__ == '__main__':
    split_list = ('train','test')
    for split in split_list:
        parse_data(split)
