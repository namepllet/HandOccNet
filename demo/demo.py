import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.vis import save_obj
from utils.mano import MANO
mano = MANO()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_demo.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
img_path = 'input.png'
original_img = load_img(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox
bbox = [340.8, 232.0, 20.7, 20.7] # xmin, ymin, width, height 

bbox = process_bbox(bbox, original_img_width, original_img_height)
img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]

# forward
inputs = {'img': img}
targets = {}
meta_info = {}
with torch.no_grad():
    out = model(inputs, targets, meta_info, 'test')
img = (img[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8) # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
verts_out = out['mesh_coord_cam'][0].cpu().numpy()

# bbox for input hand image
bbox_vis = np.array(bbox, int)
bbox_vis[2:] += bbox_vis[:2]
cvimg = cv2.rectangle(original_img.copy(), bbox_vis[:2], bbox_vis[2:], (255,0,0), 3)
cv2.imwrite('hand_bbox.png', cvimg[:,:,::-1])

## input hand image
cv2.imwrite('hand_image.png', img[:,:,::-1])

# save mesh (obj)
save_obj(verts_out*np.array([1,-1,-1]), mano.face, 'output.obj')
