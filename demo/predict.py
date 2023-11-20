""" Generates 3D model of hand given image with possible occlusions"""
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import argparse
import os
import os.path as osp
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from cog import BaseModel, BasePredictor, Input, Path
from torch.nn.parallel.data_parallel import DataParallel

sys.path.insert(0, osp.join("main"))
sys.path.insert(0, osp.join("common"))

from config import cfg
from model import get_model
from utils.mano import MANO
from utils.preprocessing import generate_patch_image, load_img, process_bbox
from utils.vis import save_obj


class Output(BaseModel):
    bbox_img: Path
    obj_model: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.mano = MANO()

        # argument parsing
        gpu = "0"
        cfg.set_args(gpu)
        cudnn.benchmark = True

        # snapshot load
        model_path = "./demo/snapshot_demo.pth.tar"
        assert osp.exists(model_path), "Cannot find model at " + model_path
        print("Load checkpoint from {}".format(model_path))

        # get model
        self.model = get_model("test")
        self.model = DataParallel(self.model).cuda()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt["network"], strict=False)
        self.model.eval()

        # prepare input image
        self.transform = transforms.ToTensor()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        bbox_coords: str = Input(
            description="Input comma-separated bounding box coordinates of hand (xmin,ymin,width,height)"
        )
    ) -> Output:
        img_path = str(image)

        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
        # prepare bbox

        print("Preprocessing bounding boxes.......")
        bbox = bbox_coords.split(",")
        bbox = [float(i) for i in bbox] # xmin, ymin, width, height

        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(
            original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
        )
        img = self.transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]

        # forward
        print("Running model inference.......")
        inputs = {"img": img}
        targets = {}
        meta_info = {}
        with torch.no_grad():
            out = self.model(inputs, targets, meta_info, "test")
        img = (img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(
            np.uint8
        )  # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
        verts_out = out["mesh_coord_cam"][0].cpu().numpy()

        # bbox for input hand image
        bbox_vis = np.array(bbox, int)
        bbox_vis[2:] += bbox_vis[:2]
        cvimg = cv2.rectangle(
            original_img.copy(), bbox_vis[:2], bbox_vis[2:], (255, 0, 0), 3
        )

        print("Generating outputs.......")
        # save hand image with bbox
        bbox_path = Path(tempfile.mkdtemp()) / "hand_bbox.png"
        cv2.imwrite(str(bbox_path), cvimg[:, :, ::-1])

        # save mesh
        zip_path = Path(tempfile.mkdtemp()) / "hand_model_3d.zip"
        obj_path = 'hand_model_3d.obj'
        save_obj(verts_out * np.array([1, -1, -1]), self.mano.face, obj_path)

        print('Zipping .obj file......')
        with zipfile.ZipFile(str(zip_path), "w") as zip_obj:
            zip_obj.write(obj_path)

        return Output(bbox_img=bbox_path, obj_model=zip_path)
