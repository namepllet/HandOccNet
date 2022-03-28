import json
import os
import os.path as osp
import numpy as np
import cv2
import copy
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, '../common')
sys.path.insert(0, '../main')
# vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
# render_mesh(img, mesh, face, cam_param):
# vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

from utils.vis import vis_mesh, render_mesh, vis_keypoints_with_skeleton
from utils.mano import MANO

mano = MANO()

def load_objects_HO3D(obj_root):
    import trimesh
    object_names = ['011_banana', '021_bleach_cleanser', '003_cracker_box', '035_power_drill', '025_mug',
                    '006_mustard_bottle', '019_pitcher_base', '010_potted_meat_can', '037_scissors', '004_sugar_box']
    all_models = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, 'points.xyz')
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = np.array(mesh.vertices)
    return all_models

def get_bbox21_3d_from_dict(vertex):
    bbox21_3d = {}
    for key in vertex:
        vp = vertex[key][:]
        x = vp[:, 0].reshape((1, -1))
        y = vp[:, 1].reshape((1, -1))
        z = vp[:, 2].reshape((1, -1))
        x_max, x_min, y_max, y_min, z_max, z_min = np.max(x), np.min(x), np.max(y), np.min(y), np.max(z), np.min(z)
        p_blb = np.array([x_min, y_min, z_min])  # bottem, left, behind
        p_brb = np.array([x_max, y_min, z_min])  # bottem, right, behind
        p_blf = np.array([x_min, y_max, z_min])  # bottem, left, front
        p_brf = np.array([x_max, y_max, z_min])  # bottem, right, front
        #
        p_tlb = np.array([x_min, y_min, z_max])  # top, left, behind
        p_trb = np.array([x_max, y_min, z_max])  # top, right, behind
        p_tlf = np.array([x_min, y_max, z_max])  # top, left, front
        p_trf = np.array([x_max, y_max, z_max])  # top, right, front
        #
        p_center = (p_tlb + p_brf) / 2
        #
        p_ble = (p_blb + p_blf) / 2  # bottem, left, edge center
        p_bre = (p_brb + p_brf) / 2  # bottem, right, edge center
        p_bfe = (p_blf + p_brf) / 2  # bottem, front, edge center
        p_bbe = (p_blb + p_brb) / 2  # bottem, behind, edge center
        #
        p_tle = (p_tlb + p_tlf) / 2  # top, left, edge center
        p_tre = (p_trb + p_trf) / 2  # top, right, edge center
        p_tfe = (p_tlf + p_trf) / 2  # top, front, edge center
        p_tbe = (p_tlb + p_trb) / 2  # top, behind, edge center
        #
        p_lfe = (p_tlf + p_blf) / 2  # left, front, edge center
        p_lbe = (p_tlb + p_blb) / 2  # left, behind, edge center
        p_rfe = (p_trf + p_brf) / 2  # left, front, edge center
        p_rbe = (p_trb + p_brb) / 2  # left, behind, edge center

        pts = np.stack((p_blb, p_brb, p_blf, p_brf,
                        p_tlb, p_trb, p_tlf, p_trf,
                        p_ble, p_bre, p_bfe, p_bbe,
                        p_tle, p_tre, p_tfe, p_tbe,
                        p_lfe, p_lbe, p_rfe, p_rbe,
                        p_center))
        bbox21_3d[key] = pts
    return bbox21_3d

def pose_from_RT(R, T):
    pose = np.zeros((4, 4))
    pose[:3, 3] = T
    pose[3, 3] = 1
    R33, _ = cv2.Rodrigues(R)
    pose[:3, :3] = R33
    # change from OpenGL coord to normal coord
    pose[1, :] = -pose[1, :]
    pose[2, :] = -pose[2, :]
    return pose


def projectPoints(xyz, K, rt=None):
    xyz = np.array(xyz)
    K = np.array(K)
    if rt is not None:
        uv = np.matmul(K, np.matmul(rt[:3, :3], xyz.T) + rt[:3, 3].reshape(-1, 1)).T
    else:
        uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def parse_data(split):
    root_dir = "../data/HO3D/data"
    save_data = {'images':[], 'annotations': []}
    jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                      1, 2, 3, 17,
                                      4, 5, 6, 18,
                                      10, 11, 12, 19,
                                      7, 8, 9, 20]
    jointsMapSimpleToMano = np.argsort(jointsMapManoToSimple)
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

    with open(osp.join(root_dir, "{}.txt".format(split))) as f:
        set_list = [line.strip() for line in f]

    ## obj
    # object informations
    obj_model_root = '/home/namepllet/Codes/HO3D_joon/assets/object_models'
    obj_mesh = load_objects_HO3D(obj_model_root)
    obj_bbox3d = get_bbox21_3d_from_dict(obj_mesh)

    for idx in tqdm(range(len(set_list))):    
        seqName, id = set_list[idx].split("/")
        annotations = np.load(osp.join(root_dir, split, seqName, 'meta', id + '.pkl'),allow_pickle=True)

        ## images
        save_data['images'].append({'id': idx, 'file_name': osp.join(seqName, 'rgb', id+'.png'),
                'width': 640, 'height': 480})
        
        ## annotations
        if split == 'train':
            joints_coord_cam = (annotations['handJoints3D'] * np.array([1, -1, -1]))[jointsMapManoToSimple].tolist()
            camMat = annotations['camMat']
            cam_param = {'focal': [float(camMat[0,0]), float(camMat[1,1])], 'princpt': [float(camMat[0,2]), float(camMat[1,2])]}

            mano_param = {}

            mano_pose = annotations['handPose']
            root_pose = copy.deepcopy(mano_pose[:3])
            per_rdg, _ = cv2.Rodrigues(root_pose)
            resrot, _ = cv2.Rodrigues(np.dot(coord_change_mat, per_rdg))
            mano_pose[:3] = resrot.ravel()
            mano_pose[3:] -= mano.layer.th_hands_mean[0].numpy()

            mano_param['pose'] = mano_pose.tolist()
            mano_param['shape'] = annotations['handBeta'].tolist()
            #mano_param['trans'] = (annotations['handTrans'] * np.array([1, -1, -1])).tolist()

            save_data['annotations'].append({'id': idx, 'image_id': idx, 'joints_coord_cam': joints_coord_cam, 'cam_param': cam_param, 'mano_param': mano_param})

            ## degug
            vis = False
            if vis:
                # joints
                img = cv2.imread(osp.join(root_dir, 'train', osp.join(seqName, 'rgb', id+'.png')))
                from utils.transforms import cam2pixel
                joints_coord_img = cam2pixel(np.array(joints_coord_cam), cam_param['focal'], cam_param['princpt'])
                img = vis_keypoints_with_skeleton(img, joints_coord_img.T, mano.skeleton, kp_thresh=-1000, alpha=1)
                cv2.imwrite('joints.png', img)
                """
                # mesh
                img = cv2.imread(osp.join(root_dir, 'train', osp.join(seqName, 'rgb', id+'.png')))
                verts, joints = mano.layer(torch.FloatTensor(mano_param['pose']).view(1,-1), torch.FloatTensor(mano_param['shape']).view(1,-1), torch.FloatTensor(mano_param['trans']).view(1,-1))
                verts = verts[0].numpy()
                
                #verts += np.array(joints_coord_cam[0])*1000

                img_mesh = render_mesh(img.copy(), verts/1000, mano.face, cam_param)
                cv2.imwrite('test.png', img_mesh)

                verts -= joints.numpy()[0][0]
                verts += np.array(joints_coord_cam[0])*1000
                img_mesh = render_mesh(img.copy(), verts/1000, mano.face, cam_param)
                cv2.imwrite('test2.png', img_mesh)
                """
        else:
            root_joint_cam = (annotations['handJoints3D'] * np.array([1, -1, -1])).tolist()
            bbox = np.array(annotations['handBoundingBox'], dtype=np.float32) #xyxy
            bbox[2:] -= bbox[:2]
            camMat = annotations['camMat']
            cam_param = {'focal': [float(camMat[0,0]), float(camMat[1,1])], 'princpt': [float(camMat[0,2]), float(camMat[1,2])]}

            ## obj
            sample = {}
            sample["obj_cls"] = annotations['objName']
            sample["obj_bbox3d"] = obj_bbox3d[sample["obj_cls"]]

            obj_pose = pose_from_RT(annotations['objRot'].reshape((3,)), annotations['objTrans'])
            K = camMat
            p2d = projectPoints(sample["obj_bbox3d"], K, rt=obj_pose)

            vis=False
            if vis:
                img = cv2.imread(osp.join(root_dir, split, save_data['images'][-1]['file_name']))
                for _p2d in p2d:
                    cv2.circle(img, _p2d.astype(int), 3, (0,0,255), -1)
                cv2.imwrite('test.png', img)
                import pdb;pdb.set_trace()

            save_data['annotations'].append({'id': idx, 'image_id': idx, 'root_joint_cam': root_joint_cam, 'cam_param': cam_param, 'bbox': bbox.tolist(), 'p2d': p2d.tolist()})
    with open(osp.join(root_dir, "annotations", "HO3D_{}_data_obj.json".format(split)), "w") as f:
        json.dump(save_data, f)

if __name__ == '__main__':
    split_list = ('evaluation',)
    for split in split_list:
        parse_data(split)
