import numpy as np
import torch
import torch.nn as nn



def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype=dtype)


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])

class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=300, ftol=1e-10, gtol=1e-09,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='mano',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0

        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        pass
    
    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized

            Returns
            -------
                loss: float
                The final loss value
        '''
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break
            
            prev_loss = loss.item()
        
        return prev_loss

    def create_fitting_closure(self,
                               optimizer, 
                               camera=None,
                               joint_cam=None, 
                               joint_img=None,
                               hand_translation=None,
                               hand_scale=None,
                               loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               create_graph=False,
                               **kwargs):

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            total_loss = loss(camera=camera,
                              joint_cam=joint_cam,
                              joint_img=joint_img,
                              hand_translation=hand_translation,
                              hand_scale =hand_scale,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1

            return total_loss

        return fitting_func




class ScaleTranslationLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e3, dtype=torch.float32,
                 **kwargs):
        super(ScaleTranslationLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, camera, joint_cam, joint_img, hand_translation, hand_scale, **kwargs):

        projected_joints = camera(
            hand_scale * joint_cam + hand_translation)
        
        joint_error = \
            torch.index_select(joint_img, 1, self.init_joints_idxs) - \
            torch.index_select(projected_joints, 1, self.init_joints_idxs)
        joint_loss = torch.sum(joint_error.abs()) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not None):
            depth_loss = self.depth_loss_weight * torch.sum((
                hand_translation[2] - self.trans_estimation[2]).abs() ** 2)

        return joint_loss + depth_loss