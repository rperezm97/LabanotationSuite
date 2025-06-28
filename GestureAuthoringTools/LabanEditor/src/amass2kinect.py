
import os,math,copy
import numpy as np
import torch
from tqdm import tqdm

import os
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel

from pathlib import Path
from scipy.spatial.transform import Rotation as Rscipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from pytorch3d.transforms import matrix_to_euler_angles

from scipy.spatial.transform import Rotation as Rscipy
import numpy as np
from tqdm import tqdm
import torch
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rscipy


#------------------------------------------------------------------------------
# AMASS Joint Mapping to Kinect Body Format
AMASS_TO_KINECT_MAP = {
    "spineB": 0, "spineM": 3,
    "neck": 12, "head": 15,
    "shoulderL": 16, "elbowL": 18, "wristL": 20, "handL": 25,  # Left arm
    "shoulderR": 17, "elbowR": 19, "wristR": 21, "handR": 41,  # Right arm
    "hipL": 1, "kneeL": 4, "ankleL": 7, "footL": 10,  # Left leg
    "hipR": 2, "kneeR": 5, "ankleR": 8, "footR": 11,  # Right leg
    "spineS": 6,"handTL": 34, "thumbL": 35, "handTR": 49, "thumbR": 50  # Hands
}  
skeleton_connections = [
        ('spineB', 'spineM'), ('spineM', 'spineS'),
        ('spineS', 'neck'), ('neck', 'head'),
        ('spineS', 'shoulderL'), ('shoulderL', 'elbowL'), ('elbowL', 'wristL'), ('wristL', 'handL'),
        ('spineS', 'shoulderR'), ('shoulderR', 'elbowR'), ('elbowR', 'wristR'), ('wristR', 'handR'),
        ('spineB', 'hipL'), ('hipL', 'kneeL'), ('kneeL', 'ankleL'), ('ankleL', 'footL'),
        ('spineB', 'hipR'), ('hipR', 'kneeR'), ('kneeR', 'ankleR'), ('ankleR', 'footR')
    ]

    

# Kinect-compatible data types
jType = np.dtype({'names': ['x', 'y', 'z', 'ts'], 'formats': [float, float, float, int]})

# a body
bType = np.dtype({'names':[ 'timeS',                # milliseconds
                            'filled',               # filled gap
                            'spineB', 'spineM',     # meter
                            'neck', 'head',
                            'shoulderL', 'elbowL', 'wristL', 'handL', # tracked=2, inferred=1, nottracked=0
                            'shoulderR', 'elbowR', 'wristR', 'handR',
                            'hipL', 'kneeL', 'ankleL', 'footL',
                            'hipR', 'kneeR', 'ankleR', 'footR',
                            'spineS', 'handTL', 'thumbL', 'handTR', 'thumbR', 
                            'T', 'R',
                            ],
                'formats':[ int,
                            bool,
                            jType, jType,
                            jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType, jType,
                            (float, 3), (float, 3)]})

# Get the world positions of left and right hips at frame 0
def convert_to_cannon(joint_positions):
     # Swap Y and Z to convert from AMASS to Kinect coordinate system
    joint_positions[:, :, [1, 2]] = joint_positions[:, :, [2, 1]]  # Y ↔ Z

    # Compute canonical frame from first frame's hips
    hipL_0 = joint_positions[0, AMASS_TO_KINECT_MAP["hipL"]]  # (3,)
    hipR_0 = joint_positions[0, AMASS_TO_KINECT_MAP["hipR"]]
    pelvis_0 = joint_positions[0, AMASS_TO_KINECT_MAP["spineB"]]

    side_0 = hipR_0 - hipL_0
    side_0 = side_0 / torch.norm(side_0)
    up_vec = torch.tensor([0, 1, 0], dtype=joint_positions.dtype, device=joint_positions.device)
    fwd_0 = torch.cross(up_vec, side_0)
    fwd_0 = fwd_0 / torch.norm(fwd_0)
    side_0 = torch.cross(fwd_0, up_vec)

    canonical_rotation = torch.stack([side_0, up_vec, fwd_0], dim=1)  # (3, 3)
    canonical_rotation_inv = canonical_rotation.T  # inverse (canonical → global)

    #  Rotate and center all joint positions 
    joint_positions_centered = joint_positions - pelvis_0[None, None, :]  # center to pelvis
    joint_positions_canon = torch.matmul(joint_positions_centered, canonical_rotation_inv)  # (N, 52, 3)

    #  Relative translation (in canonical frame) 
    pelvis = joint_positions[:, AMASS_TO_KINECT_MAP["spineB"]]  # (N, 3)
    relative_trans = torch.matmul(pelvis - pelvis_0, canonical_rotation_inv)  # (N, 3)

    #  Relative rotation (per-frame) using hips 
    hipL = joint_positions[:, AMASS_TO_KINECT_MAP["hipL"]]  # (N, 3)
    hipR = joint_positions[:, AMASS_TO_KINECT_MAP["hipR"]]  # (N, 3)
    side = hipR - hipL
    side = side / side.norm(dim=1, keepdim=True)
    fwd = torch.cross(up_vec.expand_as(side), side, dim=1)
    fwd = fwd / fwd.norm(dim=1, keepdim=True)
    side = torch.cross(fwd, up_vec.expand_as(fwd), dim=1)

    R_frames = torch.stack([side, up_vec.expand_as(side), fwd], dim=-1)  # (N, 3, 3)
    R_frames = R_frames.permute(0, 2, 1)  # each rotation matrix in (N, 3, 3)

    R_canon_inv = canonical_rotation_inv.expand(R_frames.shape[0], -1, -1)  # (N, 3, 3)
    R_rel = torch.matmul(R_frames, R_canon_inv)  # (N, 3, 3)

    #  Convert to Euler angles ---
    relative_rot = matrix_to_euler_angles(R_rel, convention='XYZ')  # (N, 3)

    # --- Final output: convert to numpy ---
    joint_positions_canon_np = joint_positions_canon.cpu().numpy()
    relative_trans_np = relative_trans.cpu().numpy()
    relative_rot_np = relative_rot.cpu().numpy()

    return joint_positions_canon_np, relative_trans_np, relative_rot_np


def loadAMASSData(filePath, fps=120, device='cpu', max_frames=20000):
    """
    Loads AMASS motion data and converts it to Kinect-compatible format using PyTorch optimizations.
    
    Args:
        filePath (str): Path to the AMASS dataset (.pt).
        fps (int): Frames per second (default is 30).
        device (str): Computation device ('cpu' or 'cuda').
        
    Returns:
        List of Kinect-compatible motion frames.
    """
    # ✅ Load dataset
    ds = torch.load(filePath)
    
    # ✅ Load SMPL-H Model
    support_dir = Path(filePath).resolve().parents[1]  # Adjust for correct model path
    model_path = os.path.join(support_dir, 'amass/body_models/smplh/male/model.npz')
    bm = BodyModel(bm_fname=model_path, num_betas=16).to(device)

    # ✅ Convert pose, translation & betas into tensors (efficient batch processing)
    pose_torch = ds['pose'].clone().detach().to(dtype=torch.float32, device=device)     # (N, 156)
    trans_torch = ds['trans'].clone().detach().to(dtype=torch.float32, device=device)   # (N, 3)
    # betas_torch = ds['betas'][:16].clone().detach().to(dtype=torch.float32, device=device).reshape(1, -1)

    num_frames =min(pose_torch.shape[0], max_frames) 
    
    # ✅ Extract root orientation and body pose
    root_orient = pose_torch[:, :3]  # (N, 3)  -> Euler angles (axis-angle)
    pose_body = pose_torch[:, 3:66]  # (N, 63)

    # ✅ Compute global joint positions in batch
    smpl_output = bm(pose_body=pose_body,
                     root_orient=root_orient,
                     trans=trans_torch)

    joint_positions = smpl_output.Jtr
    # trans_np = trans_torch.cpu().numpy()  # (N, 3)
    # root_np = root_orient.cpu().numpy()  # (N, 3)  ✅ Kept in Euler angles

   
    joint_positions_canon,relative_trans, relative_root= convert_to_cannon(joint_positions)

    # ✅ Convert batch data into motion_data format
    motion_data = []
    start_time = 0

    for idx in tqdm(range(num_frames), desc="Processing Frames"):
        tempBody = np.zeros(1, dtype=bType)
        tempBody['filled'] = False

        if idx == 0:
            tempBody['timeS'] = 1
            start_time = 1
        else:
            tempBody['timeS'] = int(start_time + (idx * (1000 / fps)))
        
        # ✅ Store joint positions in Kinect format
        for kinect_joint, amass_idx in AMASS_TO_KINECT_MAP.items():
            
            tempPoint = np.zeros(1, dtype=jType)
            tempPoint['x'], tempPoint['y'], tempPoint['z'] =   joint_positions_canon[idx, amass_idx]    # Corrected Coordinates
            tempPoint['ts'] = 2  # Fully tracked
            
            tempBody[0][kinect_joint] = tempPoint

        tempBody[0]['T'] = relative_trans[idx]  # ✅ Store transformed global translation
        tempBody[0]['R'] = relative_root[idx]   # ✅ Store transformed global rotation (Euler)

        motion_data.append(tempBody)
        

    # ✅ Visualize 5 frames
    # plot_skeleton(motion_data, num_frames=5)

    # print(num_frames, num_frames/120)
    return motion_data,joint_positions_canon[:num_frames], relative_trans[:num_frames], relative_root[:num_frames]


# def apply_transformation(joint_positions, T, R):
#     """
#     Applies global translation (T) and rotation (R) to joint positions.
    
#     Args:
#         joint_positions (dict): Dictionary of joint names and their local 3D positions.
#         T (np.array): Global translation (3,)
#         R (np.array): Global rotation (3,) (Euler angles)

#     Returns:
#         transformed_positions (dict): Dictionary of transformed joint positions.
#     """

#     # Convert Euler angles (R) to rotation matrix
   

#     transformed_positions = {}
#     for joint, pos in joint_positions.items():
#         pos = np.array(pos).reshape(3, 1)  # Ensure it's column vector
#         rotated_pos = R @ pos  # Apply rotation
#         transformed_positions[joint] = (rotated_pos.flatten() ).tolist()  # Apply translation

#     return transformed_positions

# def plot_skeleton(frames, num_frames=4, show_centered=True):
#     """
#     Visualizes multiple skeleton frames, both transformed and optionally centered.
#     """
#     fig = plt.figure(figsize=(15, 10 if show_centered else 5))

#     for i in range(num_frames):
#         idx = 100 * i
#         frame = frames[idx]

#         T = frame['T'][0]
#         R_euler = frame['R'][0]
#         R_matrix = Rscipy.from_euler('xyz', R_euler, degrees=False).as_matrix()

#         joint_positions = {
#             joint: np.array([frame[joint]['x'][0], frame[joint]['y'][0], frame[joint]['z'][0]])
#             for joint in frame.dtype.names[2:-2]
#         }

#         # --- Transformed (Global Space) ---
#         # transformed = apply_transformation(joint_positions, T, R_matrix)

#         ax1 = fig.add_subplot(2 if show_centered else 1, num_frames, i + 1, projection='3d')
#         plot_single_skeleton(ax1, joint_positions, title=f"Global Frame {idx}")

#         # --- Centered (Canonical Space) ---
#         if show_centered:
#             centered = {
#                 joint: R_matrix.T @ np.array(pos) - T
#                 for joint, pos in joint_positions.items()
#             }

#             ax2 = fig.add_subplot(2, num_frames, num_frames + i + 1, projection='3d')
#             plot_single_skeleton(ax2, centered, title=f"Centered Frame {idx}")
#     # plt.show()
    
# def plot_single_skeleton(ax, joint_dict, title="Skeleton"):
  
#     for joint_start, joint_end in skeleton_connections:
#         if joint_start in joint_dict and joint_end in joint_dict:
#             start = joint_dict[joint_start]
#             end = joint_dict[joint_end]
#             ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'bo-')

#     ax.set_title(title)
#     # ax.set_xlim([-1, 1])
#     # ax.set_ylim([0, 2])
#     # ax.set_zlim([0, 2])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.view_init(elev=90, azim=-90)

