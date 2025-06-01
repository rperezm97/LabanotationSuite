import numpy as np
from scipy.ndimage import gaussian_filter1d

# Mapping from AMASS joint names to Kinect joint indices
AMASS_TO_KINECT_MAP = {
    "spineB": 0, "spineM": 3, "spineS": 6, "neck": 12, "head": 15,
    "shoulderL": 16, "elbowL": 18, "wristL": 20, "handL": 25,
    "shoulderR": 17, "elbowR": 19, "wristR": 21, "handR": 41,
    "hipL": 1, "kneeL": 4, "ankleL": 7, "footL": 10,
    "hipR": 2, "kneeR": 5, "ankleR": 8, "footR": 11,
    "handTL": 34, "thumbL": 35, "handTR": 49, "thumbR": 50
}

# Parent-child relationships for computing local spherical coordinates
PARENT_RELATIONS = {
    "elbowR": "shoulderR",
    "elbowL": "shoulderL",
    "wristR": "elbowR",
    "wristL": "elbowL",
    "kneeR": "hipR",
    "kneeL": "hipL",
    "ankleR": "kneeR",
    "ankleL": "kneeL",
    "footR": "ankleR",
    "footL": "ankleL",
    "head": "neck",
    "spineS": "spineM",
    "shoulderR": "spineS",
    "shoulderL": "spineS"
}
STAFF_ORDER = [
    "elbowL", "wristL",
    "spineS",#"shoulderL",
    "ankleL", "footL", "kneeL",
    "kneeR", "footR", "ankleR",
    "spineS",#"shoulderR",
    "wristR", "elbowR",
    "head"
]
# -------------------------------------------------------------
def smooth_positions(joint_positions, sigma, window_size):
    """
    Apply Gaussian smoothing along time for each joint.
    """
    truncate = (window_size - 1) / (2 * sigma)
    return gaussian_filter1d(
        joint_positions,
        sigma=sigma,
        truncate=truncate,
        axis=0,
        mode='reflect'
    )

# -------------------------------------------------------------
def compute_lma_indices(jp, keyframes, times):
    """
    Compute global LMA Effort descriptors between keyframes using all joints and timestamps.
    Returns a list of dicts with keys 'weight', 'time', 'space', 'flow'.
    """
    T_total, J, _ = jp.shape
    frames =  [kf for kf in keyframes]
    segs = []
    eps = 1e-6

    for i in range(len(frames) - 1):
        s, e = frames[i], frames[i+1]
        seg = jp[s:e+1]
        t_seg = times[s:e+1]
        M = seg.shape[0]
        if M < 5:
            segs.append({'weight': 0, 'time': 0, 'space': 0, 'flow': 0})
            continue

        # Velocity and norm
        dt1 = t_seg[1:] - t_seg[:-1]
        vel = (seg[1:] - seg[:-1]) / dt1[:, None, None]
        vn = np.linalg.norm(vel, axis=2)

        # Weight: peak kinetic energy (sum of v^2)
        E = np.sum(vn**2, axis=1)
        W = np.max(E)

        # Acceleration and norm
        dt2 = t_seg[2:] - t_seg[1:-1]
        acc = (vel[1:] - vel[:-1]) / dt2[:, None, None]
        an = np.linalg.norm(acc, axis=2)
        T_eff = np.sum(an) / (an.size if an.size > 0 else eps)

        # Space: path length / net displacement per joint
        disp = np.linalg.norm(seg[1:] - seg[:-1], axis=2)
        net = np.linalg.norm(seg[-1] - seg[0], axis=1)
        S = 0
        for j in range(J):
            path = np.sum(disp[:, j])
            denom = net[j] if net[j] > eps else path
            S += path / denom

        # Flow: jerk norm
        if M >= 4:
            dt3 = t_seg[3:] - t_seg[2:-1]
            jerk = (acc[1:] - acc[:-1]) / dt3[:, None, None]
            jn = np.linalg.norm(jerk, axis=2)
            F = np.sum(jn) / (jn.size if jn.size > 0 else eps)
        else:
            F = 0

        segs.append({'weight': W, 'time': T_eff, 'space': S, 'flow': F})
    return segs

# -------------------------------------------------------------
def detect_ground_contacts(data_fps, sm_large):
    """
    Detect ground-contact keyframes based on peak vertical ankle acceleration
    and compute ground height from smoothed ankle positions.
    Returns (ground_keyframes, ground_heights).
    """
    dt = 1.0 / data_fps
    t_max = sm_large.shape[0]

    def vert_acc(sm, joint):
        idx = AMASS_TO_KINECT_MAP[joint]
        return (sm[2:, idx, 1] - 2*sm[1:-1, idx, 1] + sm[:-2, idx, 1]) / (dt*dt)

    acc_L = vert_acc(sm_large, 'footL')
    acc_R = vert_acc(sm_large, 'footR')
 
    peaks_L = np.where((acc_L[:-1] < 0) & (acc_L[1:] >= 0))[0] + 1
    peaks_R = np.where((acc_R[:-1] < 0) & (acc_R[1:] >= 0))[0] + 1
    
    ground_keyframes = [0]
    
    margin = 5
    for l in peaks_L:
        for r in peaks_R:
            if abs(int(l) - int(r)) <= margin:
                ground_keyframes.append(int(round((l + r) / 2)))
    ground_keyframes = sorted(set(ground_keyframes))
    
    ground_heights = np.min([
        sm_large[:, AMASS_TO_KINECT_MAP['footL'], 1],
        sm_large[:, AMASS_TO_KINECT_MAP['footR'], 1]], axis=0
    )
    return ground_keyframes, ground_heights

def norm1d(v: np.ndarray) -> np.ndarray:
    """Return a unit‐length version of v (shape (3,)), or zero if ‖v‖=0."""
    mag = np.linalg.norm(v)
    return (v / mag) if mag > 1e-8 else np.zeros(3)

def calculate_base_rotation(joint_positions: np.ndarray) -> np.ndarray:
    """
    Given a single frame’s joint positions in a NumPy array of shape (N_joints, 3),
    compute a 3×3 “body‐aligned” rotation matrix exactly as before—BUT indexing
    into joint_positions via AMASS_TO_KINECT_MAP instead of a dict.

    Inputs:
      - joint_positions: np.ndarray of shape (N_joints, 3).
        E.g. joint_positions[AMASS_TO_KINECT_MAP['shoulderL'], :] == [x_L, y_L, z_L].

    Returns:
      - base_rotation: a 3×3 orthonormal matrix R such that
          R @ (world‐coords) = “body‐aligned” coords.
    """
    # 1) Pull out the three key points:
    shL_idx = AMASS_TO_KINECT_MAP["shoulderL"]
    shR_idx = AMASS_TO_KINECT_MAP["shoulderR"]
    spM_idx = AMASS_TO_KINECT_MAP["spineM"]

    shL = joint_positions[shL_idx]  # shape (3,)[-0.23155918717384338, 0.43370020389556885, 0.018255101516842842]

    shR = joint_positions[shR_idx]  # shape (3,)[0.17075686156749725, 0.44278275966644287, 0.030035967007279396]

    spM = joint_positions[spM_idx]  # shape (3,)

    # 2) Recreate the same logic you had before:
    #    - v1 = vector from shoulderR → shoulderL  (your “y‐axis” before normalization)
    #    - v2 = [0, -1, 0]  (you’d earlier commented out spM–shR, but you chose [0,-1,0] instead)
    v1 = shL - shR                  # points “from Right shoulder toward Left shoulder”[-0.40231604874134064, -0.009082555770874023, -0.011780865490436554]

    v2 = np.array([0.0, -1.0, 0.0]) # a fixed direction (as you had in your older code)

    # 3) Build the three (unnormalized) body‐axes:
    #    - x‐axis is perpendicular to the plane of (v2, v1)
    #    - y‐axis is exactly v1
    #    - z‐axis = x × y
    x_axis_unnorm = np.cross(v2, v1)    # normal to the (v2, v1) plane
    y_axis_unnorm = v1
    z_axis_unnorm = np.cross(x_axis_unnorm, y_axis_unnorm)
    
    # =
# array([ 0.01178087,  0.        , -0.40231605])
# 1 =
# array([-0.40231605, -0.00908256, -0.01178087])
# 2 =
# array([-3.65405795e-03,  1.61996992e-01, -1.07000368e-04])

    # 4) Normalize each to unit length:
    x_axis = norm1d(x_axis_unnorm)
    y_axis = norm1d(y_axis_unnorm)
    z_axis = norm1d(z_axis_unnorm)#
#
# 0 =
# array([ 0.02927007,  0.        , -0.99957154])
# 1 =
# array([-0.99931713, -0.02256026, -0.02926262])
# 2 =
# array([-2.25505912e-02,  9.99745485e-01, -6.60340253e-04])
    # 5) Stack them (as rows) to form a 3×3 matrix whose rows are the new axes,
    #    then transpose so that columns become those axes. This R maps “world → body”.
    #       [ x_axis ]
    #  R =  [ y_axis ]
    #       [ z_axis ]
    #  so that R @ world_point = coordinates in the “body‐aligned” frame.
    R_rows = np.vstack([x_axis, y_axis, z_axis])  # shape (3,3)
    base_rotation = R_rows.T                     # now shape (3,3)
 
    return base_rotation
# -------------------------------------------------------------
def joints_to_spherical(sm_large):
    """
    Convert smoothed joint positions to local spherical coords per frame.
    Returns array (T, N_pairs, 3).
    """
    pairs = []
    for child in STAFF_ORDER:
        child_idx = AMASS_TO_KINECT_MAP[child]
        parent_idx = AMASS_TO_KINECT_MAP[PARENT_RELATIONS[child]]
        pairs.append((child_idx, parent_idx))

    T = sm_large.shape[0]
    N = len(pairs)
    sph = np.zeros((T, N, 3))
    base_rotations = np.zeros((T, 3, 3))
    for t in range(T):
        base_rotations[t] = calculate_base_rotation(sm_large[t])

    for i, (c, p) in enumerate(pairs):
        #elL[-0.20552543,  0.17987859,  0.04762597])
        #shL array([-0.23155919,  0.4337002 ,  0.0182551 ])
        vec = sm_large[:, c, :] - sm_large[:, p, :]
        for t in range(T):
            Rb_inv = base_rotations[t].T
            vec[t] = np.dot(Rb_inv,vec[t])
        r = np.linalg.norm(vec, axis=1)
        with np.errstate(invalid='ignore'):
            theta = np.degrees(np.arccos(np.clip(vec[:, 2] / r, -1, 1)))
        phi = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
        # sph[:, i, 0] = r
        sph[:, i, 0] = np.nan_to_num(theta)
        sph[:, i, 1] = phi
       
        #  172.04001408123108, -143.5141970424375]
        
    return sph

# -------------------------------------------------------------
def calculate_physical_indices(joint_positions, keyframes, times, data_fps,
             sigma_large=20, window_size=21, sigma_small=1):
    """
    Complete pipeline:
        1) Smooth positions (large and small scales)
        2) Compute spherical coords from large-scale smoothed data
        3) Compute LMA indices on raw data
        4) Detect ground contacts and heights
    Returns dict with 'spherical', 'lma', 'ground_keyframes', 'ground_heights'.
    """
    sm_large = smooth_positions(joint_positions, sigma_large, window_size)
   
    spherical = joints_to_spherical(joint_positions)
    lma = compute_lma_indices(joint_positions, keyframes, np.array(times))
    ground_kfs, all_ground_heights = detect_ground_contacts(data_fps, sm_large)
    ground_heights=[]
    for frame_id in keyframes:
        closest_gorund_kf = max([kf for kf in ground_kfs if kf <= frame_id])        
        ground_heights.append(all_ground_heights[closest_gorund_kf])
        
    footL_idx = AMASS_TO_KINECT_MAP["footL"]
    footR_idx = AMASS_TO_KINECT_MAP["footR"]

    
    # 2) Grab the y-coordinate of footL, footR (axis=1 is the "vertical" index):
    footL_y = joint_positions[keyframes, footL_idx, 1]
    footR_y = joint_positions[keyframes, footR_idx, 1]
    foot_pos=np.stack([footL_y, footR_y], axis=1)

    
    
    return  spherical, lma, (foot_pos-np.stack([ground_heights, ground_heights], axis=1)).tolist()
    

# -------------------------------------------------------------
if __name__ == '__main__':
    # Test with fake data
    T = 100
    J = max(AMASS_TO_KINECT_MAP.values()) + 1

    joint_positions = np.random.rand(T, J, 3)
    times = np.linspace(0, 1, T)
    keyframes = [20, 40, 60, 80]
    data_fps = 120

    out = calculate_physical_indices(joint_positions, keyframes, times, data_fps)
    print("Spherical shape:", out[0].shape)
    # print("Number of LMA segments:", len(out['lma']))
    # print("LMA indices:", out['lma'])
    # print("Ground keyframes:", out['ground_keyframes'])
    # print("Ground heights (first 5):", out['ground_heights'][:5])
