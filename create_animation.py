import os
from os.path import join as pjoin
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from tqdm import tqdm
import torch

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)

def qinv(q):
    assert q.shape[-1] == 4
    return torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype) * q

def recover_root_rot_pos(data):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = data.clone().detach().float()
    
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), dtype=torch.float32, device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,), dtype=torch.float32, device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = data.clone().detach().float()
    
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # Get local joint positions from the vector
    # Skip root rotation velocity (1), root linear velocity (2), root height (1)
    # Take (joints_num-1)*3 elements for positions
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.reshape(data.shape[0], joints_num - 1, 3)

    # Apply rotation
    r_rot_quat_expanded = qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,))
    positions = qrot(r_rot_quat_expanded, positions)

    # Add root position
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatenate root position with joint positions
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions.detach().cpu().numpy()

def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()
    
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred','darkred','darkred','darkred']
    
    frame_number = data.shape[0]
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0]-trajec[index, 0], MAXS[0]-trajec[index, 0], 0, 
                     MINS[2]-trajec[index, 1], MAXS[2]-trajec[index, 1])
        
        if index > 1:
            ax.plot3D(trajec[:index, 0]-trajec[index, 0], 
                     np.zeros_like(trajec[:index, 0]), 
                     trajec[:index, 1]-trajec[index, 1], 
                     linewidth=1.0, color='blue')
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], 
                     data[index, chain, 1], 
                     data[index, chain, 2], 
                     linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, 
                       interval=1000/fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()

def main():
    # Define directories
    src_dir = './HumanML3D/new_joint_vecs/'
    tgt_ani_dir = "./HumanML3D/animations/"
    os.makedirs(tgt_ani_dir, exist_ok=True)

    # Define kinematic chain
    kinematic_chain = [
        [0, 2, 5, 8, 11],  # Right leg
        [0, 1, 4, 7, 10],  # Left leg
        [0, 3, 6, 9, 12, 15],  # Spine
        [9, 14, 17, 19, 21],  # Right arm
        [9, 13, 16, 18, 20]   # Left arm
    ]
    kinematic_chain_sep = [[k] for k in kinematic_chain]

    # Get list of npy files
    npy_files = sorted(os.listdir(src_dir))
    
    # Process each file
    for npy_file in tqdm(npy_files, desc="Processing motion files"):
        # Load motion vector data
        vec_data = np.load(pjoin(src_dir, npy_file))
        
        # Convert to 3D positions
        joints_num = 22  # HumanML3D uses 22 joints
        pos_data = recover_from_ric(vec_data, joints_num)
        
        # Read corresponding text file
        txt_src_dir = src_dir.replace('new_joint_vecs', 'texts')
        txt_file = npy_file.replace('.npy', '.txt')
        
        try:
            with open(pjoin(txt_src_dir, txt_file), 'r') as f:
                text_data = f.read()
            
            result_text = ""
            for line in text_data.split('\n'):
                result_text += line.split('#')[0] + '\n'

            # Generate animation
            save_path = pjoin(tgt_ani_dir, "vecs_"+npy_file[:-3] + 'mp4')
            plot_3d_motion(save_path, kinematic_chain, pos_data, 
                        title=result_text, fps=20, radius=4)

            # Generate separate animations for each kinematic chain (only for first file)
            if npy_file == npy_files[0]:
                for idx, k in enumerate(kinematic_chain_sep):
                    save_path = pjoin(tgt_ani_dir, "vecs_"+npy_file[:-3] + f'_kc_{idx}.mp4')
                    plot_3d_motion(save_path, k, pos_data, 
                                title=result_text, fps=20, radius=4)
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 