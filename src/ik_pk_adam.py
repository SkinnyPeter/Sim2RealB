import torch

try:
    import pytorch_kinematics as pk
    _PK_AVAILABLE = True
except ImportError:
    _PK_AVAILABLE = False

_FRANKA_Q_LO = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_FRANKA_Q_HI = torch.tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


def build_pk_chain(urdf_path, device):
    """Load Franka FK chain from URDF onto `device`. Raises if pytorch_kinematics is missing."""
    if not _PK_AVAILABLE:
        raise RuntimeError("pytorch_kinematics is not installed — pip install pytorch-kinematics")
    with open(urdf_path, "rb") as f:
        urdf_bytes = f.read()
    chain = pk.build_serial_chain_from_urdf(
        urdf_bytes, "panda_hand", root_link_name="panda_link0"
    ).to(device=device)
    return chain


def _quat_xyzw_to_rotmat(q_xyzw, device):
    """(4,) xyzw ndarray -> (3,3) rotation matrix tensor."""
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    s = 2.0 / (x*x + y*y + z*z + w*w)
    return torch.tensor([
        [1 - s*(y*y + z*z),  s*(x*y - z*w),      s*(x*z + y*w)     ],
        [s*(x*y + z*w),      1 - s*(x*x + z*z),   s*(y*z - x*w)     ],
        [s*(x*z - y*w),      s*(y*z + x*w),        1 - s*(x*x + y*y)],
    ], dtype=torch.float32, device=device)


def pk_adam_ik_batch(chain, targets, q_warms, device,
                     n_iter=50, w_rot=0.2, pos_thresh=0.01):
    """IK for N targets via L-BFGS on a batched FK chain.

    L-BFGS uses approximate second-order curvature and converges in ~20-50
    function evaluations vs ~200 for Adam — typically 5-10x faster.
    Joint limits are enforced inside the closure so gradients go to zero at boundaries.

    targets   : list of (pos_np, quat_xyzw_np) tuples — length N
    q_warms   : list of (7,) tensors — warm-starts, length N
    n_iter    : max L-BFGS function evaluations (default 50)
    Returns   : list of (q_np ndarray, success bool) tuples, length N
    """
    N = len(targets)
    q_lo = _FRANKA_Q_LO.to(device)
    q_hi = _FRANKA_Q_HI.to(device)

    p_tgt = torch.stack([torch.tensor(p, dtype=torch.float32, device=device) for p, _ in targets])  # (N,3)
    R_tgt = torch.stack([_quat_xyzw_to_rotmat(q, device) for _, q in targets])  # (N,3,3)

    q = torch.stack([w.clone().detach().clamp(q_lo, q_hi) for w in q_warms]).requires_grad_(True)  # (N,7)
    opt = torch.optim.LBFGS([q], lr=1.0, max_iter=n_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        q_c = q.clamp(q_lo, q_hi)            # enforce limits; gradient goes to 0 at boundary
        mat = chain.forward_kinematics(q_c).get_matrix()  # (N,4,4)
        loss = ((mat[:, :3, 3] - p_tgt).pow(2).sum()
                + w_rot * (mat[:, :3, :3] - R_tgt).pow(2).sum())
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        q_final = q.clamp(q_lo, q_hi)
        mat_f = chain.forward_kinematics(q_final).get_matrix()
        pos_errs = (mat_f[:, :3, 3] - p_tgt).norm(dim=1)

    return [(q_final[i].detach().cpu().numpy(), pos_errs[i].item() < pos_thresh) for i in range(N)]


def pk_adam_ik(chain, target_pos_np, target_quat_xyzw_np, q_warm, device,
               n_iter=200, lr=0.05, w_rot=0.2, pos_thresh=0.01):
    """IK via Adam on a pytorch_kinematics FK chain.

    chain                : SerialChain on `device`, built with build_pk_chain()
    target_pos_np        : (3,)  ndarray, in robot base frame
    target_quat_xyzw_np  : (4,)  ndarray xyzw
    q_warm               : (7,)  tensor, warm-start (previous solution)
    Returns              : (7,)  ndarray, success bool
    """
    q_lo = _FRANKA_Q_LO.to(device)
    q_hi = _FRANKA_Q_HI.to(device)

    p_tgt = torch.tensor(target_pos_np, dtype=torch.float32, device=device)
    R_tgt = _quat_xyzw_to_rotmat(target_quat_xyzw_np, device)

    q = q_warm.clone().detach().clamp(q_lo, q_hi).requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        mat  = chain.forward_kinematics(q.unsqueeze(0)).get_matrix()  # (1,4,4)
        p_fk = mat[0, :3, 3]
        R_fk = mat[0, :3, :3]
        loss = (p_fk - p_tgt).pow(2).sum() + w_rot * (R_fk - R_tgt).pow(2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.clamp_(q_lo, q_hi)

    with torch.no_grad():
        q_final = q.clamp(q_lo, q_hi)
        mat_f   = chain.forward_kinematics(q_final.unsqueeze(0)).get_matrix()
        pos_err = (mat_f[0, :3, 3] - p_tgt).norm().item()

    return q_final.detach().cpu().numpy(), pos_err < pos_thresh
