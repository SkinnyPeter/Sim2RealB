#!/usr/bin/env python3
"""
Mesh-based alignment for Franka Panda + 90deg Connector + OrcaHand assembly.

Extracts screw-hole centres and mating-face normals directly from STL meshes,
then computes the rigid transforms for the URDF joints:
    fer_link8 -> connector_mount
    connector_mount -> orcahand_world

Supports both left and right hand configurations via --hand flag.

Usage:
    python3 compute_alignment.py           # defaults to left
    python3 compute_alignment.py --hand left
    python3 compute_alignment.py --hand right
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from collections import defaultdict
import trimesh
import os, itertools, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(BASE, "meshes")

# ── Per-hand URDF constants ──────────────────────────────────────────
# These values come from the standalone orcahand_{left,right}_extended.urdf:
#   world2{side}_tower_fixed joint origin  →  WORLD_TO_TOWER_XYZ
#   {side}_visual_tower_camera_mesh visual origin  →  TOWER_CAMERA_VIS_{XYZ,RPY}

HAND_PARAMS = {
    "left": dict(
        WORLD_TO_TOWER_XYZ=np.array([-0.04, 0.0, 0.04575]),
        TOWER_CAMERA_VIS_XYZ=np.array([-0.04141561887638722, 0.05282580179465202, -0.019856742052109723]),
        TOWER_CAMERA_VIS_RPY=np.array([3.0660133933203646, -1.0811766886677918, -1.4653561026636517]),
        TOWER_CAMERA_MESH="orcahand/left/visual/left_visual_tower_camera_mesh.stl",
    ),
    "right": dict(
        WORLD_TO_TOWER_XYZ=np.array([0.04, 0.0, 0.04575]),
        TOWER_CAMERA_VIS_XYZ=np.array([0.04141561887638722, 0.052825801794652016, -0.019856742052109723]),
        TOWER_CAMERA_VIS_RPY=np.array([-3.066013393320364, -1.0811766886677918, -1.6762365509261419]),
        TOWER_CAMERA_MESH="orcahand/right/visual/right_visual_tower_camera_mesh.stl",
    ),
}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hand", choices=["left", "right"], default="left",
                    help="Which hand to compute alignment for (default: left)")
args = parser.parse_args()

HAND = args.hand
hp = HAND_PARAMS[HAND]
WORLD_TO_TOWER_XYZ    = hp["WORLD_TO_TOWER_XYZ"]
TOWER_CAMERA_VIS_XYZ  = hp["TOWER_CAMERA_VIS_XYZ"]
TOWER_CAMERA_VIS_RPY  = hp["TOWER_CAMERA_VIS_RPY"]
TOWER_CAMERA_MESH     = hp["TOWER_CAMERA_MESH"]

print(f"Computing alignment for {HAND.upper()} hand")
print()


# ── Utility ──────────────────────────────────────────────────────────

def cluster_normals(mesh, angle_tol_deg=5.0, max_planes=30):
    normals, areas = mesh.face_normals, mesh.area_faces
    cos_tol = np.cos(np.radians(angle_tol_deg))
    used = np.zeros(len(normals), dtype=bool)
    planes = []
    for _ in range(max_planes):
        if used.all(): break
        ra = areas.copy(); ra[used] = 0
        seed = int(np.argmax(ra))
        if ra[seed] < 1e-12: break
        members = (~used) & (np.dot(normals, normals[seed]) > cos_tol)
        avg_n = np.average(normals[members], axis=0, weights=areas[members])
        avg_n /= np.linalg.norm(avg_n)
        members = (~used) & (np.dot(normals, avg_n) > cos_tol)
        fi = np.where(members)[0]; used[members] = True
        tc = mesh.triangles_center[fi]
        rep = np.average(tc, axis=0, weights=areas[fi])
        planes.append(dict(normal=avg_n, fi=fi, area=float(areas[fi].sum()),
                           center=rep, n_faces=int(fi.size)))
    planes.sort(key=lambda p: p["area"], reverse=True)
    return planes


def boundary_loops(mesh, face_indices):
    faces = mesh.faces[face_indices]
    ec = defaultdict(int)
    for f in faces:
        for e in ((min(f[0],f[1]),max(f[0],f[1])),
                  (min(f[1],f[2]),max(f[1],f[2])),
                  (min(f[0],f[2]),max(f[0],f[2]))):
            ec[e] += 1
    bnd = [e for e,c in ec.items() if c == 1]
    if not bnd: return []
    adj = defaultdict(set)
    for a,b in bnd: adj[a].add(b); adj[b].add(a)
    visited = set(); loops = []
    for start in adj:
        if start in visited: continue
        loop = [start]; visited.add(start); cur = start
        for _ in range(len(bnd)+1):
            nxt = next((n for n in adj[cur] if n not in visited), None)
            if nxt is None:
                if start in adj[cur]: loops.append(loop)
                break
            loop.append(nxt); visited.add(nxt); cur = nxt
    return loops


def detect_holes(mesh, fi, normal, min_r=0.002, max_r=0.006,
                 max_circ=0.15, min_verts=12):
    holes = []
    for loop in boundary_loops(mesh, fi):
        v = mesh.vertices[loop]; c = v.mean(axis=0)
        d = np.linalg.norm(v - c, axis=1)
        r, s = float(d.mean()), float(d.std())
        circ = s / (r + 1e-12)
        if min_r <= r <= max_r and circ < max_circ and len(loop) >= min_verts:
            holes.append(dict(center=c, radius=r, circularity=circ, n_verts=len(loop)))
    return holes


def build_frame(p1, p2, normal):
    u = p2 - p1; u /= np.linalg.norm(u)
    n = normal / np.linalg.norm(normal)
    v = np.cross(n, u); v /= np.linalg.norm(v)
    u = np.cross(v, n);  u /= np.linalg.norm(u)
    return np.column_stack([u, v, n])


def mate(F_src, mid_src, F_dst, mid_dst, flip):
    """Compute R,t so that src mates onto dst with normals opposing.
    flip='A' keeps U, flips V,N;  flip='B' flips U, keeps V, flips N."""
    Fm = F_dst.copy()
    if flip == 'A':
        Fm[:,1] = -F_dst[:,1]; Fm[:,2] = -F_dst[:,2]
    else:
        Fm[:,0] = -F_dst[:,0]; Fm[:,2] = -F_dst[:,2]
    R = Fm @ np.linalg.inv(F_src)
    return R, mid_dst - R @ mid_src


def fmt(v, scale=1): return f"({v[0]*scale:+.4f}, {v[1]*scale:+.4f}, {v[2]*scale:+.4f})"


# ── 1. Connector ─────────────────────────────────────────────────────
print("=" * 72)
print("STEP 1 — Connector mesh")
print("=" * 72)

conn = trimesh.load(os.path.join(MESH_DIR, "mount/Gavin90DegMount_edited_P1S.stl"))
conn.vertices *= 0.001   # mm -> m
planes_c = cluster_normals(conn)

# Find two perpendicular mounting faces
p0 = planes_c[0]
p1 = next(p for p in planes_c[1:] if abs(np.dot(p0["normal"], p["normal"])) < 0.1)

# Assign panda vs orca side: panda face has bolt-circle center closer to STL origin
if np.linalg.norm(p0["center"]) < np.linalg.norm(p1["center"]):
    face_P, face_O = p0, p1
else:
    face_P, face_O = p1, p0

nP  = face_P["normal"]     # panda-face outward normal
nOC = face_O["normal"]     # orca-face outward normal

print(f"  Panda face  n={fmt(nP)}  area={face_P['area']*1e6:.0f} mm²")
print(f"  Orca  face  n={fmt(nOC)}  area={face_O['area']*1e6:.0f} mm²")
print(f"  dot = {np.dot(nP, nOC):.6f}")

holes_P  = detect_holes(conn, face_P["fi"], nP)
holes_OC = detect_holes(conn, face_O["fi"], nOC)
hP  = [h["center"] for h in holes_P ]
hOC = [h["center"] for h in holes_OC]
cP  = np.mean(hP,  axis=0)
cOC = np.mean(hOC, axis=0)

print(f"\n  Panda-face holes ({len(hP)}):")
for i, h in enumerate(hP):
    print(f"    {i}: {fmt(h, 1000)} mm   r={holes_P[i]['radius']*1000:.2f} mm")
print(f"  Bolt-circle centre: {fmt(cP, 1000)} mm")

print(f"\n  Orca-face holes ({len(hOC)}):")
for i, h in enumerate(hOC):
    print(f"    {i}: {fmt(h, 1000)} mm   r={holes_OC[i]['radius']*1000:.2f} mm")
print(f"  Bolt-circle centre: {fmt(cOC, 1000)} mm")

rP  = np.mean([np.linalg.norm(h-cP)  for h in hP])
rOC = np.mean([np.linalg.norm(h-cOC) for h in hOC])
print(f"\n  Bolt-circle radii:  panda = {rP*1000:.2f} mm   orca = {rOC*1000:.2f} mm")


# ── 2. OrcaHand mounting face ────────────────────────────────────────
print("\n" + "=" * 72)
print(f"STEP 2 — OrcaHand {HAND} tower-camera mesh (orcahand world frame)")
print("=" * 72)

orca_mesh = trimesh.load(os.path.join(MESH_DIR, TOWER_CAMERA_MESH))
T_vis = np.eye(4)
T_vis[:3,:3] = Rotation.from_euler('xyz', TOWER_CAMERA_VIS_RPY).as_matrix()
T_vis[:3, 3] = TOWER_CAMERA_VIS_XYZ
orca_mesh.apply_transform(T_vis)
orca_mesh.vertices += WORLD_TO_TOWER_XYZ

planes_o = cluster_normals(orca_mesh)
mount_pl = next(p for p in planes_o if abs(p["normal"][2]) > 0.95)
nOM = mount_pl["normal"]   # orca mounting-face normal (~+Z)
print(f"  Mounting face  n={fmt(nOM)}  area={mount_pl['area']*1e6:.0f} mm²")

holes_OM_all = detect_holes(orca_mesh, mount_pl["fi"], nOM,
                            min_r=0.002, max_r=0.005, max_circ=0.35, min_verts=8)
holes_OM = [h for h in holes_OM_all
            if 0.015 < abs(h["center"][1]) < 0.040
            and abs(h["center"][2]) < 0.010]

hOM = [h["center"] for h in holes_OM]
cOM = np.mean(hOM, axis=0)
print(f"  Mounting holes ({len(hOM)}):")
for i, h in enumerate(hOM):
    print(f"    {i}: {fmt(h, 1000)} mm   r={holes_OM[i]['radius']*1000:.2f} mm")
d_orca = np.linalg.norm(hOM[0]-hOM[1])*1000
print(f"  Hole spacing: {d_orca:.2f} mm")
print(f"  Centre: {fmt(cOM, 1000)} mm")


# ── 3. Interface 1 — link8 → connector (panda face) ─────────────────
print("\n" + "=" * 72)
print("STEP 3 — link8 → connector_mount")
print("=" * 72)

F_link8 = np.eye(3)
conn_centroid = conn.centroid

results_1 = []
for perm in itertools.permutations(range(len(hP))):
    h0, h1 = hP[perm[0]], hP[perm[1]]
    F_cP = build_frame(h0, h1, nP)
    for flip in ('A','B'):
        R, t = mate(F_cP, cP, F_link8, np.zeros(3), flip)
        mapped = np.array([R @ h + t for h in hP])
        z_err = np.abs(mapped[:,2]).max()
        r_err = np.abs(np.linalg.norm(mapped[:,:2], axis=1) - rP).max()
        cc_z  = (R @ conn_centroid + t)[2]
        results_1.append(dict(R=R, t=t, flip=flip, z_err=z_err, r_err=r_err,
                              cc_z=cc_z, mapped=mapped))

# Keep only physically valid: connector centroid in +Z from link8
results_1 = [r for r in results_1 if r["cc_z"] > 0]

# Among valid candidates, pick the one where the orca-face normal direction
# in the link8 frame best matches the current URDF's orientation.
# Current URDF maps the orca-face normal to approximately (-1, 0, 0) in link8.
# This is a DESIGN CHOICE (which direction the connector bends from the flange).
R_old_1 = Rotation.from_euler('xyz', [2.424095, -0.034947, -1.572674]).as_matrix()
nOC_target = R_old_1 @ nOC   # orca face normal direction from old URDF
nOC_target /= np.linalg.norm(nOC_target)

for r in results_1:
    r["nOC_link8"] = r["R"] @ nOC
    r["ang_match"] = np.dot(r["nOC_link8"], nOC_target)

# Sort by angular match (higher dot = better match), then by quality
results_1.sort(key=lambda r: (-r["ang_match"], r["z_err"] + r["r_err"]))
best1 = results_1[0]
R1, t1 = best1["R"], best1["t"]

print(f"  Best: flip={best1['flip']}  z_err={best1['z_err']*1e3:.4f} mm  "
      f"r_err={best1['r_err']*1e3:.4f} mm  cc_z={best1['cc_z']*1e3:.1f} mm  "
      f"ang_match={best1['ang_match']:.4f}")
print(f"  Panda-face holes in link8 frame:")
for i, p in enumerate(best1["mapped"]):
    print(f"    {i}: {fmt(p, 1000)} mm  radial={np.linalg.norm(p[:2])*1000:.2f} mm")

nOC_link8 = R1 @ nOC
print(f"\n  Orca face normal in link8: {fmt(nOC_link8)}")
print(f"  (This is the direction the orcahand extends from the flange)")
print(f"  Target from old URDF:      {fmt(nOC_target)}")


# ── 4. Interface 2 — connector → orcahand (orca face) ────────────────
print("\n" + "=" * 72)
print("STEP 4 — connector_mount → orcahand_world")
print("=" * 72)

# Find which pair of connector orca-face holes best matches the orca spacing
orca_sp = np.linalg.norm(hOM[0] - hOM[1])
best_pair, best_err = None, 1e9
for i in range(len(hOC)):
    for j in range(i+1, len(hOC)):
        e = abs(np.linalg.norm(hOC[i]-hOC[j]) - orca_sp)
        if e < best_err:
            best_err = e; best_pair = (i,j)
ci, cj = best_pair
print(f"  Matched connector pair: ({ci},{cj})  "
      f"spacing={np.linalg.norm(hOC[ci]-hOC[cj])*1000:.2f} mm  "
      f"err={best_err*1000:.2f} mm vs orca {orca_sp*1000:.2f} mm")

results_2 = []
for ci2, cj2 in itertools.combinations(range(len(hOC)), 2):
  for oi, oj in [(0,1),(1,0)]:
    Fc = build_frame(hOC[ci2], hOC[cj2], nOC)
    Fo = build_frame(hOM[oi], hOM[oj], nOM)
    mid_c = (hOC[ci2] + hOC[cj2]) / 2
    mid_o = (hOM[oi] + hOM[oj]) / 2
    for flip in ('A','B'):
        R, t = mate(Fo, mid_o, Fc, mid_c, flip)
        mp = np.array([R @ hOM[oi] + t, R @ hOM[oj] + t])
        tp = np.array([hOC[ci2], hOC[cj2]])
        h_err = np.linalg.norm(mp - tp, axis=1).max()
        # body probe: 50mm below mounting face (into orca body)
        probe = R @ (cOM - 0.05*nOM) + t
        side  = np.dot(probe - cOC, nOC)   # should be >0 (outward)
        results_2.append(dict(R=R, t=t, ci=ci2, cj=cj2, oi=oi, oj=oj,
                              flip=flip, h_err=h_err, side=side))

results_2.sort(key=lambda r: r["h_err"])

# Separate by body direction
opt_out = [r for r in results_2 if r["side"] > 0]
opt_in  = [r for r in results_2 if r["side"] <= 0]
print(f"  Candidates outward: {len(opt_out)}   inward: {len(opt_in)}")

if opt_out:
    best2 = opt_out[0]
else:
    best2 = results_2[0]
    print("  WARNING: all inward — using best fit anyway")

R2, t2 = best2["R"], best2["t"]
print(f"  Best: ci={best2['ci']} cj={best2['cj']} oi={best2['oi']} oj={best2['oj']} "
      f"flip={best2['flip']}  h_err={best2['h_err']*1e3:.2f} mm  side={best2['side']:.4f}")

# Alternate (opposite flip for same hole pair, or inward)
alt2 = None
for r in (opt_in if opt_out else opt_out):
    alt2 = r; break
if alt2 is None:
    for r in results_2:
        if r is not best2:
            alt2 = r; break


# ── 5. Output ─────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 5 — URDF-ready values")
print("=" * 72)

def show(label, R, t):
    rpy = Rotation.from_matrix(R).as_euler('xyz')
    rpd = np.degrees(rpy)
    q   = Rotation.from_matrix(R).as_quat()  # xyzw
    print(f"\n  {label}:")
    print(f"    xyz:       {t[0]:+.6f}  {t[1]:+.6f}  {t[2]:+.6f}")
    print(f"    rpy (deg): {rpd[0]:+.3f}  {rpd[1]:+.3f}  {rpd[2]:+.3f}")
    print(f"    rpy (rad): {rpy[0]:+.6f}  {rpy[1]:+.6f}  {rpy[2]:+.6f}")
    print(f"    quat xyzw: {q[0]:+.6f}  {q[1]:+.6f}  {q[2]:+.6f}  {q[3]:+.6f}")
    print(f'    URDF: <origin xyz="{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}" '
          f'rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>')

show("fer_link8 → connector_mount", R1, t1)
show("connector_mount → orcahand_world [OPTION I]", R2, t2)
if alt2:
    show("connector_mount → orcahand_world [OPTION II — 180° flip]", alt2["R"], alt2["t"])

# Direct link8 → orcahand
R_dir = R1 @ R2; t_dir = R1 @ t2 + t1
show("link8 → orcahand_world (direct, option I)", R_dir, t_dir)


# ── 6. Verification ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("STEP 6 — Verification")
print("=" * 72)

# Normals
chk_nP = R1 @ nP
print(f"\n  Panda face normal in link8: {fmt(chk_nP)}   "
      f"dot(0,0,-1) = {np.dot(chk_nP, [0,0,-1]):.6f}")
chk_nOM = R2 @ nOM
print(f"  Orca mount normal in connector: {fmt(chk_nOM)}   "
      f"dot(-nOC) = {np.dot(chk_nOM, -nOC):.6f}")

# Mapped holes
print(f"\n  Orca holes mapped to connector frame:")
for i, h in enumerate(hOM):
    p = R2 @ h + t2
    print(f"    {i}: {fmt(p, 1000)} mm")
print(f"  Connector orca-face holes:")
for i, h in enumerate(hOC):
    print(f"    {i}: {fmt(h, 1000)} mm")

# Compare with current URDF
print(f"\n  ── Comparison with current URDF ──")
R_old = Rotation.from_euler('xyz', [2.424095, -0.034947, -1.572674]).as_matrix()
t_old = np.array([0.053088, 0.080038, 0.001877])
nP_old = R_old @ nP
print(f"  Old: panda normal in link8 = {fmt(nP_old)}  dot(0,0,-1)={np.dot(nP_old,[0,0,-1]):.4f}")
nOC_old = R_old @ nOC
print(f"  Old: orca face normal in link8 = {fmt(nOC_old)}")
cc_old = R_old @ conn_centroid + t_old
print(f"  Old: connector centroid in link8 = {fmt(cc_old, 1000)} mm")
cc_new = R1 @ conn_centroid + t1
print(f"  New: connector centroid in link8 = {fmt(cc_new, 1000)} mm")

print("\nDone.")
