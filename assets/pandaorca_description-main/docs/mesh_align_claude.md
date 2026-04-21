#[Deprecated] Franka Panda + 90° Connector + OrcaHand: Assembly Transform Computation

## Purpose

Compute the rigid-body transforms needed to assemble three components in Isaac Sim (or Blender/URDF):

1. **Franka Panda** robot arm (URDF name: `fer`) — already placed at world origin
2. **90° connector** (STL: `Gavin90DegMount_edited_P1S.stl`) — bolts onto the Franka flange (link8)
3. **OrcaHand left** (URDF: `orcahand_left_extended.urdf`) — bolts onto the other face of the connector

The three components were imported into Blender via the Phobos add-on. Screw hole positions and surface planes were sampled manually in Blender to compute mating transforms.

---

## 1. Input Files

| File | Description |
|------|-------------|
| `fer.urdf` | Franka Panda arm, 7-DOF, links `fer_link0` through `fer_link8` (flange) |
| `Gavin90DegMount_edited_P1S.stl` | 90° connector, units in mm, bounding box 155×172×87mm |
| `orcahand_left_extended.urdf` | OrcaHand left, root link = `world`, tower + wrist + 5 fingers |

---

## 2. Raw Input Coordinates (from Blender)

All coordinates are in **meters**, in the **Blender world frame** (which equals the Franka panda base frame, since the panda was imported at the origin).

### 2.1 Screw Hole Positions

Eight points were placed at screw hole locations on the mating surfaces. Pairs 1&2 are on the panda↔connector interface; pairs 3&4 are on the connector↔orca interface. Within each pair, the `_panda`/`_orca` point is on the robot/hand, and the `_connector` point is the corresponding hole on the connector.

**Panda ↔ Connector interface (pairs 1, 2):**

| Point | X | Y | Z | Source |
|-------|---|---|---|--------|
| `1_panda` | 0.12495 | 0.017661 | 0.92391 | Blender screenshot (Image 7) |
| `1_connector` | 0.097678 | 0.009915 | 0.012902 | Blender screenshot (Image 8) |
| `2_panda` | 0.089685 | −0.017694 | 0.92638 | Blender screenshot (Image 5) |
| `2_connector` | 0.062322 | 0.035725 | 0.037065 | Blender screenshot (Image 6) |

**Connector ↔ OrcaHand interface (pairs 3, 4) — CORRECTED VALUES:**

| Point | X | Y | Z | Source |
|-------|---|---|---|--------|
| `3_connector` | 0.08 | 0.15149 | 0.015312 | Blender screenshot (corrected, Image from 00:03:13) |
| `3_orca` | −0.080653 | 0.02506 | −0.003879 | Blender screenshot (Image 3) |
| `4_connector` | 0.08 | 0.11732 | 0.051812 | Blender screenshot (corrected, Image from 00:03:15) |
| `4_orca` | −0.08058 | −0.02508 | −0.003878 | Blender screenshot (Image 1) |

Note: `3_connector` and `4_connector` were corrected in a second round of screenshots. The original values (0.155, 0.018592) and (0.12082, 0.055093) were slightly off.

### 2.2 Surface Plane Sample Points

Because the screw holes were picked assuming surfaces parallel to world axes (they are not), three additional points per surface were sampled to define the actual mating planes. Each group of 3 points defines a plane via cross-product of two edge vectors.

**Panda flange plane** (for `1_panda`, `2_panda`):

| Point | X | Y | Z |
|-------|---|---|---|
| A | 0.091726 | 0.020231 | 0.92623 |
| B | 0.12817 | −0.01883 | 0.93321 |
| C | 0.088751 | −0.020803 | 0.92644 |

**Connector face A** (panda-side face, for `1_connector`, `2_connector`):

| Point | X | Y | Z |
|-------|---|---|---|
| A | 0.060497 | 0.038032 | 0.039225 |
| B | 0.10026 | 0.033841 | 0.035301 |
| C | 0.064147 | 0.007608 | 0.010741 |

**OrcaHand mounting plane** (for `3_orca`, `4_orca`):

| Point | X | Y | Z |
|-------|---|---|---|
| A | −0.083545 | −0.026597 | −0.003879 |
| B | −0.080767 | 0.02837 | −0.003879 |
| C | −0.10229 | −0.02076 | −0.003879 |

**Connector face B** (orca-side face, for `3_connector`, `4_connector`):

| Point | X | Y | Z |
|-------|---|---|---|
| A | 0.081825 | 0.11866 | 0.0574 |
| B | 0.051474 | 0.13856 | 0.036153 |
| C | 0.08365 | 0.155 | 0.018592 |

---

## 3. Algorithm

### 3.1 Plane Fitting and Point Projection

For each surface, fit a plane from the 3 sample points:
```
normal = normalize(cross(B − A, C − A))
```

Then project each screw hole point onto its plane:
```
projected = point − dot(point − A, normal) * normal
```

**Computed plane normals:**

| Surface | Normal (X, Y, Z) |
|---------|-------------------|
| Panda flange | (+0.170085, −0.017374, −0.985276) |
| Connector face A | (−0.000004, +0.683449, −0.729998) |
| OrcaHand mount | (+0.000000, −0.000000, +1.000000) |
| Connector face B | (−0.000098, −0.729934, −0.683518) |

**Projected screw hole positions:**

| Point | X | Y | Z | Δ from raw (mm) |
|-------|---|---|---|-----------------|
| `1_panda` | 0.123592 | 0.017800 | 0.931774 | 7.98 |
| `2_panda` | 0.089657 | −0.017691 | 0.926542 | 0.16 |
| `1_connector` | 0.097678 | 0.009916 | 0.012901 | 0.00 |
| `2_connector` | 0.062322 | 0.035725 | 0.037065 | 0.00 |
| `3_connector` | 0.080000 | 0.154997 | 0.018596 | 4.80 |
| `4_connector` | 0.080000 | 0.120822 | 0.055091 | 4.80 |
| `3_orca` | −0.080653 | 0.025060 | −0.003879 | 0.00 |
| `4_orca` | −0.080580 | −0.025080 | −0.003879 | 0.00 |

**Inter-hole distances** (sanity check — should match across mating pairs):

| Pair | Distance |
|------|----------|
| Panda 1↔2 | 49.38 mm |
| Connector 1↔2 (face A) | 50.00 mm |
| Connector 3↔4 (face B) | 50.00 mm |
| OrcaHand 3↔4 | 50.14 mm |

### 3.2 Frame Construction

For each mating surface, build an orthonormal frame from 2 hole positions + plane normal:

```
U = normalize(hole2 − hole1)          # in-plane, along hole pair
V = normalize(cross(Normal, U))       # in-plane, perpendicular to U
U = normalize(cross(V, Normal))       # recomputed for exact orthogonality
Frame = [U | V | Normal]              # 3×3 matrix, columns are basis vectors
```

### 3.3 Mating Transform Computation

For two surfaces to mate face-to-face, their normals must oppose (point into each other). Given source frame `F_src` and destination frame `F_dst`, there are **two valid mating orientations** (differing by 180° rotation around the face normal):

**Option A** — flip V and N:
```
F_mated = [+U_dst | −V_dst | −N_dst]
R = F_mated × inverse(F_src)
```

**Option B** — flip U and N:
```
F_mated = [−U_dst | +V_dst | −N_dst]
R = F_mated × inverse(F_src)
```

Both are valid rigid-body matings with normals opposing (verified: dot product = −1.0000). They differ by 180° rotation around the destination normal. One places the moved part on the "correct" side (extending outward from the joint), the other on the "wrong" side (extending inward/through the other part).

**Translation** is computed from the midpoints:
```
t = midpoint_dst − R × midpoint_src
```

In Option A, holes match straight (1↔1, 2↔2). In Option B, holes cross-match (1↔2, 2↔1). Both are physically valid for symmetric bolt patterns.

### 3.4 Forward Kinematics: Base → link8

The URDF `fer_link8` frame at zero joint configuration was computed via forward kinematics by chaining all joint transforms from the URDF:

| Joint | XYZ (m) | RPY (rad) |
|-------|---------|-----------|
| `fer_base_joint` | (0, 0, 0) | (0, 0, 0) |
| `fer_joint1` | (0, 0, 0.333) | (0, 0, 0) |
| `fer_joint2` | (0, 0, 0) | (−π/2, 0, 0) |
| `fer_joint3` | (0, −0.316, 0) | (+π/2, 0, 0) |
| `fer_joint4` | (0.0825, 0, 0) | (+π/2, 0, 0) |
| `fer_joint5` | (−0.0825, 0.384, 0) | (−π/2, 0, 0) |
| `fer_joint6` | (0, 0, 0) | (+π/2, 0, 0) |
| `fer_joint7` | (0.088, 0, 0) | (+π/2, 0, 0) |
| `fer_joint8` | (0, 0, 0.107) | (0, 0, 0) |

**Result:**
```
link8 position (world):  (0.088000, 0.000000, 0.926000)
link8 rotation (world):  Rx(180°) — the link8 frame is flipped 180° about X relative to world!
```

**This is critical:** The link8 X-axis points in the same direction as world X, but link8 Y and Z are negated relative to world Y and Z. Any transform expressed in world frame must be converted to link8 frame via:
```
R_in_link8 = link8_R^T × R_in_world
t_in_link8 = link8_R^T × (t_in_world − link8_position)
```

### 3.5 Determining Correct Option (A vs B)

Options A and B give identical numerical quality (same hole errors, same normal alignment). They differ by which side of the mating surface the moved part ends up on. During this work, we were unable to computationally determine the correct option — the body-direction heuristic was ambiguous for both interfaces because the connector is a 90° bend (its body direction is not perpendicular to either face).

**Recommendation:** Try Option A first in Blender. If the connector mesh overlaps with / penetrates the panda arm, switch to Option B. Same for the orca-on-connector joint.

---

## 4. Final Transform Values

### Conventions

- **xyz**: Translation in meters
- **rpy**: Rotation as roll-pitch-yaw (XYZ Euler convention), given in both degrees and radians
- **quat wxyz**: Quaternion as (w, x, y, z) — used by Isaac Sim, Blender quaternion mode
- **quat xyzw**: Quaternion as (x, y, z, w) — used by scipy, ROS
- **World frame** = Blender scene origin = Franka panda base frame
- **link8 frame** = `fer_link8` local frame = Rx(180°) from world at zero config

### 4.1 Option A

#### Connector in world frame (Blender)

```
xyz:          +0.026893   +0.032488   +0.917503
rpy (deg):    +135.726    -9.776      +0.971
rpy (rad):    +2.368862   -0.170624   +0.016939
quat wxyz:    +0.374772   +0.923155   -0.024291   +0.082104
quat xyzw:    +0.923155   -0.024291   +0.082104   +0.374772
```

#### OrcaHand in world frame (Blender)

```
xyz:          +0.175468   -0.093350   +1.013625
rpy (deg):    +88.844     -9.693      +0.975
rpy (rad):    +1.550626   -0.169166   +0.017008
quat wxyz:    +0.711120   +0.697925   -0.054404   +0.065182
quat xyzw:    +0.697925   -0.054404   +0.065182   +0.711120
```

#### fer_link8 → connector (URDF / Isaac Sim)

```
xyz:          -0.061107   -0.032488   +0.008497
rpy (deg):    -44.274     +9.776      -0.971
rpy (rad):    -0.772730   +0.170624   -0.016939
quat wxyz:    +0.923155   -0.374772   +0.082104   +0.024291
quat xyzw:    -0.374772   +0.082104   +0.024291   +0.923155
```

#### fer_link8 → orcahand direct (URDF / Isaac Sim)

```
xyz:          +0.087468   +0.093350   -0.087625
rpy (deg):    -91.156     +9.693      -0.975
rpy (rad):    -1.590967   +0.169166   -0.017008
quat wxyz:    -0.697925   +0.711120   -0.065182   -0.054404
quat xyzw:    +0.711120   -0.065182   -0.054404   -0.697925
```

#### connector → orcahand (if orca is child of connector)

```
xyz:          +0.160617   +0.140662   +0.039568
rpy (deg):    -46.881     -0.057      -0.061
rpy (rad):    -0.818224   -0.000996   -0.001067
quat wxyz:    +0.917474   -0.397795   -0.000245   -0.000687
quat xyzw:    -0.397795   -0.000245   -0.000687   +0.917474
```

---

### 4.2 Option B (180° from Option A)

#### Connector in world frame (Blender)

```
xyz:          +0.185458   -0.032288   +0.946018
rpy (deg):    +138.047    +9.777      -179.029
rpy (rad):    +2.409369   +0.170633   -3.124654
quat wxyz:    +0.076542   -0.038383   +0.930038   +0.357347
quat xyzw:    -0.038383   +0.930038   +0.357347   +0.076542
```

#### OrcaHand in world frame (Blender)

```
xyz:          +0.172838   +0.098598   +1.010005
rpy (deg):    -91.167     -9.860      +0.978
rpy (rad):    -1.591166   -0.172086   +0.017068
quat wxyz:    -0.697778   +0.711090   +0.066216   +0.055430
quat xyzw:    +0.711090   +0.066216   +0.055430   -0.697778
```

#### fer_link8 → connector (URDF / Isaac Sim)

```
xyz:          +0.097458   +0.032288   -0.020018
rpy (deg):    -41.953     -9.777      +179.029
rpy (rad):    -0.732223   -0.170633   +3.124654
quat wxyz:    +0.038383   +0.076542   -0.357347   +0.930038
quat xyzw:    +0.076542   -0.357347   +0.930038   +0.038383
```

#### fer_link8 → orcahand direct (URDF / Isaac Sim)

```
xyz:          +0.084838   -0.098598   -0.084005
rpy (deg):    +88.833     +9.860      -0.978
rpy (rad):    +1.550426   +0.172086   -0.017068
quat wxyz:    +0.711090   +0.697778   +0.055430   -0.066216
quat xyzw:    +0.697778   +0.055430   -0.066216   +0.711090
```

#### connector → orcahand (if orca is child of connector)

```
xyz:          -0.000616   +0.140820   +0.039422
rpy (deg):    +46.881     +0.057      +179.939
rpy (rad):    +0.818224   +0.000996   +3.140526
quat wxyz:    +0.000687   -0.000245   +0.397795   +0.917474
quat xyzw:    -0.000245   +0.397795   +0.917474   +0.000687
```

---

## 5. Verification

| Metric | Option A | Option B |
|--------|----------|----------|
| Interface 1 hole error | 0.31 mm | 0.31 mm |
| Interface 2 hole error | 0.07 mm | 0.07 mm |
| Interface 1 normal dot | −1.0000 | −1.0000 |
| Interface 2 normal dot | −1.0000 | −1.0000 |
| Option A hole matching | straight (1↔1, 2↔2) | — |
| Option B hole matching | — | cross (1↔2, 2↔1) |

Both options produce identical geometric quality. They differ only in which side of the mounting face the part extends to.

---

## 6. How to Use

### In Blender

1. Import panda URDF via Phobos (it sits at world origin)
2. Import connector STL as a separate object
3. Import orca URDF via Phobos as a separate collection
4. Select the connector root object, set Mode to "XYZ Euler"
5. Enter the **connector world-frame** Location and Rotation values from Option A or B
6. Select the orca root object, enter the **orcahand world-frame** values
7. Check visually — if parts penetrate each other, switch to the other option

### In URDF (for Isaac Sim or ROS)

Use the **fer_link8 → connector** and **connector → orcahand** values:

```xml
<joint name="flange_to_connector" type="fixed">
  <parent link="fer_link8"/>
  <child link="connector_link"/>
  <!-- Use Option A or B fer_link8→connector values here -->
  <origin xyz="..." rpy="..."/>
</joint>

<joint name="connector_to_orcahand" type="fixed">
  <parent link="connector_link"/>
  <child link="orcahand_world"/>
  <!-- Use Option A or B connector→orcahand values here -->
  <origin xyz="..." rpy="..."/>
</joint>
```

Or skip the connector and use the **direct** fer_link8 → orcahand transform.

### In Isaac Sim (Python API)

Use position (xyz) + orientation as quaternion (wxyz convention for Isaac Sim).

---

## 7. Known Issues and Caveats

1. **180° ambiguity not resolved.** The user reported that early solutions had the hand "attached inwards not outwards." Both Option A and Option B are provided — one is correct, the other is 180° wrong. This must be verified visually.

2. **~10° pitch on panda interface.** The panda flange plane normal has a 9.8° tilt from the expected direction. This is either real geometry from the Blender mesh at zero config, or slight imprecision in the 3 plane sample points. The exact values include this tilt; the snapped values (multiples of 45°) ignore it.

3. **link8 frame is Rx(180°) from world.** At zero configuration, `fer_link8` has a 180° rotation about X relative to the panda base frame. Earlier iterations of this computation incorrectly used world-frame values as if they were link8-relative, which is why "fer→connector" was broken.

4. **Screw hole positions are approximate.** They were picked by hand in Blender on mesh vertices, not from parametric CAD. Errors of 0.3–5mm are expected.

5. **The `1_panda` point moved 8mm during projection.** This suggests it was picked on a vertex that was significantly off the flange plane. The other points moved < 0.2mm.

---

## 8. Appendix: Reproduction Script

```python
import numpy as np
from scipy.spatial.transform import Rotation

def fit_plane(pts):
    n = np.cross(pts[1]-pts[0], pts[2]-pts[0])
    n /= np.linalg.norm(n)
    return n, pts[0]

def project(pt, n, p0):
    return pt - np.dot(pt - p0, n) * n

def build_frame(p1, p2, normal):
    u = p2 - p1; u /= np.linalg.norm(u)
    n = normal / np.linalg.norm(normal)
    v = np.cross(n, u); v /= np.linalg.norm(v)
    u = np.cross(v, n); u /= np.linalg.norm(u)
    return np.column_stack([u, v, n])

def mating_transform(F_src, F_dst, option='A'):
    Fm = F_dst.copy()
    if option == 'A':
        Fm[:, 1] = -F_dst[:, 1]  # flip V
        Fm[:, 2] = -F_dst[:, 2]  # flip N
    else:  # option B
        Fm[:, 0] = -F_dst[:, 0]  # flip U
        Fm[:, 2] = -F_dst[:, 2]  # flip N
    return Fm @ np.linalg.inv(F_src)

# --- Plane sample points ---
plane_panda = np.array([[0.091726,0.020231,0.92623],
                         [0.12817,-0.01883,0.93321],
                         [0.088751,-0.020803,0.92644]])
plane_connA = np.array([[0.060497,0.038032,0.039225],
                         [0.10026,0.033841,0.035301],
                         [0.064147,0.007608,0.010741]])
plane_orca  = np.array([[-0.083545,-0.026597,-0.003879],
                         [-0.080767,0.02837,-0.003879],
                         [-0.10229,-0.02076,-0.003879]])
plane_connB = np.array([[0.081825,0.11866,0.0574],
                         [0.051474,0.13856,0.036153],
                         [0.08365,0.155,0.018592]])

# --- Fit planes ---
n_panda, pt_panda = fit_plane(plane_panda)
n_connA, pt_connA = fit_plane(plane_connA)
n_orca,  pt_orca  = fit_plane(plane_orca)
n_connB, pt_connB = fit_plane(plane_connB)

# --- Raw hole positions → project onto planes ---
p1_panda = project(np.array([0.12495, 0.017661, 0.92391]), n_panda, pt_panda)
p2_panda = project(np.array([0.089685,-0.017694, 0.92638]), n_panda, pt_panda)
p1_conn  = project(np.array([0.097678, 0.009915, 0.012902]), n_connA, pt_connA)
p2_conn  = project(np.array([0.062322, 0.035725, 0.037065]), n_connA, pt_connA)
p3_conn  = project(np.array([0.08, 0.15149, 0.015312]), n_connB, pt_connB)
p4_conn  = project(np.array([0.08, 0.11732, 0.051812]), n_connB, pt_connB)
p3_orca  = project(np.array([-0.080653, 0.02506,-0.003879]), n_orca, pt_orca)
p4_orca  = project(np.array([-0.08058,-0.02508,-0.003878]), n_orca, pt_orca)

# --- Build frames ---
F_connA = build_frame(p1_conn, p2_conn, n_connA)
F_panda = build_frame(p1_panda, p2_panda, n_panda)
F_orca  = build_frame(p3_orca, p4_orca, n_orca)
F_connB = build_frame(p3_conn, p4_conn, n_connB)

# --- Compute transforms (choose 'A' or 'B') ---
option = 'A'  # or 'B'

R1 = mating_transform(F_connA, F_panda, option)
t1 = (p1_panda + p2_panda)/2 - R1 @ (p1_conn + p2_conn)/2

R2 = mating_transform(F_orca, F_connB, option)
t2 = (p3_conn + p4_conn)/2 - R2 @ (p3_orca + p4_orca)/2

# Combined: panda_base → orcahand
R_direct = R1 @ R2
t_direct = R1 @ t2 + t1

# --- Convert to link8-relative (for URDF) ---
# FK: link8 at zero config
def T(xyz, rpy):
    M = np.eye(4)
    M[:3,:3] = Rotation.from_euler('xyz', rpy).as_matrix()
    M[:3, 3] = xyz
    return M

FK = np.eye(4)
for xyz, rpy in [
    ([0,0,0],[0,0,0]), ([0,0,0.333],[0,0,0]),
    ([0,0,0],[-1.5707963,0,0]), ([0,-0.316,0],[1.5707963,0,0]),
    ([0.0825,0,0],[1.5707963,0,0]), ([-0.0825,0.384,0],[-1.5707963,0,0]),
    ([0,0,0],[1.5707963,0,0]), ([0.088,0,0],[1.5707963,0,0]),
    ([0,0,0.107],[0,0,0])
]:
    FK = FK @ T(xyz, rpy)

l8_pos = FK[:3, 3]  # (0.088, 0, 0.926)
l8_R = FK[:3, :3]   # Rx(180°)

R1_l8 = l8_R.T @ R1
t1_l8 = l8_R.T @ (t1 - l8_pos)

R_direct_l8 = l8_R.T @ R_direct
t_direct_l8 = l8_R.T @ (t_direct - l8_pos)

# --- Print results ---
for label, R, t_vec in [
    ("connector (world)", R1, t1),
    ("orcahand (world)", R_direct, t_direct),
    ("fer_link8 → connector", R1_l8, t1_l8),
    ("fer_link8 → orcahand", R_direct_l8, t_direct_l8),
    ("connector → orcahand", R2, t2),
]:
    rpy_deg = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    rpy_rad = Rotation.from_matrix(R).as_euler('xyz')
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    print(f"\n{label}:")
    print(f"  xyz:       {t_vec[0]:+.6f}  {t_vec[1]:+.6f}  {t_vec[2]:+.6f}")
    print(f"  rpy (deg): {rpy_deg[0]:+.3f}  {rpy_deg[1]:+.3f}  {rpy_deg[2]:+.3f}")
    print(f"  rpy (rad): {rpy_rad[0]:+.6f}  {rpy_rad[1]:+.6f}  {rpy_rad[2]:+.6f}")
    print(f"  quat wxyz: {q[3]:+.6f}  {q[0]:+.6f}  {q[1]:+.6f}  {q[2]:+.6f}")
```