# pandaorca_description

## Usage

[`pandaorca_single.usd`](usd/pandaorca_single.usd) has a single-arm setup corresponding to the object_in_bowl_processed_50hz dataset. [`pandaorca_dual.usd`](usd/pandaorca_dual.usd) has a dual-arm setup corresponding to the bag_groceries dataset.

Optionally, you may use the individual USD files for the fer (previously Franka Emika Panda) and the Orcahand to create a script-based setup in Python. Consider using this approach for future setups to make it easier to modify.

## To-do

- [ ] The alignment between Orcahand and the connector has a small error leading to a few mm offset. Fix this by adjusting the URDF then re-generating the USD files.
- [x] Modify initial joint angles to fit the teleoperation setup of the datasets. Attempts in [scripts](scripts/invkin_pose.py) but not working. (DONE IN `dataset_replay`)
- [x] Add camera sensors to the USD files to match the teleoperation setup. (DONE IN `dataset_replay`)
- [ ] Add texture to the table, which is a generic gray box right now.

## Source Models

Relevant meshes and URDF files were sourced from the following repoisitories:

- Franka Emika Panda (fer): [franka_description](https://github.com/frankarobotics/franka_description)
- Orcahand: [orcahand_description](https://github.com/orcahand/orcahand_description)
- Mount Connector: provided by the project supervisors

## How models were made

I initially tried to use CAD softwares to manually measure the transforms between the Panda, the connector, and the Orcahand, but it turned out to be difficult with the given files. Instead, I used Claude to generate a script that automatically detects the screw whole positions and used it for alignment. As a result, there is a small error between the Orcahand and the connector, which leads to a few mm offset. More information on this is available in [docs](docs/mesh_alignment_report.html)

If anyone has the CAD skills necessary, it is recommended to fix the alignment transforms to manual values instead of the automatic hole detection for accuracy.

Transform between the table and the arms were measured and implemented manually, based on the information given by the project supervisors.
