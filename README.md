This branchs contains a Simulator.py module where every simulation related scripts were transformed into methods.
I have also changed the scene (should actually be names stage) so it contains two robot arm, two orca hands and a camera.

I used Tristan code for the arm trajectories, and duplicated it for both arms.

In the future:
- the stage needs to be same to the video (objects coordinates and orientations)
- same for the camera (intrinsics and extrinsics)
- also simulate the hands trajectories from the h5 file
- in the end there will be a lot in play(), should be decomposed for cleaner code
