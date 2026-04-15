import h5py
with h5py.File("/home/teamb/Desktop/Sim2RealB/data/20250827_151212.h5", "r") as f:
    f.visit(print)