# This script loads a .mat file containing motion capture data and prints the number of frames in the dataset.
import scipy.io

mat = scipy.io.loadmat('Nick_3.mat', struct_as_record=False, squeeze_me=True)
mo = mat['Nick_3']
skel = mo.Skeletons
P = skel.PositionData  # shape: (3, J, T)
num_frames = P.shape[2]

print(f"Numero di frame in Nick_3.mat: {num_frames}")