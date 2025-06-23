#!/usr/bin/env python3
"""
Load motion capture data from a MATLAB .mat file and print the total number of frames.
"""

import scipy.io

# Load the .mat file, without converting MATLAB structs to arrays
# - struct_as_record=False: keep MATLAB structs as Python objects
# - squeeze_me=True: remove singleton dimensions
mat = scipy.io.loadmat('Nick_3.mat', struct_as_record=False, squeeze_me=True)

# Access the top-level struct by its variable name in the .mat file
mo = mat['Nick_3']

# Within that struct, get the 'Skeletons' field (another struct)
skel = mo.Skeletons

# Extract the PositionData array
# The data layout is (3, J, T):
#   3 coordinates (X,Y,Z),
#   J joints,
#   T time frames
P = skel.PositionData

# The number of frames corresponds to the third dimension
num_frames = P.shape[2]

# Print out the result
print(f"Number of frames in Nick_3.mat: {num_frames}")
