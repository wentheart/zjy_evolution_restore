import scipy.io
import numpy as np

mat_data = scipy.io.loadmat("./net_data/ants/mat/ants.mat")

print(mat_data["net"].shape)