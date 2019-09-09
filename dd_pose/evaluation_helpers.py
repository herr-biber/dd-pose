import numpy as np

# a coordinate frame which allows for identity transformation for a head frontally looking inside the camera
# (x pointing inside the camera (opposite to camera viewing direction)
# (y pointing towards right in camera image)
# (z pointing upwards in camera image)
T_camdriver_headfrontal = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

T_headfrontal_camdriver = np.linalg.inv(T_camdriver_headfrontal)
