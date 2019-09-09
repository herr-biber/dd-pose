import numpy

# PinholeCameraModel "ducktype compatible" with ROS image_geometry PinholeCameraModel
class PinholeCameraModel:

    def __init__(self):
        self.K = None
        self.D = None
        self.R = None
        self.P = None
        self.full_K = None
        self.full_P = None
        self.width = None
        self.height = None
        self.binning_x = None
        self.binning_y = None
        self.raw_roi = None
        self.tf_frame = None
        self.stamp = None

    def project3dToPixel(self, point):
        """
        :param point:     3D point
        :type point:      (x, y, z)

        Projects 3D point to pixel coordinates (u, v) of the rectified camera image.
        """
        src = numpy.array([point[0], point[1], point[2], 1.0], dtype='float64').reshape((4,1))
        dst = self.P * src
        x = dst[0,0]
        y = dst[1,0]
        w = dst[2,0]
        if w != 0:
            return (x / w, y / w)
        else:
            return (float('nan'), float('nan'))