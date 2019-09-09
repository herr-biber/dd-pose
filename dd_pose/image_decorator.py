import numpy as np
import cv2

class ImageDecorator:
    def __init__(self, image, pcm, axis_length=0.1):
        assert image.shape[2] == 3
        self.image = image
        self.pcm = pcm
        self.axis_length = axis_length # in m
        
    def draw_axis(self, T_cam_axis, use_gray=False):
        """Draw rgb axis into camera image."""
        if self.pcm is None:
            return

        origin_axis = np.array([0.0, 0.0, 0.0, 1.0])
        x_axis = np.array([self.axis_length, 0.0, 0.0, 1.0])
        y_axis = np.array([0.0, self.axis_length, 0.0, 1.0])
        z_axis = np.array([0.0, 0.0, self.axis_length, 1.0])
        
        # transform to cam coordinate system
        origin_cam, x_cam, y_cam, z_cam = (T_cam_axis.dot(p) for p in (origin_axis, x_axis, y_axis, z_axis))
    
        # project into image
        origin_uv, x_uv, y_uv, z_uv = (self.pcm.project3dToPixel(p) for p in (origin_cam, x_cam, y_cam, z_cam))
    
        origin_uv, x_uv, y_uv, z_uv = ((int(u), int(v)) for (u,v) in (origin_uv, x_uv, y_uv, z_uv))
        origin_uv, x_uv, y_uv, z_uv

        # draw lines (BGR), x red, y green, z blue
        if use_gray:
            color_x = (127, 127, 127)
            color_y = (127, 127, 127)
            color_z = (127, 127, 127)
        else:
            color_x = (0, 0, 255) # r
            color_y = (0, 255, 0) # g
            color_z = (255, 0, 0) # b

        # draw red on top, as the frontal looking axis X is likyly hiding the other axes
        cv2.line(self.image, origin_uv, y_uv, color=color_y, thickness=3) # g
        cv2.line(self.image, origin_uv, z_uv, color=color_z, thickness=3) # b
        cv2.line(self.image, origin_uv, x_uv, color=color_x, thickness=3) # r
        
    def draw_text(self, text):
        cv2.putText(self.image, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)