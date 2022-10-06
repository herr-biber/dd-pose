import numpy as np
import cv2


class ImageDecorator:
    def __init__(self, image, pcm, axis_length=0.1):
        assert image.shape[2] == 3
        self.image = image
        self.pcm = pcm
        self.axis_length = axis_length # in m
        
    def draw_axis(self, T_cam_axis, use_gray=False, thickness=3):
        """Draw rgb axis into camera image."""
        if self.pcm is None:
            return

        # allow to misuse param for color
        if isinstance(use_gray, bool):
            color_gray = (127, 127, 127)
        else:
            assert len(use_gray) == 3
            color_gray = use_gray

        origin_axis = np.array([0.0, 0.0, 0.0, 1.0])
        x_axis = np.array([self.axis_length, 0.0, 0.0, 1.0])
        y_axis = np.array([0.0, self.axis_length, 0.0, 1.0])
        z_axis = np.array([0.0, 0.0, self.axis_length, 1.0])
        
        # transform to cam coordinate system
        origin_cam, x_cam, y_cam, z_cam = (T_cam_axis.dot(p) for p in (origin_axis, x_axis, y_axis, z_axis))
    
        # project into image
        origin_uv, x_uv, y_uv, z_uv = (self.pcm.project3dToPixel(p) for p in (origin_cam, x_cam, y_cam, z_cam))
    
        origin_uv, x_uv, y_uv, z_uv = ((int(u), int(v)) for (u,v) in (origin_uv, x_uv, y_uv, z_uv))
        # origin_uv, x_uv, y_uv, z_uv

        # draw lines (BGR), x red, y green, z blue
        if use_gray:
            color_x = color_gray
            color_y = color_gray
            color_z = color_gray
        else:
            color_x = (0, 0, 255)  # r
            color_y = (0, 255, 0)  # g
            color_z = (255, 0, 0)  # b

        # draw red on top, as the frontal looking axis X is likely hiding the other axes
        cv2.line(self.image, origin_uv, y_uv, color=color_y, thickness=thickness)  # g
        cv2.line(self.image, origin_uv, z_uv, color=color_z, thickness=thickness)  # b
        cv2.line(self.image, origin_uv, x_uv, color=color_x, thickness=thickness)  # r
        
    def draw_text(self, text):
        cv2.putText(self.image, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    def draw_rect_2d(self, uvwh, color=(0, 0, 0), thickness=1, confidence=1.0):
        # change color if a confidence is specified
        if confidence < 1:
            color = self.saturate_color(color, confidence)

        uvwh = np.array(uvwh).astype(np.int)  # cast
        pt1 = uvwh[0:2]
        pt2 = pt1 + uvwh[2:4]
        # pt1, pt2

        assert self.image.data.contiguous, "cv2.rectangle expects contiguous array"
        cv2.rectangle(self.image, tuple(pt1), tuple(pt2), color, thickness)

    @staticmethod
    def saturate_color(color=(0, 0, 0), value=1.0):
        """ changes color intensity by value
            value=1 returns color
            value=0 returns grey hue of matching intensity.
            with value=0, color (0,0,0) stays black, but (255,255,255) becomes slightly darker.

        """
        gray = 0.2989 * color[0] + 0.5870 * color[1] + 0.1140 * color[
            2]  # weights from random forum, aparently CCIR 601 spec.
        # gray = (color[0] + color[1] + color[2])/3.0 # alternative, simpler definition for gray.
        r = min(int(gray * (1 - value) + color[2] * value), 255)  # r
        g = min(int(gray * (1 - value) + color[1] * value), 255)  # g
        b = min(int(gray * (1 - value) + color[0] * value), 255)  # b
        return (r, g, b)

    def draw_points_3d(self, points, color=(0, 255, 255), size=4, thickness=-1):
        if self.pcm is None:
            return

        for p in points:
            # make sure point is in front of camera
            if p[2] > 0.0:
                u, v = self.pcm.project3dToPixel(p)
                u, v = int(u), int(v)
                try:
                    cv2.circle(self.image, (u, v), size, color, thickness) # yellow
                except OverflowError:
                    print("Could not draw circle. Waaay outside the image.")

    def draw_points_2d(self, points, color=(0, 255, 255), size=4, thickness=-1):
        for u, v in points:
            u, v = int(u), int(v)
            try:
                cv2.circle(self.image, (u, v), size, color, thickness)  # yellow
            except OverflowError:
                print("Could not draw circle. Waaay outside the image.")
