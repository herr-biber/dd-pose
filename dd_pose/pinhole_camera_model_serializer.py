import numpy as np
import json
from dd_pose.pinhole_camera_model import PinholeCameraModel

class PinholeCameraModelSerializer:
    # stores pinhole camera model parameters in inernal dict
    # can be converted to/from ROS PinholeCameraModel
    # can load/save to disk (via json)
    
    def __init__(self):
        self.camera_info = dict()
    
    def from_pinhole_camera_model(self, pcm):
        self.camera_info = {
            'height': pcm.height,
            'width': pcm.width,
            'distortion_model': 'plumb_bob',
            'D': pcm.D.tolist(),
            'K': pcm.K.tolist(),
            'R': pcm.R.tolist(),
            'P': pcm.P.tolist(),
            'binning_x': pcm.binning_x,
            'binning_y': pcm.binning_y,
            'roi': {
                'x_offset': pcm.raw_roi.x_offset,
                'y_offset': pcm.raw_roi.y_offset,
                'height': pcm.raw_roi.height,
                'width': pcm.raw_roi.width,
                'do_rectify': pcm.raw_roi.do_rectify
            }
        }
        
    def to_pinhole_camera_model(self):
        pcm = PinholeCameraModel()
        
        pcm.height = self.camera_info['height']
        pcm.width = self.camera_info['width']
        pcm.D = np.asarray(self.camera_info['D'])
        pcm.K = np.asarray(self.camera_info['K'])
        pcm.R = np.asarray(self.camera_info['R'])
        pcm.P = np.asarray(self.camera_info['P'])
        pcm.binning_x = self.camera_info['binning_x']
        pcm.binning_y = self.camera_info['binning_y']
#        pcm.raw_roi = sensor_msgs.msg.RegionOfInterest()
#        pcm.raw_roi.x_offset = self.camera_info['roi']['x_offset']
#        pcm.raw_roi.y_offset = self.camera_info['roi']['y_offset']
#        pcm.raw_roi.height = self.camera_info['roi']['height']
#        pcm.raw_roi.width = self.camera_info['roi']['width']
#        pcm.raw_roi.do_rectify = self.camera_info['roi']['do_rectify']
        
        return pcm
    
    def save(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.camera_info, fp, sort_keys=True)
            
    def load(self, filename):
        with open(filename) as fp:
            self.camera_info = json.load(fp)