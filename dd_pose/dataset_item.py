import os
import json
import warnings

import numpy as np
import imageio

try:
    from builtins import int # long integer in python2
except:
    pass

dd_pose_data_dir = os.environ['DD_POSE_DATA_DIR']

class StampedTransforms:
    def __init__(self, fp=None, transform_name=None):
        """
        fp: file-like object pointing to json content
        """
        
        self.transforms = dict()
        if fp is not None:
            master_stamps_dict = json.load(fp)
            
            for stamp, transform in master_stamps_dict.items():
                stamp = int(stamp)
                transform = np.array(transform)
                if transform.shape != (4, 4):
                    raise ValueError("Transform for stamp %ld is malformed" % stamp)
                self.transforms[int(stamp)] = (stamp, transform)
                
    def get_transform(self, stamp):
        if stamp not in self.transforms:
            return None
        return self.transforms[stamp][1]
    
    def __len__(self):
        return len(self.transforms)
    
    def has_stamp(self, stamp):
        return stamp in self.transforms
    
    def get_stamps(self):
        return self.transforms.keys()
    

class DatasetItem:
    def __init__(self, dataset_item, is_small_resolution=True, is_lazy=True):
        self.dataset_item = dataset_item
        self.is_test = dataset_item['is-test']
        self.is_small_resolution = is_small_resolution
        
        self.dataset_item_dir = os.path.join(dd_pose_data_dir, 'subject-%02d' % dataset_item['subject'], 'scenario-%02d' % dataset_item['scenario'], dataset_item['humanhash'])
        assert os.path.isdir(self.dataset_item_dir), "dataset item dir does not exist. did you run 04-untar.sh?"
        
        # load stamps
        with open(os.path.join(self.dataset_item_dir, 'stamps.txt')) as fp:
            self.stamps = set(int(line.strip()) for line in fp.readlines())

        # load camdriver-head transforms
        self.load_T_camdriver_head()

        # lazy loaded
        self.static_transforms = None
        self.stw_angles = None
        self.gps = None
        self.headings = None
        self.velocity = None
        self.dynamics = None
        self.occlusion_states = None
        self.driver_left_dont_care_bboxes = None
            
        if is_small_resolution:
            resdir = 'small-resolution'
        else:
            resdir = 'full-resolution'

        self.img_driver_left_dir = os.path.join(self.dataset_item_dir, resdir, 'driver-left-img')
        if not os.path.isdir(self.img_driver_left_dir):
            print("EE: Could not find driver left img dir: %s" % self.img_driver_left_dir)

        self.img_docu_dir = os.path.join(self.dataset_item_dir, resdir, 'docu-img')
        if not os.path.isdir(self.img_docu_dir):
            print("WW: Could not find docu img dir: %s" % self.img_docu_dir)
            
        if not is_lazy:
            self.load_all()
            
    def load_all(self):
        # load all lazy loadable attributes
        self.load_static_transforms()
        self.load_stw_angles()
        self.load_gps()
        self.load_headings()
        self.load_velocity()
        self.load_dynamics()
        self.load_occlusion_states()
        self.load_driver_left_dont_care_bboxes()

    def load_T_camdriver_head(self):
        T_camdriver_head_file = os.path.join(self.dataset_item_dir, 't-camdriver-head.json')
        if os.path.exists(T_camdriver_head_file):
            with open(T_camdriver_head_file) as fp:
                self.T_camdriver_head_transforms = StampedTransforms(fp, transform_name=None)
            if self.is_test:
                warnings.warn("WW: working with test set which has head pose measurements")
        else:
            self.T_camdriver_head_transforms = StampedTransforms()
            assert self.is_test, "head pose measurements not found. did you run 04-untar.sh?"

    def load_static_transforms(self):
        static_transforms_path = os.path.join(self.dataset_item_dir, 'static-transforms.json')
        assert os.path.exists(static_transforms_path)
        with open(static_transforms_path) as fp:
            static_transforms = json.load(fp)
        self.static_transforms = dict()
        for k, v in static_transforms.items():
            if v is not None:
                v = np.array(v)
            self.static_transforms[k] = v
       
    def load_stw_angles(self):
        # load stw angles
        stw_angles_path = os.path.join(self.dataset_item_dir, 'stw-angles.json')
        assert os.path.exists(stw_angles_path)
        with open(stw_angles_path) as fp:
            stw_angles = json.load(fp)
        self.stw_angles = {int(k): v for k, v in stw_angles.items()}
            
    def load_gps(self):
        # load gps
        gps_path = os.path.join(self.dataset_item_dir, 'gps.json')
        assert os.path.exists(gps_path)
        with open(gps_path) as fp:
            gps = json.load(fp)

        self.gps = dict()
        for k, v in gps.items():
            if v is not None:
                v['stamp'] = int(v['stamp'])
            self.gps[int(k)] = v
            
    def load_headings(self):
        # load headings
        headings_path = os.path.join(self.dataset_item_dir, 'gps-headings.json')
        assert os.path.exists(headings_path)
        with open(headings_path) as fp:
            headings = json.load(fp)

        self.headings = dict()
        for k, v in headings.items():
            if v is not None:
                v['stamp'] = int(v['stamp'])
            self.headings[int(k)] = v
        
    def load_velocity(self):
        # load velocity
        velocity_path = os.path.join(self.dataset_item_dir, 'vehicle-velocity.json')
        assert os.path.exists(velocity_path)
        with open(velocity_path) as fp:
            velocity = json.load(fp)

        self.velocity = dict()
        for k, v in velocity.items():
            if v is not None:
                v['stamp'] = int(v['stamp'])
            self.velocity[int(k)] = v
            
    def load_dynamics(self):
        # load dynamics
        dynamics_path = os.path.join(self.dataset_item_dir, 'vehicle-dynamics.json')
        assert os.path.exists(dynamics_path)
        with open(dynamics_path) as fp:
            dynamics = json.load(fp)

        self.dynamics = dict()
        for k, v in dynamics.items():
            if v is not None:
                v['stamp'] = int(v['stamp'])
            self.dynamics[int(k)] = v

    def load_occlusion_states(self):
        occlusion_states_path = os.path.join(self.dataset_item_dir, 'occlusion-labels.json')
        assert os.path.exists(occlusion_states_path)
        with open(occlusion_states_path) as fp:
            occlusion_states = json.load(fp)

        self.occlusion_states = {int(stamp): occlusion_state for stamp, occlusion_state in occlusion_states.items()}

    def load_driver_left_dont_care_bboxes(self):
        dont_care_bboxes_path = os.path.join(self.dataset_item_dir, 'driver-left-dont-care-bboxes.json')
        if not os.path.exists(dont_care_bboxes_path):
            warnings.warn("don't care boxes file does no exist. Not ignoring any boxes.")
            self.driver_left_dont_care_bboxes = dict()  # empty
            return

        with open(dont_care_bboxes_path) as fp:
            dont_care_bboxes = json.load(fp)

        dont_care_bboxes = {int(stamp): np.asarray(dont_care_bbox) for stamp, dont_care_bbox in dont_care_bboxes.items()}
        # dont care boxes are given in full resolution of left image
        # scale down for small resolution
        if self.is_small_resolution:
            dont_care_bboxes = {k: (v / 2.0).astype(int) for k, v in dont_care_bboxes.items()}

        self.driver_left_dont_care_bboxes = dont_care_bboxes

    def get_subject(self):
        return self.dataset_item['subject']

    def get_scenario(self):
        return self.dataset_item['scenario']

    def get_humanhash(self):
        return self.dataset_item['humanhash']
    
    def get_occlusion_state(self, stamp):
        if self.occlusion_states is None:
            self.load_occlusion_states()
        if not stamp in self.occlusion_states:
            return None
        return self.occlusion_states[stamp]

    def get_driver_left_dont_care_bboxes(self, stamp):
        if self.driver_left_dont_care_bboxes is None:
            self.load_driver_left_dont_care_bboxes()
        return self.driver_left_dont_care_bboxes.get(stamp, None)
    
    def has_stamp(self, stamp):
        return stamp in self.stamps
        
    def __len__(self):
        return len(self.stamps)
    
    def get_stamps(self):
        return sorted(self.stamps)
        
    # DYNAMIC TRANSFORMS
    def get_T_camdriver_head(self, stamp):
        return self.T_camdriver_head_transforms.get_transform(stamp)

    # STATIC TRANSFORMS
    def get_T_camdriver_camdocu(self):
        if self.static_transforms is None:
            self.load_static_transforms()
        return self.static_transforms['T-camdriver-camdocu']

    def get_T_camdriver_gps(self):
        if self.static_transforms is None:
            self.load_static_transforms()
        return self.static_transforms['T-camdriver-gps']
    
    def get_T_camdriver_body(self):
        if self.static_transforms is None:
            self.load_static_transforms()
        return self.static_transforms['T-camdriver-body']
    
    def get_stw_angle(self, stamp):
        if self.stw_angles is None:
            self.load_stw_angles()
        if stamp not in self.stw_angles:
            return None
        s = self.stw_angles[stamp]
        if s is None:
            return None
        return (s['angle-deg'], s['angle-speed'])

    def get_gps(self, stamp):
        if self.gps is None:
            self.load_gps()
        if stamp not in self.gps:
            return None
        return self.gps[stamp]
    
    def get_velocity(self, stamp):
        """m/s"""
        if self.velocity is None:
            self.load_velocity()
        if stamp not in self.velocity:
            return None
        vv = self.velocity[stamp]
        if vv is None:
            return None
        return (vv['stamp'], vv['velocity'])
    
    def get_yaw_rate(self, stamp):
        """deg/s"""
        if self.dynamics is None:
            self.load_dynamics()
        if stamp not in self.dynamics:
            return None
        vv = self.dynamics[stamp]
        if vv is None:
            return None
        return (vv['stamp'], vv['yaw-rate'])

    def get_heading(self, stamp):
        """deg: 0..359"""
        if self.headings is None:
            self.load_headings()
        if stamp not in self.headings:
            return None
        vv = self.headings[stamp]
        if vv is None:
            return None
        return (vv['stamp'], vv['gps-heading'])

    def has_img_driver_left_files(self, stamp):
        if not self.has_stamp(stamp):
            print("stamp not found")
            return False

        img_file = os.path.join(self.img_driver_left_dir, '%ld.png' % stamp)

        ci_file = os.path.join(self.img_driver_left_dir, '%ld.json' % stamp)
        return os.path.isfile(img_file) and os.path.isfile(ci_file)

    def get_img_driver_left(self, stamp, shift=True):
        # lazy import
        from dd_pose.pinhole_camera_model_serializer import PinholeCameraModelSerializer
        
        if not self.has_stamp(stamp):
            print("stamp not found")
            return None, None
        img_file = os.path.join(self.img_driver_left_dir, '%ld.png' % stamp)

        if not os.path.isfile(img_file):
            return None, None
        
        img = imageio.imread(img_file)
        img = img.astype(np.uint16)
        
        ci_file = os.path.join(self.img_driver_left_dir, '%ld.json' % stamp)
        if os.path.isfile(ci_file):
        
            pcms = PinholeCameraModelSerializer()
            pcms.load(ci_file)
            pcm = pcms.to_pinhole_camera_model()
        else:
            print("could not load pcm file")
            pcm = None
        
        if shift:
            img >>=8
            img = img.astype(np.uint8)
        
        return img, pcm

    def get_img_docu(self, stamp):
        # lazy import
        from dd_pose.pinhole_camera_model_serializer import PinholeCameraModelSerializer
        
        if not self.has_stamp(stamp):
            print("stamp not found")
            return None
        
        img_file = os.path.join(self.img_docu_dir, '%ld.png' % stamp)

        if not os.path.isfile(img_file):
            return None, None
        img = imageio.imread(img_file)
        img = img[...,::-1].copy() #rgb -> bgr, copy to have "aligned" memory which e.g. cv2.line may work with inplace
        
        ci_file = os.path.join(self.img_docu_dir, '%ld.json' % stamp)
        if os.path.isfile(ci_file):
            pcms = PinholeCameraModelSerializer()
            pcms.load(ci_file)
            pcm = pcms.to_pinhole_camera_model()
        else:
            print("Could not load pcm file")
            pcm = None
        
        return img, pcm

    def __str__(self):
        s = "DatasetItem(subject=%02d, scenario=%02d, humanhash=%s, len=%d)" % (self.get_subject(), self.get_scenario(), self.get_humanhash(), len(self))
        return s

    def __repr__(self):
        return str(self)