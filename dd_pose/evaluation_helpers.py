import warnings

import math
import numpy as np
import zipfile

import os
import json
import pandas as pd
import transformations as tr
from multiprocess import Pool
import logging

import plotly
import plotly.graph_objs as go

from dd_pose.dataset_item import DatasetItem, StampedTransforms

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


def angle_difference(angle1_rad, angle2_rad):
    # create resulting signed difference between two angles given in rad
    # https://stackoverflow.com/a/7869457
    a = angle1_rad - angle2_rad
    a = (a + np.pi) % (2 * np.pi) - np.pi
    return a


def angle_from_matrix(matrix, eps=1e-8):
    """Return rotation angle from rotation matrix.

    Adapted from https://github.com/davheld/tf/blob/master/src/tf/transformations.py
    with larger eps (and removed direction and point computation).

    Additionally removed eigenvalue==1 check and used eigenvalue closest to 1.

    Args:
        matrix: 3x3 numpy array rotation matrix
        eps: epsilon to use for numeric comparisons

    Returns
        Rotation angle in radians
    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    index_with_eigenvalue_close_to_one = np.argmin(abs(np.real(l) - 1.0))
    direction = np.real(W[:, index_with_eigenvalue_close_to_one]).squeeze()

    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > eps:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > eps:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return abs(angle)



class FilePredictor:
    def __init__(self, predictions_dir, di_dict=None):
        self.predictions_file = os.path.join(predictions_dir,\
                                       'subject-%02d' % di_dict['subject'],\
                                       'scenario-%02d' % di_dict['scenario'],\
                                       di_dict['humanhash'],\
                                       't-camdriver-head-predictions.json')

        with open(self.predictions_file) as fp:
            self.predictions = StampedTransforms(fp)

        try:
            with open(os.path.join(predictions_dir, 'metadata.json')) as fp:
                self.metadata = json.load(fp)
        except:
            self.metadata = dict()
        
    def get_T_camdriver_head(self, stamp):
        return self.predictions.get_transform(stamp)
    
    def get_t_camdriver_head(self, stamp):
        T_camdriver_head = self.get_T_camdriver_head(stamp)
        if T_camdriver_head is None:
            return None
        
        return T_camdriver_head[0:3,3]
    
    def get_T_headfrontal_head(self, stamp):
        T_camdriver_head = self.get_T_camdriver_head(stamp)
        if T_camdriver_head is None:
            return None
        
        T_headfrontal_head = np.dot(T_headfrontal_camdriver, T_camdriver_head)
        return T_headfrontal_head


class ZipFilePredictor(FilePredictor):
    def __init__(self, zip_file, di_dict=None):
        self.zf = zipfile.ZipFile(zip_file)
        self.predictions_file = os.path.join('subject-%02d' % di_dict['subject'],\
                                               'scenario-%02d' % di_dict['scenario'],\
                                               di_dict['humanhash'],\
                                               't-camdriver-head-predictions.json')

        if self.predictions_file not in self.zf.namelist():
            logging.warning("File %s not in zipfile", self.predictions_file)
            self.predictions = StampedTransforms(fp=None)
        else:
            with self.zf.open(self.predictions_file) as fp:
                try:
                    self.predictions = StampedTransforms(fp)
                except ValueError as e:
                    e.message = 'File %s is malformed json' % self.predictions_file
                    raise e
        
        if 'metadata.json' not in self.zf.namelist():
            logging.warning("File metadata.json not in zipfile")
            self.metadata = dict()
        else:
            with self.zf.open('metadata.json') as fp:
                try:
                    self.metadata = json.load(fp)
                except:
                    self.metadata = dict()


class EvaluationData:
    """
    EvaluationData ground truth and hypotheses in a pandas dataframe.
    
    It allows to filtering to subsets (easy, moderate, hard) and compute metrics.
    Correspondence of ground truth and hypotheses is given via integer stamp.
    """
    def __init__(self, is_ignore_full_occlusion=True):
        self.df = pd.DataFrame()
        self.df.index.name = 'stamp'
        self.name = ""
        self.is_ignore_full_occlusion = is_ignore_full_occlusion

        self.test_timestamps_wihout_images = {
            1540223383838000000,
            1540223395535000000,
            1540223463096000000,
            1540223703877000000,
            1540223715463000000,
            1540223718201000000,
            1540223733307000000,
            1540223745695000000,
            1540223754335000000,
            1540223755956000000,
            1540223949141000000,
            1540223977199000000,
            1540223981990000000,
            1540224341594000000,
            1540224348301000000,
            1541004815253000000,
            1541004831903000000,
            1541004973296000000,
            1541516264884000000,
            1541516279751000000,
            1541516304696000000,
            1541516310455000000,
            1541516312713000000,
            1541516390284000000,
            1541516406184000000,
            1541516410595000000,
            1541516430154000000,
            1541516527570000000,
            1541516571167000000,
            1541516575817000000,
            1541516607711000000,
            1541516628418000000,
            1541516636302000000,
            1541516636976000000,
            1541516657301000000,
            1541516990372000000,
            1541517238972000000,
            1541517256347000000,
            1541517269729000000,
            1541517271672000000,
            1541517273455000000,
            1541517635490000000,
            1541517694114000000,
            1541517718910000000,
            1541517817406000000,
            1541520716745000000,
            1541520724023000000,
            1541520726487000000,
            1541520765768000000,
            1541520946433000000,
            1541520988282000000,
            1541520990759000000,
            1541521009519000000,
            1541521087992000000,
            1541521091522000000,
            1541521125331000000,
            1541521134394000000,
            1541521134937000000,
            1541521470473000000,
            1541521476402000000,
            1541521477381000000,
            1541521501746000000,
            1541521537254000000,
            1541521591155000000,
            1541683614199000000,
            1541684382537000000,
            1541769285463000000,
            1541769291944000000,
            1541770428576000000,
            1541770440373000000,
            1541774806125000000,
            1541775089804000000,
            1541775274467000000,
            1541775282793000000,
            1541775283659000000,
            1541776122040000000,
            1541776124335000000
        }
        
    def load(self, di_dict, predictor):
        di = DatasetItem(di_dict)
        self.df['subject'] = pd.Series(data=di.get_subject(), index=di.get_stamps())
        self.df['scenario'] = di.get_scenario()
        self.df['humanhash'] = di.get_humanhash()
        
        for stamp in di.get_stamps():
            # Skip timestamps without an image.
            if stamp in self.test_timestamps_wihout_images:
                continue

            occlusion_state = di.get_occlusion_state(stamp)
            if self.is_ignore_full_occlusion and occlusion_state in {'full', 'full-auto'}:
                continue

            T_camdriver_head = di.get_T_camdriver_head(stamp)
            
            assert T_camdriver_head is not None
            
            T_headfrontal_head = T_headfrontal_camdriver.dot(T_camdriver_head)
            self.df.at[stamp, 'gt_roll'], self.df.at[stamp, 'gt_pitch'], self.df.at[stamp, 'gt_yaw'] = tr.euler_from_matrix(T_headfrontal_head)
            self.df.at[stamp, 'gt_x'], self.df.at[stamp, 'gt_y'], self.df.at[stamp, 'gt_z'] = T_camdriver_head[0:3,3]
            
            gt_angle_from_zero = angle_from_matrix(T_headfrontal_head)
            self.df.at[stamp, 'gt_angle_from_zero'] = abs(gt_angle_from_zero)

            self.df.at[stamp, 'occlusion_state'] = occlusion_state
            
            hypo_T_headfrontal_head = predictor.get_T_headfrontal_head(stamp)
            if hypo_T_headfrontal_head is None:
                self.df.at[stamp, 'hypo_roll'] = None
                self.df.at[stamp, 'hypo_pitch'] = None
                self.df.at[stamp, 'hypo_yaw'] = None
                self.df.at[stamp, 'angle_diff'] = None
                self.df.at[stamp, 'hypo_x'] = None
                self.df.at[stamp, 'hypo_y'] = None
                self.df.at[stamp, 'hypo_z'] = None
            else:
                self.df.at[stamp, 'hypo_roll'], self.df.at[stamp, 'hypo_pitch'], self.df.at[stamp, 'hypo_yaw'] = tr.euler_from_matrix(hypo_T_headfrontal_head)
                T_gt_hypo = tr.inverse_matrix(T_headfrontal_head).dot(hypo_T_headfrontal_head)
                angle_difference = angle_from_matrix(T_gt_hypo)  # rad
                self.df.at[stamp, 'angle_diff'] = abs(angle_difference)

                self.df.at[stamp, 'hypo_x'], self.df.at[stamp, 'hypo_y'], self.df.at[stamp, 'hypo_z'] = predictor.get_t_camdriver_head(stamp)

#                 print gt_angle_from_zero, angle_difference, np.rad2deg(angle_difference), position_difference

        # Remove rows with nondefined occlusion state.
        self.df = self.df[~self.df.occlusion_state.isna()]


    @staticmethod
    def load_evaluation_data(di_dict, predictor_class, predictor_kwargs):
        """
        Factory method creating an EvaluationData object with loaded ground truth and predictions from predictor.
        """
        ed = EvaluationData()
        predictor_kwargs.update({'di_dict': di_dict})
        predictor = predictor_class(**predictor_kwargs)
        ed.load(di_dict, predictor)
        return ed

    def load_all(self, di_dicts, predictor_class, predictor_kwargs, is_parallel=True):
        """
        Load both ground truth and predictions for all di_dicts.
        """
        if is_parallel:
            p = Pool(12)
            eds = p.map(lambda di_dict: EvaluationData.load_evaluation_data(di_dict, predictor_class, predictor_kwargs), di_dicts)
        else:
            eds = map(lambda di_dict: EvaluationData.load_evaluation_data(di_dict, predictor_class, predictor_kwargs), di_dicts)
            
        self.df = pd.concat([e.df for e in eds], sort=True)

        del eds
        
        diff = self.df[['gt_x','gt_y', 'gt_z']].values - self.df[['hypo_x', 'hypo_y', 'hypo_z']].values
        self.df['pos_diff'] = np.linalg.norm(diff, axis=1)
        
    def get_dx(self):
        return abs((self.df.hypo_x - self.df.gt_x)).mean(skipna=True)

    def get_dy(self):
        return abs((self.df.hypo_y - self.df.gt_y)).mean(skipna=True)

    def get_dz(self):
        return abs((self.df.hypo_z - self.df.gt_z)).mean(skipna=True)

    def get_dxyz(self):
        """
        Get mean absoulte L2 distance.
        """    
        return abs(self.df.pos_diff).mean(skipna=True)
    
    def get_recall(self):
        """
        Get recall, i.e. ratio of available predictions and ground truth measurements.
        """
        # ignore invalid frames (no gt)
        df = self.df[~self.df.gt_x.isna()]
        n_gt = df.gt_x.count()
        n_pos = df.hypo_x.count()

        if n_gt > 0:
            recall = float(n_pos)/n_gt
        else:
            recall = np.nan
        return recall

    def get_drpy(self):
        """
        Returns np.ndarray 3

        rad.
        """
        valid_rows = ~self.df.hypo_roll.isna()
        if not valid_rows.any():
            return np.full(3, fill_value=np.nan)
        # rad
        return np.abs(angle_difference(self.df[['gt_roll', 'gt_pitch', 'gt_yaw']][valid_rows].values,
                                       self.df[['hypo_roll', 'hypo_pitch', 'hypo_yaw']][valid_rows].values)).mean(axis=0)
    
    def get_mae(self):
        """ deg! """
        mae = np.rad2deg(self.df.angle_diff.mean(skipna=True))
        return mae
    
    def new_by_angle_range(self, angle_rad_min, angle_rad_max):
        ed = EvaluationData()
        ed.df = self.df[(self.df.gt_angle_from_zero >= angle_rad_min) & (self.df.gt_angle_from_zero < angle_rad_max)]
        ed.name = self.name + "%.0f<=a<%.0f" % (angle_rad_min, angle_rad_max)
        return ed

    def new_by_roll_range(self, angle_rad_min, angle_rad_max):
        ed = EvaluationData()
        ed.df = self.df[(self.df.gt_roll.abs() >= angle_rad_min) & (self.df.gt_roll.abs() < angle_rad_max)]
        return ed

    def new_by_pitch_range(self, angle_rad_min, angle_rad_max):
        ed = EvaluationData()
        ed.df = self.df[(self.df.gt_pitch.abs() >= angle_rad_min) & (self.df.gt_pitch.abs() < angle_rad_max)]
        return ed

    def new_by_yaw_range(self, angle_rad_min, angle_rad_max):
        ed = EvaluationData()
        ed.df = self.df[(self.df.gt_yaw.abs() >= angle_rad_min) & (self.df.gt_yaw.abs() < angle_rad_max)]
        return ed

    def new_by_occlusion_none(self):
        ed = EvaluationData()
        ed.df = self.df[(self.df.occlusion_state == 'none-auto') | (self.df.occlusion_state == 'none')]
        ed.name = self.name + " occl=none"
        return ed

    def new_by_occlusion_none_partial(self):
        ed = EvaluationData()
        ed.df = self.df[(self.df.occlusion_state == 'none-auto') | (self.df.occlusion_state == 'none') | (self.df.occlusion_state == 'partial') | (self.df.occlusion_state == 'partial-auto')]
        ed.name = self.name + " occl<=partial"
        return ed
    
    def new_by_dist_z(self, min_z, max_z=None):
        ed = EvaluationData()

        ed.df = self.df[self.df.gt_z >= min_z]
        ed.name = self.name + " z>=%.2f" % min_z
        if max_z is not None:
            ed.df = ed.df[ed.df.gt_z < max_z]
            ed.name += " z<%.2f" % max_z
        return ed
   
    def new_easy(self):
        """Easy subset: angle in [0..35), occlusion none, min dist 0.4m"""
        ed = self.new_by_angle_range(np.deg2rad(0), np.deg2rad(35))
        ed = ed.new_by_occlusion_none()
        ed.name = self.name + " easy"
        return ed

    def new_moderate(self):
        """Moderate subset: angle in [35..60), occlusion none or partial, min dist 0.4m"""
        ed = self.new_by_angle_range(np.deg2rad(0), np.deg2rad(60))
        ed = ed.new_by_occlusion_none_partial()
        # remove easy ones
        ed.df = ed.df[~((ed.df.gt_angle_from_zero < np.deg2rad(35)) & ((ed.df.occlusion_state == 'none') | (ed.df.occlusion_state == 'none-auto')))]
        ed.name = self.name + " mod"
        return ed

    def new_hard(self, is_full_occlusion=True):
        """Hard subset: angle in [60..inf) or <0.4m, occlusion all types"""
        ed = EvaluationData()
        if is_full_occlusion:
            ed.df = self.df[(self.df.gt_angle_from_zero >= np.deg2rad(60)) | (self.df.occlusion_state == 'full') | (self.df.occlusion_state == 'full-auto')]
            ed.name = self.name + " hard(full-occl)"
        else:
            ed.df = self.df[(self.df.gt_angle_from_zero >= np.deg2rad(60))]
            ed.name = self.name + " hard(partial-occl)"

        return ed
    
    def new_test_split(self):
        """Test split"""
        ed = EvaluationData()
        ed.df = self.df[self.df.subject.isin(self.test_subjects)]  # TODO
        ed.name = self.name + " test"
        return ed

    def new_trainval_split(self):
        """Trainval split"""
        ed = EvaluationData()
        ed.df = self.df[~self.df.subject.isin(self.test_subjects)]  # TODO
        ed.name = self.name + " trainval"
        return ed
        
    def get_angle_recalls(self, d=5, k=75):
        """deg!"""
        angles_deg = np.array(range(0, k-1, d))
        recalls = []
        for angle_deg in angles_deg:
            recall = self.new_by_angle_range(np.deg2rad(angle_deg), np.deg2rad(angle_deg+d)).get_recall()
            recalls.append(recall)
        
        return angles_deg, recalls

    def get_gt_count(self):
        return (~self.df.gt_x.isna()).count()

    def get_hypo_count(self):
        return (~self.df.hypo_x.isna()).count()

    def get_angle_gt_counts(self, d=5, k=75):
        bins = dict()
        for i in range(0, k-1, d):
            bins[i] = self.new_by_angle_range(np.deg2rad(i), np.deg2rad(i+d)).get_gt_count()

        angles, counts = zip(*[(k, v) for k, v in sorted(bins.items()) if not np.isnan(v)])
        angles = np.array(angles)
        counts = np.array(counts)
        return angles, counts

    def get_angle_hypo_counts(self, d=5, k=75):
        bins = dict()
        for i in range(0, k-1, d):
            bins[i] = self.new_by_angle_range(np.deg2rad(i), np.deg2rad(i+d)).get_hypo_count()

        angles, counts = zip(*[(k, v) for k, v in sorted(bins.items()) if not np.isnan(v)])
        angles = np.array(angles)
        counts = np.array(counts)
        return angles, counts

    def get_angle_maes(self, d=5, k=75, k_min=0):
        """deg!"""
        assert k_min % d == 0
        angles_deg = np.array(range(k_min, k-1, d))
        maes = []
        for angle_deg in angles_deg:
            mae = self.new_by_angle_range(np.deg2rad(angle_deg), np.deg2rad(angle_deg + d)).get_mae()
            maes.append(mae)
        
        maes = np.array(maes)
        return angles_deg, maes

    def get_angle_rpys(self, d=5, k=75):
        """deg!"""
        angles_deg = np.array(range(0, k-1, d))
        rpys_deg = []
        for angle_deg in angles_deg:
            rpy_deg = self.new_by_angle_range(np.deg2rad(angle_deg), np.deg2rad(angle_deg+d)).get_drpy()
            rpys_deg.append(rpy_deg)
        
        rpys_deg  = np.rad2deg(np.array(rpys_deg))
        return angles_deg, rpys_deg

    def get_angle_rolls(self, d=5, k=75):
        """deg!"""
        bins = dict()
        for i in range(0, k-1, d):
            bins[i] = self.new_by_roll_range(np.deg2rad(i), np.deg2rad(i+d)).get_drpy()
        
        angles, rpys = zip(*[(k,v) for k,v in sorted(bins.items()) if not np.any(np.isnan(v))])
        angles = np.array(angles)
        rpys  = np.rad2deg(np.array(rpys))
        return angles, rpys[:,0] # ROLL

    def get_angle_pitches(self, d=5, k=75):
        """deg!"""
        bins = dict()
        for i in range(0, k-1, d):
            bins[i] = self.new_by_pitch_range(np.deg2rad(i), np.deg2rad(i+d)).get_drpy()
        
        angles, rpys = zip(*[(k,v) for k,v in sorted(bins.items()) if not np.any(np.isnan(v))])
        angles = np.array(angles)
        rpys  = np.rad2deg(np.array(rpys))
        return angles, rpys[:,1] # PITCH

    def get_angle_yaws(self, d=5, k=75):
        """deg!"""
        bins = dict()
        for i in range(0, k-1, d):
            bins[i] = self.new_by_yaw_range(np.deg2rad(i), np.deg2rad(i+d)).get_drpy()
        
        angles, rpys = zip(*[(k,v) for k,v in sorted(bins.items()) if not np.any(np.isnan(v))])
        angles = np.array(angles)
        rpys  = np.rad2deg(np.array(rpys))
        return angles, rpys[:,2] # YAW

    def get_bmae(self, d=5, k=75, k_min=0):
        """deg!"""
        assert k_min % d == 0
        assert k % d == 0
        _, maes_deg = self.get_angle_maes(d, k, k_min=k_min)
        print(maes_deg)
        count = np.count_nonzero(np.isfinite(maes_deg))  # number on nonempty bins
        if count == 0:
            print("Warn: no valid MAEs when computing BMAE!")
            bmae_deg_invalid = np.nan
            return bmae_deg_invalid
        if count != ((k - k_min) // d):
            print("Warn: some empty MAEs when computing BMAE!")
        bmae_deg = np.nansum(maes_deg) / float(count)
        return bmae_deg


class Plotter:
    def __init__(self, subset_eds):
        """
        subset_eds: dict which maps from name to evaluation data objects
        """
        self.subset_eds = subset_eds

        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        # blue, red, green, violet, orange, lightblue, ...
        # colors = ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3', '#FF6692',
        #           '#B6E880', '#FF97FF', '#FECB52']
        # juggled a little
        # blue, green, violet, orange, red, ...
        self.colors = ['#636efa', '#00cc96', '#ab63fa', '#FFA15A', '#EF553B', '#19d3f3', '#FF6692',
                  '#B6E880', '#FF97FF', '#FECB52']

    def get_maes_figure(self, layout=None, binsize_deg=5, max_angle_deg=75):
       
        data = []
        binsize = binsize_deg

        for i, (name, ed) in enumerate(self.subset_eds.items()):
            x, y = ed.get_angle_maes(d=binsize, k=max_angle_deg)
            x = x + float(binsize)/2.0
            data.append(go.Scatter(x=x, y=y, name=name, mode='lines+markers', marker_color=self.colors[i]))

        if layout is None:
            layout = go.Layout(
                xaxis=dict(
                    title='angle from frontal (&deg;), binsize = %d&deg;' % binsize,
                    nticks=16, # or tickvals,
                    titlefont=dict(
                        family='serif',
                        size=35,
                    ),
                    tickfont=dict(
                        family='serif',
                        size=30
                    )

                ),
                yaxis=dict(
                    title='MAE<sub>R</sub> within bin (&deg;)',
                    titlefont=dict(
                        family='serif',
                        size=35,
                    ),
                    tickfont=dict(
                        family='serif',
                        size=30
                    ),
                    range=[0,40]
                ),
                margin=dict(l=80, r=0, t=10, b=85),
                legend=dict(
                    x=0.05,
                    y=0.95,
                    font=dict(
                        family='serif',
                        size=25,
                    ),
                    borderwidth=1
                )
            )
        fig = go.Figure(data=data, layout=layout)
        return fig
            
    def get_recalls_figure(self, layout=None, binsize_deg=5, max_angle_deg=75):
        data = []
        binsize = binsize_deg
        max_angle = max_angle_deg

        for i, (name, ed) in enumerate(self.subset_eds.items()):
            x, y = ed.get_angle_recalls(d=binsize, k=max_angle)
            x = x + float(binsize)/2.0
            data.append(go.Scatter(x=x, y=y, name=name, mode='lines+markers', marker_color=self.colors[i]))

        if layout is None:
            layout = go.Layout(
                xaxis=dict(
                    title='angle from frontal (&deg;), binsize = %d&deg;' % binsize,
                    nticks=16,
                    titlefont=dict(
                        family='serif',
                        size=35,
                    ),
                    tickfont=dict(
                        family='serif',
                        size=30
                    )
                ),
                yaxis=dict(
                    title='recall within bin',
                    titlefont=dict(
                        family='serif',
                        size=35,
                    ),
                    tickfont=dict(
                        family='serif',
                        size=30
                    ),
                    range=[-0.01,1.05]
                ),
                margin=dict(l=80, r=0, t=10, b=85),
                legend=dict(
                    x=0.87,
                    y=0.92,
            #         x=0.04,
            #         y=0.03,

                    font=dict(
                        family='serif',
                        size=25,
                    ),
                    borderwidth=1,
            #         bgcolor = 'rgba(255,255,255,0.3)'  #transparent bg
                )
            )
        fig = go.Figure(data=data, layout=layout)
        return fig

    def get_rpys_figure(self, layout=None, binsize_deg=5, max_angle_deg=75):
       
        # mae for RPY
        data = []
        binsize = binsize_deg

        for name, ed in self.subset_eds.items():
            x, y = ed.get_angle_rpys(d=binsize, k=max_angle_deg)
            x = x + float(binsize)/2.0
            data.append(go.Scatter(x=x, y=y[:, 0], name=name + ' r', mode='lines+markers'))
            data.append(go.Scatter(x=x, y=y[:, 1], name=name + ' p', mode='lines+markers'))
            data.append(go.Scatter(x=x, y=y[:, 2], name=name + ' y', mode='lines+markers'))

        if layout is None:
            layout = go.Layout(
                xaxis=dict(
                    title='angle from frontal (&deg;), binsize = %d&deg;' % binsize,
                    nticks=16, # or tickvals,
                    titlefont=dict(
                        family='serif',
                        size=35,
                    ),
                    tickfont=dict(
                        family='serif',
                        size=30
                    )

                ),
                yaxis=dict(
                    title='MAE<sub>R</sub> within bin (deg)',
                    titlefont=dict(
                        family='serif',
                        size=35,
                    ),
                    tickfont=dict(
                        family='serif',
                        size=30
                    ),
                    range=[-0.1,70]
                ),
                margin=dict(l=80, r=0, t=10, b=85),
                legend=dict(
                    x=0.05,
                    y=0.95,
                    font=dict(
                        family='serif',
                        size=25,
                    ),
                    borderwidth=1
                )
            )
        fig = go.Figure(data=data, layout=layout)
        return fig

    def get_counts_figure(self, binsize=5, max_angle=75):
        name = "# samples"
        # just take first ed (since data distribution is the same for all
        ed = next(iter(self.subset_eds.values()))
        bin_lefts, counts = ed.get_angle_gt_counts(d=binsize, k=max_angle)
        bin_centers = bin_lefts + float(binsize) / 2.0
        data = go.Bar(x=bin_centers, y=counts, name=name)
        layout = go.Layout(
            xaxis=dict(
                title='angle from frontal (&deg;), binsize = %d&deg;' % binsize,
                #         nticks=16, # or tickvals,
                titlefont=dict(
                    family='serif',
                    size=35,
                ),
                tickfont=dict(
                    family='serif',
                    size=30
                )
            ),
            yaxis=dict(
                title="# samples",
                titlefont=dict(
                    family='serif',
                    size=25,
                ),
                tickfont=dict(
                    family='serif',
                    size=30
                ),
            ),
            margin=dict(l=80, r=0, t=10, b=85),
            legend=dict(
                x=0.60,
                y=0.95,
                font=dict(
                    family='serif',
                    size=25,
                ),
                borderwidth=1
            )
        )
        fig = go.Figure(data=data, layout=layout)
        return fig
