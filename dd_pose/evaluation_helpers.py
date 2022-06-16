import numpy as np
import zipfile

import os
import json
import pandas as pd
import transformations as tr
from multiprocess import Pool

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
        
        with self.zf.open(self.predictions_file) as fp:
            try:
                self.predictions = StampedTransforms(fp)
            except ValueError as e:
                e.message = 'File %s is malformed json' % self.predictions_file
                raise e
        
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
    def __init__(self):
        self.df = pd.DataFrame()
        self.df.index.name = 'stamp'
        self.name = ""
        
    def load(self, di_dict, predictor):
        di = DatasetItem(di_dict)
        self.df['subject'] = pd.Series(data=di.get_subject(), index=di.get_stamps())
        self.df['scenario'] = di.get_scenario()
        self.df['humanhash'] = di.get_humanhash()
        
        for stamp in di.get_stamps():
            T_camdriver_head = di.get_T_camdriver_head(stamp)
            
            assert T_camdriver_head is not None
            
            T_headfrontal_head = T_headfrontal_camdriver.dot(T_camdriver_head)
            self.df.at[stamp, 'gt_roll'], self.df.at[stamp, 'gt_pitch'], self.df.at[stamp, 'gt_yaw'] = tr.euler_from_matrix(T_headfrontal_head)
            self.df.at[stamp, 'gt_x'], self.df.at[stamp, 'gt_y'], self.df.at[stamp, 'gt_z'] = T_camdriver_head[0:3,3]
            
            gt_angle_from_zero, _, _ = tr.rotation_from_matrix(T_headfrontal_head)
            self.df.at[stamp, 'gt_angle_from_zero'] = abs(gt_angle_from_zero)

            self.df.at[stamp, 'occlusion_state'] = di.get_occlusion_state(stamp)
            
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

                angle_difference, _, _ = tr.rotation_from_matrix(tr.inverse_matrix(T_headfrontal_head).dot(hypo_T_headfrontal_head))
                self.df.at[stamp, 'angle_diff'] = abs(angle_difference)

                self.df.at[stamp, 'hypo_x'], self.df.at[stamp, 'hypo_y'], self.df.at[stamp, 'hypo_z'] = predictor.get_t_camdriver_head(stamp)

#                 print gt_angle_from_zero, angle_difference, np.rad2deg(angle_difference), position_difference


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
        return abs((self.df.hypo_x - self.df.gt_x)).mean()

    def get_dy(self):
        return abs((self.df.hypo_y - self.df.gt_y)).mean()

    def get_dz(self):
        return abs((self.df.hypo_z - self.df.gt_z)).mean()

    def get_dxyz(self):
        """
        Get mean absoulte L2 distance.
        """    
        return abs(self.df.pos_diff).mean()
    
    def get_recall(self):
        """
        Get recall, i.e. ratio of available predictions and ground truth measurements.
        """
        n_gt =  self.df.gt_x.count()
        n_pos = self.df[~self.df.gt_x.isna()].hypo_x.count()

        if n_gt > 0:
            recall = float(n_pos)/n_gt
        else:
            recall = np.nan
        return recall
    
    def get_drpy(self):
        valid_rows = ~self.df.hypo_roll.isna()
        # rad
        return np.abs(angle_difference(self.df[['gt_roll', 'gt_pitch', 'gt_yaw']][valid_rows].values,
                                       self.df[['hypo_roll', 'hypo_pitch', 'hypo_yaw']][valid_rows].values)).mean(axis=0)
    
    def get_mae(self):
        mae = self.df.angle_diff.mean()
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

    def new_hard(self):
        """Hard subset: angle in [60..inf) or <0.4m, occlusion all types"""
        ed = EvaluationData()
        ed.df = self.df[(self.df.gt_angle_from_zero >= np.deg2rad(60)) | (self.df.occlusion_state == 'full') | (self.df.occlusion_state == 'full-auto')]
        ed.name = self.name + " hard"
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
        bins = dict()
        for i in range(0, k-1, d):
            bins[i] = self.new_by_angle_range(np.deg2rad(i), np.deg2rad(i+d)).get_recall()
        
        angles, recalls = zip(*[(k,v) for k,v in sorted(bins.items()) if not np.isnan(v)])
        angles = np.array(angles)
        return angles, recalls

    def get_angle_maes(self, d=5, k=75):
        """deg!"""
        bins = dict()
        for i in range(0, k-1, d):
            bins[i] = self.new_by_angle_range(np.deg2rad(i), np.deg2rad(i+d)).get_mae()
        
        angles, maes = zip(*[(k,v) for k,v in sorted(bins.items()) if not np.isnan(v)])
        angles = np.array(angles)
        maes = np.rad2deg(np.array(maes))
        return angles, maes

    def get_angle_rpys(self, d=5, k=75):
        """deg!"""
        bins = dict()
        for i in range(0, k-1, d):
            bins[i] = self.new_by_angle_range(np.deg2rad(i), np.deg2rad(i+d)).get_drpy()
        
        angles, rpys = zip(*[(k,v) for k,v in sorted(bins.items()) if not np.any(np.isnan(v))])
        angles = np.array(angles)
        rpys  = np.rad2deg(np.array(rpys))
        return angles, rpys

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

    def get_bmae(self, d=5, k=75):
        """deg!"""
        _, maes_deg = self.get_angle_maes(d, k)
        count = sum(not np.isnan(mae) for mae in maes_deg)  # number on nonempty bins
        if count != (k/d):
            print("Warn: empty MAEs when computing BMAE!")
        bmae = 1.0/float(count) * sum(maes_deg)
        return bmae


class Plotter:
    def __init__(self, subset_eds):
        """
        subset_eds: dict which maps from name to evaluation data objects
        """
        self.subset_eds = subset_eds

        
    def get_maes_figure(self):
       
        data = []
        binsize = 5

        for name, ed in self.subset_eds.items():
            x, y = ed.get_angle_maes(d=binsize)
            x = x + float(binsize)/2.0
            data.append(go.Scatter(x=x, y=y, name=name))

        layout = go.Layout(
            xaxis=dict(
                title='angle from frontal (deg), binsize = %d deg' % binsize,
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
                title='MAE within bin (deg)',
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
                    size=30,
                ),
                borderwidth=1
            )
        )
        fig = go.Figure(data=data, layout=layout)
        return fig
            
        
    def get_recalls_figure(self):
        data = []
        binsize = 5

        for name, ed in self.subset_eds.items():
            x, y = ed.get_angle_recalls(d=binsize)
            x = x + float(binsize)/2.0
            data.append(go.Scatter(x=x, y=y, name=name))

        layout = go.Layout(
            xaxis=dict(
                title='angle from frontal (deg), binsize = %d deg' % binsize,
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


    def get_rpys_figure(self):
       
        # mae for RPY
        data = []
        binsize = 5

        for name, ed in self.subset_eds.items():
            x, y = ed.get_angle_rpys(d=binsize)
            x = x + float(binsize)/2.0
            data.append(go.Scatter(x=x, y=y[:,0], name=name + ' roll'))
            data.append(go.Scatter(x=x, y=y[:,1], name=name + ' pitch'))
            data.append(go.Scatter(x=x, y=y[:,2], name=name + ' yaw'))


        layout = go.Layout(
            xaxis=dict(
                title='angle from frontal (deg), binsize = %d deg' % binsize,
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
                title='MAE within bin (deg)',
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
                    size=30,
                ),
                borderwidth=1
            )
        )
        fig = go.Figure(data=data, layout=layout)
        return fig
