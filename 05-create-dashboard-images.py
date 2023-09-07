#!/usr/bin/env python

from __future__ import print_function
import os
import sys

from dd_pose.dataset import Dataset
from dd_pose.dataset_item import DatasetItem
from dd_pose.visualization_helpers import get_dashboard
from dd_pose.jupyter_helpers import showimage

import cv2
from multiprocessing import Pool


d = Dataset()
subject, scenario, humanhash = sys.argv[1:4]
subject = int(subject)
scenario = int(scenario)
print("Subject:   %02d" % subject)
print("Scenario:  %02d" % scenario)
print("Humanhash: %s"   % humanhash)

di_dict = d.get(subject, scenario, humanhash)
di = DatasetItem(di_dict)
print(di)

dashboard_images_dir = os.path.join(os.environ['DD_POSE_DATA_ROOT_DIR'], '02-dashboard-images', 'subject-%02d' % di.get_subject(), 'scenario-%02d' % di.get_scenario(), di.get_humanhash())

if not os.path.exists(dashboard_images_dir):
    os.makedirs(dashboard_images_dir)

def create_and_write_dashboard(stamp):
    print(stamp)
    dashboard_image = get_dashboard(di, stamp)
    cv2.imwrite(os.path.join(dashboard_images_dir, '%ld.png' % stamp), dashboard_image)

# create dashboards in parallel
p = Pool(8)
p.map(create_and_write_dashboard, di.get_stamps())
p.close()
p.join()

print("Done!")
