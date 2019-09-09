import numpy as np
import cv2

from dd_pose.image_decorator import ImageDecorator
from dd_pose.evaluation_helpers import T_headfrontal_camdriver
import transformations as tr


def get_dashboard(di, stamp):

    # canvas to draw stuff onto
    canvas = np.zeros((1024, 2048, 3), dtype=np.uint8)

    # camdriver>head
    T_camdriver_head = di.get_T_camdriver_head(stamp)

    # left driver image, head, landmarks, marker
    img, pcm = di.get_img_driver_left(stamp)
    img_color = np.dstack((img, img, img)) # bgr
    image_decorator = ImageDecorator(img_color, pcm)
    image_decorator.draw_axis(T_camdriver_head)
    # showimage(img_color)

    # put image on canvas
    if img is not None:
        canvas[0:1024,0:1024] = img_color

    # DOCU
    img, pcm = di.get_img_docu(stamp)
    if img is not None:
        image_decorator = ImageDecorator(img, pcm)
        T_camdriver_camdocu = di.get_T_camdriver_camdocu()

        if T_camdriver_camdocu is not None:
            T_camdocu_camdriver = np.linalg.inv(T_camdriver_camdocu)
            image_decorator.draw_axis(T_camdocu_camdriver)
            image_decorator.draw_axis(T_camdocu_camdriver.dot(T_camdriver_head))

        img = cv2.resize(img, None, fx=1.6, fy=1.6)
        h,w,_ = img.shape
        canvas[0:0+h,1024:1024+w] = img

    column1_u = 1024 + 10
    column1_v = 576 + 10

    column2_u = 1024 + 512 + 10
    column2_v = 576 + 10

    text_color = (255, 255, 255)

    font_scale = 1.5
    line_distance = 30

    cv2.putText(canvas, "humanhash: %s" % di.get_humanhash(), (column1_u, column1_v+1*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)
    cv2.putText(canvas, "subject:  %d" % di.get_subject(), (column1_u, column1_v+2*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)
    cv2.putText(canvas, "scenario: %d" % di.get_scenario(), (column1_u, column1_v+3*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)
    cv2.putText(canvas, "stamp:    %.3f" % (float(stamp)/1e9), (column1_u, column1_v+4*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)

    occlusion_state = di.get_occlusion_state(stamp)
    if occlusion_state is not None:
        cv2.putText(canvas, "occlusion state: %s" % occlusion_state, (column1_u, column1_v+6*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)

    stw = di.get_stw_angle(stamp)
    if stw is not None:
        stw_angle, stw_speed = stw
        cv2.putText(canvas, "stw angle: %.1f deg" % stw_angle, (column1_u, column1_v+8*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)
        cv2.putText(canvas, "stw speed: %.1f deg/s" % stw_speed, (column1_u, column1_v+9*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)

    gps = di.get_gps(stamp)
    if gps is not None:
        cv2.putText(canvas, "gps lat: %02.4f" % gps['latitude'], (column2_u, column2_v+3*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)
        cv2.putText(canvas, "gps lon: %02.4f" % gps['longitude'], (column2_u, column2_v+4*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)
        cv2.putText(canvas, "gps alt: %02.0f m" % gps['altitude'], (column2_u, column2_v+5*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)

    velocity = di.get_velocity(stamp)
    if velocity is not None:
        velocity_stamp, velocity = velocity
        cv2.putText(canvas, "vel:     %.1f m/s" % velocity, (column2_u, column2_v+7*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)

    yaw_rate = di.get_yaw_rate(stamp)
    if yaw_rate is not None:
        yaw_rate_stamp, yaw_rate = yaw_rate
        cv2.putText(canvas, "yaw rate: %.1f m/s2" % yaw_rate, (column2_u, column2_v+8*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)

    heading = di.get_heading(stamp)
    if heading is not None:
        heading_stamp, heading = heading
        cv2.putText(canvas, "heading:    %03.1f deg" % heading, (column2_u, column2_v+9*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)

    T_headfrontal_head = np.dot(T_headfrontal_camdriver, T_camdriver_head)
    roll, pitch, yaw = tr.euler_from_matrix(T_headfrontal_head, 'sxyz')
    cv2.putText(canvas, "rpy: (%.1f, %.1f, %.1f) deg" % tuple(np.rad2deg((roll, pitch, yaw)).tolist()), (column2_u, column2_v+10*line_distance), cv2.FONT_HERSHEY_PLAIN, font_scale, text_color, 1)

    return canvas
