#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:35:39 2021

@author: sirius

deapth camera using with realsense2

"""

import pyrealsense2 as rs 
import numpy as np
import cv2

#configure depth ans color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


pipeline.start(config)

try:
    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #get depth scale from get_distance(x, y)
        depth = frames.get_depth_frame()
        distance = depth.get_distance(320, 240)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        print("distance is :", distance)

        key = cv2.waitKey(1)

        if key & 0xFF ==ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()