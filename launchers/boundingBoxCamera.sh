#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
# launch publisher
dt-exec rosrun vision_package boundingBoxCamera.py

# wait for app to end
dt-launchfile-join
