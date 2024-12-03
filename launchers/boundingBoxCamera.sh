#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun vision_package boundingBoxCamera.py

# wait for app to end
dt-launchfile-join
