#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun motion_package personFollower.py

# wait for app to end
dt-launchfile-join
