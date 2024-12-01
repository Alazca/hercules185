
#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun vision_package raw_boundingBox.py

# wait for app to end
dt-launchfile-join
