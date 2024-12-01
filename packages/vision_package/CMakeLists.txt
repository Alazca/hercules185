cmake_minimum_required(VERSION 3.0.2)
project(vision_package)

# Find required packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  std_msgs
)

# Declare the catkin package
catkin_package()

# Specify the Python nodes to be installed
catkin_install_python(PROGRAMS
  src/boundingBoxCamera.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

