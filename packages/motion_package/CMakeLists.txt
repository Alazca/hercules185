cmake_minimum_required(VERSION 3.0.2)
project(motion_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  duckietown_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/personFollower.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
