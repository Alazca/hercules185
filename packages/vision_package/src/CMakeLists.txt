find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
)

catkin_install_python(PROGRAMS 
  src/boundingBoxCamera.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

