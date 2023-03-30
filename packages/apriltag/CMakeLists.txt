cmake_minimum_required(VERSION 2.8.3)
project(apriltag)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  # duckietown_msgs # Every duckietown packages should use this.
  sensor_msgs
  std_msgs
  geometry_msgs
  image_geometry
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
