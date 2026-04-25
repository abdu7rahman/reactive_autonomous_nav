cmake_minimum_required(VERSION 3.8)
project(reactive_nav_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(DEPS
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(astar_planner   src/astar_planner.cpp)
add_executable(smac_planner    src/smac_planner.cpp)
add_executable(dwa_controller  src/dwa_controller.cpp)

ament_target_dependencies(astar_planner   ${DEPS})
ament_target_dependencies(smac_planner    ${DEPS})
ament_target_dependencies(dwa_controller  ${DEPS})

install(TARGETS
  astar_planner
  smac_planner
  dwa_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
