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

set(INSTALL_TARGETS astar_planner smac_planner dwa_controller)

# The three C and C++ baselines, hosted in a node so RACE_FIELD=bench-cpp can
# put them on a robot. They are somebody else's code, fetched rather than
# vendored -- bench/fetch_baselines.sh -- so this target exists only when the
# fetch has happened, the same condition bench/run.sh applies to the timing
# comparison. Without it the rest of the package still builds; with it,
# bench/baseline_*.cpp compile once here and once there, off the same step_*
# the figure's traces call.
#
# goktug97's is C rather than C++ and g++ rejects its implicit void*
# conversions, so it is compiled as C, which is also how its own build does
# it -- hence the C language, enabled only when the sources are there.
# The colcon workspace reaches this package through a symlink --
# /root/ros2_ws/src/reactive_nav_cpp -> <repo>/cpp -- so "../bench" off
# CMAKE_CURRENT_SOURCE_DIR is /root/ros2_ws/src/bench and does not exist.
# REALPATH on a path that already ends in ".." does not fix it either: CMake
# collapses the ".." lexically first and hands realpath() a string the
# symlink has already been taken out of, which resolves to /root/ros2_ws/src
# and silently skipped this whole target. Resolve the package directory
# first, then take its parent.
get_filename_component(PKG_REAL ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
get_filename_component(REPO ${PKG_REAL} DIRECTORY)
set(BENCH ${REPO}/bench)
if(EXISTS ${BENCH}/.third_party/dwa.c)
  enable_language(C)
  add_library(dwa_goktug_c STATIC ${BENCH}/.third_party/dwa.c)
  target_include_directories(dwa_goktug_c PUBLIC ${BENCH}/.third_party)
  set_target_properties(dwa_goktug_c PROPERTIES
    C_STANDARD 11 POSITION_INDEPENDENT_CODE ON)
  # Their code, their warnings, same as the three wrappers below: its min/max
  # macros are GNU braced-groups and -Wpedantic has an opinion about them.
  target_compile_options(dwa_goktug_c PRIVATE -w)

  add_executable(baseline_controller
    src/baseline_controller.cpp
    ${BENCH}/baseline_cpprobotics.cpp
    ${BENCH}/baseline_goktug.cpp
    ${BENCH}/baseline_amslabtech.cpp)
  target_include_directories(baseline_controller PRIVATE
    ${BENCH} ${BENCH}/.third_party)
  target_link_libraries(baseline_controller dwa_goktug_c)
  ament_target_dependencies(baseline_controller ${DEPS})
  # Their code, their warnings: -Wall -Wextra -Wpedantic is this package's
  # standard and three fetched planners are not this package's to tidy.
  set_source_files_properties(
    ${BENCH}/baseline_cpprobotics.cpp
    ${BENCH}/baseline_goktug.cpp
    ${BENCH}/baseline_amslabtech.cpp
    PROPERTIES COMPILE_OPTIONS "-w")
  list(APPEND INSTALL_TARGETS baseline_controller)
else()
  message(STATUS
    "bench/.third_party is empty -- run bench/fetch_baselines.sh for the "
    "bench-cpp race field; skipping baseline_controller")
endif()

install(TARGETS
  ${INSTALL_TARGETS}
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
