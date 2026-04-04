cmake_minimum_required(VERSION 3.8)
project(eight_neighboring_regions)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 1. 摄像头节点
add_executable(camera_node src/camera_node.cpp)
ament_target_dependencies(camera_node rclcpp sensor_msgs cv_bridge OpenCV)

# 2. 八邻域核心算法节点
add_executable(eight_nav_node src/eight_nav_node.cpp)
ament_target_dependencies(eight_nav_node rclcpp sensor_msgs geometry_msgs std_msgs cv_bridge OpenCV)

# 3. 串口通信节点
add_executable(serial_bridge_node src/serial_bridge_node.cpp)
ament_target_dependencies(serial_bridge_node rclcpp geometry_msgs)

# 安装目标
install(TARGETS 
  camera_node 
  eight_nav_node 
  serial_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装 launch 文件夹
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()