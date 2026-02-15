cmake_minimum_required(VERSION 3.22.1)

project(storm_agimus_bridge LANGUAGES CXX)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/storm_mppi_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

include(CTest)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_auto_find_test_dependencies()
endif()

ament_auto_package(INSTALL_TO_SHARE config)

