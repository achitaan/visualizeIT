cmake_minimum_required(VERSION 3.20)
project(visualizeit_pose VERSION 0.2.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(visualizeit_pose SHARED
  src/pose.cpp
  src/raster.cpp
)

target_include_directories(visualizeit_pose PUBLIC include)
target_compile_features(visualizeit_pose PUBLIC cxx_std_17)
target_compile_definitions(visualizeit_pose PRIVATE VI_SHARED)

set_target_properties(visualizeit_pose PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(visualizeit_pose PRIVATE
    -Wall -Wextra
    $<$<NOT:$<CONFIG:Debug>>:-O3>
  )
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(visualizeit_pose PRIVATE
      $<$<NOT:$<CONFIG:Debug>>:-march=native>
    )
  endif()
endif()

include(CTest)
if(BUILD_TESTING AND NOT ANDROID AND NOT IOS)
  add_executable(visualizeit_pose_tests tests/pose_tests.cpp)
  target_link_libraries(visualizeit_pose_tests PRIVATE visualizeit_pose)
  set_target_properties(visualizeit_pose_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_test(NAME visualizeit_pose_tests COMMAND visualizeit_pose_tests)
  add_custom_target(pose_tests DEPENDS visualizeit_pose_tests)

  add_executable(visualizeit_raster_tests tests/raster_tests.cpp)
  target_link_libraries(visualizeit_raster_tests PRIVATE visualizeit_pose)
  set_target_properties(visualizeit_raster_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(visualizeit_raster_tests PRIVATE -Wall -Wextra -Werror)
  endif()
  add_test(NAME visualizeit_raster_tests COMMAND visualizeit_raster_tests)
endif()
