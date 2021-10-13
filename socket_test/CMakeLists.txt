cmake_minimum_required(VERSION 2.8)
project(tcp_client)

# SET(CMAKE_BUILD_TYPE Debug)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()


MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native -pthread")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(OpenCV 3.4.6 REQUIRED CONFIG PATHS "/usr/local/opencv-3.4.6" NO_DEFAULT_PATH)

add_executable(client client.cpp)
target_link_libraries(client ${OpenCV_LIBS})
