cmake_minimum_required(VERSION 3.19)
project(mnn_scrfd)

set(CMAKE_CXX_STANDARD 14)
set(MNN_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/MNN/darwin/lib)

find_package(OpenCV REQUIRED)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/MNN/darwin/include)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${MNN_LIBS})

add_executable(mnn_scrfd main.cpp SCRFD.cpp)

target_link_libraries(mnn_scrfd ${OpenCV_LIBS} mnn)