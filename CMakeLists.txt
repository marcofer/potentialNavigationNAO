cmake_minimum_required(VERSION 2.8)
project(potentialNavigationNAO C CXX)

find_package( OpenCV REQUIRED )
find_package(qibuild)

#add_definitions("-pthread")
#SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11 -g")

# Create a list of source files
set(_srcs
    potentialNavigationNAO.cpp
    cJoystick.cpp
    of_driving.cpp
    parallel_process.cpp
    main.cpp)

# Create a executable named potentialNavigationNAO
# with the source file: main.cpp
qi_create_bin(potentialNavigationNAO ${_srcs})
qi_use_lib(potentialNavigationNAO  ALCOMMON OPENCV2_CORE OPENCV2_HIGHGUI OPENCV2_IMGPROC OPENCV2_GPU OPENCV2_VIDEO)
