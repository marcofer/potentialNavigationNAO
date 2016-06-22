cmake_minimum_required(VERSION 2.8)
project(potentialNavigationNAO C CXX)

find_package( OpenCV 2.4.11 EXACT REQUIRED )
find_package(qibuild)

#message("${OpenCV_DIR}")

set(CMAKE_INSTALL_PREFIX /usr/local)
set(CMAKE_PREFIX_PATH /usr/local)


SET (CMAKE_CXX_FLAGS                "-std=c++0x")

# Create a list of source files
set(_srcs
    potentialNavigationNAO.cpp
    cJoystick.cpp
    of_driving.cpp
    parallel_process.cpp
    main.cpp
    utils.cpp)

# Create a executable named potentialNavigationNAO
# with the source file: main.cpp
qi_create_bin(NAO-VPN-Module_create ${_srcs})
qi_use_lib(NAO-VPN-Module_create  ALCOMMON OPENCV2_CORE OPENCV2_HIGHGUI OPENCV2_IMGPROC OPENCV2_GPU OPENCV2_VIDEO)


