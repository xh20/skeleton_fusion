cmake_minimum_required(VERSION 3.1)

project(skeleton_fusion)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/include)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(SRC_FILES src/main.cpp)

set(matplotlib_SOURCE_DIR "/home/hao/Downloads/matplotlib-cpp")
set(OsqpEigen_DIR "/home/geriatronics/hao/osqp-eigen/build/")

include_directories(SYSTEM ${matplotlib_SOURCE_DIR})

#target_link_libraries(train_FDDL stdc++fs)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED )
find_package(GFlags)
find_package(Glog)
find_package(apriltag REQUIRED)
find_package(Eigen3)
#find_package(OsqpEigen REQUIRED)
find_package(OpenPose REQUIRED)
find_package(Threads)
# find_package(matplotlib_cpp)

find_library(GLOG_LIBRARY_RELEASE glog HINTS ${FIND_LIB_PREFIX}/caffe3rdparty/lib)
find_library(GLOG_LIBRARY_DEBUG glogd HINTS ${FIND_LIB_PREFIX}/caffe3rdparty/lib)
find_library(apriltag_LIBRARY1 NAMES libapriltag.so PATHS "/usr/local/lib" )
find_library(apriltag_LIBRARY2 NAMES libapriltag.so.3 PATHS "/usr/local/lib" )
find_library(apriltag_LIBRARY2 NAMES libapriltag.so.3.4.1 PATHS "/usr/local/lib" )
set(apriltag_LIBS ${apriltag_LIBRARY1} ${apriltag_LIBRARY2} ) # ${apriltag_LIBRARY3}

if(${Python3_FOUND})
	message("-- Found python3")
endif()

if(${PythonLibs_FOUND})
	message("-- Found PythonLibs")
endif()

if(${apriltag_FOUND})
	message("-- Found apriltag ")
endif()

include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS} ${apriltag_INCLUDE_DIRS} ${OpenPose_INCLUDE_DIRS}
		${GFLAGS_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${GLOG_INCLUDE_DIR})

message(STATUS "apriltag_include dir:${apriltag_INCLUDE_DIRS}")
message(STATUS "apriltag_dir:${apriltag_DIR}")
message(STATUS "OpenPose_dir:${OpenPose_DIR}")

# message(STATUS "apriltag_libs:${apriltag_LIBS}")

add_executable(main ${SRC_FILES})
add_executable(debug src/debug.cpp)

#target_link_libraries(main ${GFLAGS_LIBRARY}  ${OpenCV_LIBS} ${OpenPose_LIBS}  ${apriltag_LIBS} ${PYTHON_LIBRARIES}
#		${realsense2_LIBRARY} Python3::NumPy stdc++fs OsqpEigen::OsqpEigen)
#
#target_link_libraries(debug ${GFLAGS_LIBRARY}  ${OpenCV_LIBS} ${OpenPose_LIBS}  ${apriltag_LIBS} ${PYTHON_LIBRARIES}
#		${realsense2_LIBRARY} Python3::NumPy stdc++fs OsqpEigen::OsqpEigen)

target_link_libraries(main ${GFLAGS_LIBRARY}  ${OpenCV_LIBS} ${OpenPose_LIBS}  ${apriltag_LIBS} ${PYTHON_LIBRARIES}
		${realsense2_LIBRARY} Python3::NumPy stdc++fs)

target_link_libraries(debug ${GFLAGS_LIBRARY}  ${OpenCV_LIBS} ${OpenPose_LIBS}  ${apriltag_LIBS} ${PYTHON_LIBRARIES}
		${realsense2_LIBRARY} Python3::NumPy stdc++fs)