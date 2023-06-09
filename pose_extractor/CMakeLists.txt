cmake_minimum_required(VERSION 3.10)
project(pose_extractor)

find_package(PythonInterp 3.5 REQUIRED)
find_package(PythonLibs 3.5 REQUIRED)

exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy; print(numpy.get_include())\""
             OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
             RETURN_VALUE NUMPY_NOT_FOUND
            )
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

set(OpenCV_DIR "/comp_robot/caohe/code/build")
find_package(OpenCV 4 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(target_name pose_extractor)
add_library(${target_name} SHARED wrapper.cpp
                                  src/extract_poses.hpp src/extract_poses.cpp
                                  src/human_pose.hpp src/human_pose.cpp
                                  src/peak.hpp src/peak.cpp)
target_include_directories(${target_name} PRIVATE src/ ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target_name} ${PYTHON_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(${target_name} PROPERTIES PREFIX "" OUTPUT_NAME "${target_name}")
if(WIN32)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".pyd")
elseif(APPLE)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".so")
endif()

