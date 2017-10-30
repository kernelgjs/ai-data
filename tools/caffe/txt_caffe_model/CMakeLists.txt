cmake_minimum_required (VERSION 2.8)

project (to_txt_model)

add_definitions(-DCPU_ONLY=1)

add_executable(to_txt_model main.cpp)

include_directories (
  /home/gjs/work/caffe/include
  /home/gjs/work/caffe/build/src
  /usr/local/include
  # /usr/local/cuda/include
  /usr/include
)

target_link_libraries(to_txt_model
    /home/gjs/work/caffe/build/lib/libcaffe.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
    )
