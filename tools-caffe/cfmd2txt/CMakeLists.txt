cmake_minimum_required (VERSION 2.8)

project (to_txt_model)

add_definitions(-DCPU_ONLY=1)

add_executable(cfmd2txt main.cpp)

include_directories (
  /home/gjs/work/caffe/build/install/include
  /usr/local/include
  /usr/include
)

target_link_libraries(cfmd2txt
    /home/gjs/work/caffe/build/lib/libcaffe.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
    )
