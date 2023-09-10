cmake_minimum_required(VERSION 3.22)
project(cparse)

set(CMAKE_CXX_STANDARD 14)

add_executable(cparse cparse.cpp)
target_link_libraries(cparse pcap)

add_executable(popets popets.cpp)
target_link_libraries(popets pcap)
