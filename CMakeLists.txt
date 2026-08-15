cmake_minimum_required(VERSION 3.15)
project(tbbsamplesort CXX)

find_package(TBB REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} onetbb::onetbb)
