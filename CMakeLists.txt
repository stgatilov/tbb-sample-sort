cmake_minimum_required(VERSION 3.15)
project(tbbsamplesort CXX)

find_package(TBB REQUIRED)

if(MSVC)
    add_compile_options("$<$<CONFIG:Release>:/Zi>")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " /OPT:REF /DEBUG")
elseif(UNIX)
	add_compile_options("$<$<CONFIG:Release>:-g>")
endif()

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/tbbss.cpp
    src/tbbss.h
    src/pcg_basic.h
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_link_libraries(${PROJECT_NAME} onetbb::onetbb)
