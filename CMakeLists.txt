cmake_minimum_required(VERSION 3.15)
project(tbbsamplesort CXX)

option(TBBSS_BUILD_PLAYGROUND "Build playground executables to generate some data and run sort on it" ON)
option(TBBSS_BUILD_TESTS "Build tests: requires more packages" ON)
set(TBBSS_SANITIZE "" CACHE STRING "Build with -fsanitize=XYZ")

# make sure Release build has debug info and is not slowed down because of it
if(MSVC)
    add_compile_options("$<$<CONFIG:Release>:/Zi>")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " /OPT:REF /DEBUG")
elseif(UNIX)
	add_compile_options("$<$<CONFIG:Release>:-g>")
endif()

# check for all the annoying warnings
if(MSVC)
    add_compile_options("/W4")
    add_compile_definitions("_CRT_SECURE_NO_WARNINGS")
elseif(UNIX)
    add_compile_options("-Wall")
endif()

# enable sanitizer
if (NOT TBBSS_SANITIZE STREQUAL "")
    message("Added: -fsanitize=${TBBSS_SANITIZE}")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

find_package(TBB REQUIRED)
add_library(tbbsamplesort
    src/tbbss.cpp
    src/tbbss.h
)
set_property(TARGET tbbsamplesort PROPERTY CXX_STANDARD 17)
target_link_libraries(tbbsamplesort PUBLIC onetbb::onetbb)
target_include_directories(tbbsamplesort INTERFACE src)

if (TBBSS_BUILD_PLAYGROUND)
    add_executable(tbbss_playground_gen
        src/playground_gen.cpp
    )
    set_property(TARGET tbbss_playground_gen PROPERTY CXX_STANDARD 11)

    add_executable(tbbss_playground_run
        src/playground_run.cpp
    )
    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
    target_link_libraries(tbbss_playground_run PRIVATE tbbsamplesort)
endif()

if (TBBSS_BUILD_TESTS)
    find_package(doctest REQUIRED)
    find_package(fmt REQUIRED)
    add_executable(tbbss_tests
        src/tests_common.h
        src/tests_main.cpp
        src/tests_element.cpp
        src/tests_allocation.cpp
        src/tests_framework.cpp
    )
    set_property(TARGET tbbss_tests PROPERTY CXX_STANDARD 17)
    target_link_libraries(tbbss_tests PRIVATE tbbsamplesort doctest::doctest fmt::fmt)
    target_compile_definitions(tbbss_tests PRIVATE
        DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS
    )
endif()
