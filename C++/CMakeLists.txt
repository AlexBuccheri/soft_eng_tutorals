cmake_minimum_required(VERSION 3.20)

#[==============================================================================================[
#                                    Basic project definition                                   #
]==============================================================================================]

project(impres)

# Define language standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

#[==============================================================================================[
#                                            Options                                            #
]==============================================================================================]

option(ENABLE_UNITTESTS "Build with unit test support" ON)


#[==============================================================================================[
#                                      Main Targets                                            #
]==============================================================================================]
# Library
add_library(algorithms_lib)
set_target_properties(algorithms_lib PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME qclib
        OUTPUT_NAME qclib
)

# Executable N/A

#[==============================================================================================[
#                                      External Libraries                                       #
]==============================================================================================]
find_package(Armadillo 10.0.0 REQUIRED)
message("-- Armadillo library found")
message("     ARMADILLO_VERSION      : " ${ARMADILLO_VERSION_STRING})
message("     ARMADILLO_LIBRARIES    : " ${ARMADILLO_LIBRARIES})
message("     ARMADILLO_INCLUDE_DIRS : " ${ARMADILLO_INCLUDE_DIRS})

# Ideally wanted to link to Armadillo::armadillo but this wasn't found, suggesting:
# 1. The version of Armadillo installed does not fully support modern CMake target conventions.
# 2. The ArmadilloConfig.cmake or FindArmadillo.cmake provided by the package lacks proper target definitions.

# Link external libs to library target
target_link_libraries(algorithms_lib PUBLIC ${ARMADILLO_LIBRARIES})
include_directories(${ARMADILLO_INCLUDE_DIRS})

if (ENABLE_UNITTESTS)
    # https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md
    find_package(Catch2 3 REQUIRED)

    message("-- Catch2 library found")
    get_target_property(CATCH2_INCLUDE_DIRS Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
    message("     CATCH2_INCLUDE_DIRS : ${CATCH2_INCLUDE_DIRS}")

    get_target_property(CATCH2_LIBS Catch2::Catch2 LOCATION)
    message("     CATCH2_LIBS        : ${CATCH2_LIBS}")
endif ()

# Source location
add_subdirectory(src)

#[==============================================================================================[
#                                         Utilities                                            #
]==============================================================================================]

# Current disabled because
# a) throws up alot of stupid errors with my code
# b) Throws up errors in arma and catch, which I do not care about (maybe disabling headers will avoid this)
# Run clang tidy on cpp and headers
#set(CMAKE_CXX_CLANG_TIDY
#        clang-tidy;
#        -header-filter=.;
#        -checks=*;)

#[==============================================================================================[
#                                         Tests                                                #
]==============================================================================================]

# -------------------------------------
# Unit Tests
# -------------------------------------
# These tests can use the Catch2-provided main
add_executable(UnitTests)
set_target_properties(UnitTests PROPERTIES
        OUTPUT_NAME unit_tests_exe
)

target_link_libraries(UnitTests PRIVATE algorithms_lib Catch2::Catch2WithMain)

# Make the project's headers visible to the tests
target_include_directories(UnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(UnitTests PRIVATE ${ARMADILLO_INCLUDE_DIRS})

# Unit test source location
add_subdirectory(unit_tests)

# Enable CTest
include(CTest)
include(Catch)
catch_discover_tests(UnitTests)
