cmake_minimum_required(VERSION 3.9)

project(lightflow VERSION 1.0.0 DESCRIPTION "Lightweight DL framework")

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Ofast -fopenmp")
add_compile_options(-Wall -Werror)

set(lightflow_src
    src/config.cc
    src/diff.cc
    src/loss.cc
    src/nn.cc
    src/optimizer.cc
    src/tensor.cc
    include/lightflow/config.h
    include/lightflow/loss.h
    include/lightflow/nn.h
    include/lightflow/optimizer.h
    include/lightflow/tensor.h
    include/lightflow/diff.h
)

add_library(${PROJECT_NAME} SHARED ${lightflow_src})

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "include/lightflow/config.h;include/lightflow/loss.h;include/lightflow/nn.h;include/lightflow/optimizer.h;include/lightflow/tensor.h;include/lightflow/diff.h;")

target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PRIVATE include/lightflow)

configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lightflow
)

install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
