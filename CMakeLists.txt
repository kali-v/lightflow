cmake_minimum_required(VERSION 3.9)

project(lightflow VERSION 1.0.0 DESCRIPTION "Lightweight DL framework")

include(CTest)
include(GNUInstallDirs)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Ofast -funroll-loops -fopenmp")
add_compile_options(-Wall -Werror)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    find_package(CUDA)
    include_directories("${CUDA_INCLUDE_DIRS}")
    set(CUDA_NVCC_FLAGS "--compiler-options -fPIC -O4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLF_CUDA_AVAIL -lcudart")
    cuda_compile(CUDA_O src/accel/cuda.cu)
endif()

set(lightflow_src
    src/config.cc
    src/diff.cc
    src/loss.cc
    src/nn.cc
    src/optimizer.cc
    src/tensor.cc
    src/accel/avx.cc
    src/accel/cpu.cc
    ${CUDA_O}
    include/lightflow/config.h
    include/lightflow/loss.h
    include/lightflow/nn.h
    include/lightflow/optimizer.h
    include/lightflow/tensor.h
    include/lightflow/diff.h
    include/lightflow/accel/avx.h
    include/lightflow/accel/cpu.h
    include/lightflow/accel/cuda.cuh
)

add_library(${PROJECT_NAME} SHARED ${lightflow_src})

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "include/lightflow/config.h;include/lightflow/loss.h;include/lightflow/nn.h;include/lightflow/optimizer.h;include/lightflow/tensor.h;include/lightflow/diff.h;")
set_target_properties(${PROJECT_NAME} PROPERTIES PRIVATE_HEADER "include/lightflow/accel/cpu.h;include/lightflow/accel/avx.h;include/lightflow/accel/cuda.cuh")

target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PRIVATE include/lightflow)
target_include_directories(${PROJECT_NAME} PRIVATE include/lightflow/accel)

configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lightflow
	PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lightflow/accel
)

install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
