cmake_minimum_required(VERSION 3.22.1)
project("llama-engine" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp)
if (NOT EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp submodule not found at ${LLAMA_DIR}")
endif()

# Build llama.cpp static targets
add_subdirectory(${LLAMA_DIR} build-llama EXCLUDE_FROM_ALL)

# JNI bridge library
add_library(llama_engine SHARED
    src/main/cpp/llama_jni.cpp
)

target_include_directories(llama_engine PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/src
)

target_compile_options(llama_engine PRIVATE
    -O3
    -march=armv8.4-a+dotprod+i8mm+fp16
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)

target_link_options(llama_engine PRIVATE
    -Wl,--exclude-libs,ALL
)

target_link_libraries(llama_engine PRIVATE
    llama
    android
    log
)
