cmake_minimum_required(VERSION 3.22.1)
project("whisper-engine" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WHISPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../whisper.cpp)
if (NOT EXISTS ${WHISPER_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "whisper.cpp submodule not found at ${WHISPER_DIR}")
endif()

# Build whisper.cpp static target
add_subdirectory(${WHISPER_DIR} build-whisper EXCLUDE_FROM_ALL)

# JNI bridge library
add_library(whisper_engine SHARED
    src/main/cpp/whisper_jni.cpp
)

target_include_directories(whisper_engine PRIVATE
    ${WHISPER_DIR}/include
    ${WHISPER_DIR}/ggml/include
)

target_compile_options(whisper_engine PRIVATE
    -O3
    -march=armv8.4-a+dotprod+i8mm+fp16
    -fvisibility=hidden
    -fvisibility-inlines-hidden
)

target_link_options(whisper_engine PRIVATE
    -Wl,--exclude-libs,ALL
)

target_link_libraries(whisper_engine PRIVATE
    whisper
    android
    log
)
