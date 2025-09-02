cmake_minimum_required(VERSION 3.10)
project(AIvoice)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BOOST_ROOT "/opt/homebrew")
find_package(Boost 1.70.0 REQUIRED COMPONENTS system thread)
find_package(nlohmann_json REQUIRED)
# find_package(FFmpeg REQUIRED COMPONENTS avcodec avformat avutil)

# onnxruntime (brew 安装，手动指定路径)
set(ONNXRUNTIME_INCLUDE_DIR "/opt/homebrew/include")
set(ONNXRUNTIME_LIB_DIR "/opt/homebrew/lib")

set(FFMPEG_INCLUDE_DIR "/opt/homebrew/include")
set(FFMPEG_LIB_DIR "/opt/homebrew/lib")

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${Nlohmann_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${FFMPEG_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

link_directories(
    ${ONNXRUNTIME_LIB_DIR}
    ${FFMPEG_INCLUDE_DIR}
)

file(GLOB_RECURSE SERVER_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(aivoice ${SERVER_SOURCES})

target_link_libraries(aivoice 
  ${Boost_LIBRARIES} 
  ${Nlohmann_LIBRARIES} 
  onnxruntime   # 直接指定库名
  avcodec
  avformat
  avutil
  pthread
)
