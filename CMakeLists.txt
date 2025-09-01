cmake_minimum_required(VERSION 3.10)
project(AIvoice)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# list(APPEND CMAKE_PREFIX_PATH "opt/homebrew")
set(BOOST_ROOT "/opt/homebrew")
find_package(Boost 1.70.0 REQUIRED COMPONENTS system thread)
# find_package(Boost.Beast 1.70.0 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(onnxruntime REQUIRED)


include_directories(
  ${Boost_INCLUDE_DIRS}
  ${Nlohmann_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/include
)

file(GLOB_RECURSE SERVER_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(aivoice ${SERVER_SOURCES})


target_link_libraries(aivoice 
${Boost_LIBRARIES} 
${Nlohmann_LIBRARIES} 
${Onnxruntime_LIBRARIES} 
pthread)
