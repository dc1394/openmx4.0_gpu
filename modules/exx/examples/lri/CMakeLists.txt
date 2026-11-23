cmake_minimum_required(VERSION 3.25.2)
project(OpenMXExxLriExample LANGUAGES CXX)
find_package(OpenMXExx 0.1 CONFIG REQUIRED)
# The example reads .pao files with the basis package itself.
find_package(OpenMXBasis 0.1 CONFIG REQUIRED)
add_executable(lri_example main.cpp)
target_link_libraries(lri_example PRIVATE OpenMX::exx OpenMX::basis)
