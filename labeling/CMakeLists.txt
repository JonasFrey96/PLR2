cmake_minimum_required(VERSION 3.10)
project(labeler)

set(LIBIGL_WITH_COMISO             OFF)
set(LIBIGL_WITH_EMBREE             OFF)
set(LIBIGL_WITH_OPENGL             ON)
set(LIBIGL_WITH_OPENGL_GLFW        ON)
set(LIBIGL_WITH_OPENGL_GLFW_IMGUI  ON)
set(LIBIGL_WITH_PNG                ON)
set(LIBIGL_WITH_TETGEN             OFF)
set(LIBIGL_WITH_TRIANGLE           OFF)
set(LIBIGL_WITH_PREDICATES         OFF)
set(LIBIGL_WITH_XML                OFF)
set(LIBIGL_WITH_PYTHON             OFF)
set(LIBIGL_USE_STATIC_LIBRARY      OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(LIBIGL REQUIRED core opengl_glfw opengl_glfw_imgui png)

include_directories(src/)

# Add your project files
# FILE(GLOB SRCFILES src/*.cpp)
add_executable(${PROJECT_NAME} src/main.cpp)
add_dependencies(${PROJECT_NAME} igl)
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw igl::opengl_glfw_imgui igl::png Boost::filesystem)

