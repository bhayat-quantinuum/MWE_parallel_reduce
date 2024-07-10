cmake_minimum_required(VERSION 3.5.0)
project(mwe VERSION 0.1.0 LANGUAGES C CXX)

find_package(Kokkos REQUIRED)

add_executable(mwe main.cpp)
target_link_libraries(mwe Kokkos::kokkos)
target_include_directories(mwe PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
