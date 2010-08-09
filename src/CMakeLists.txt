set(OPENCL_KERNELS
    opencl_ex
)

configure_file(
  ${OpenCL_EX_SOURCE_DIR}/src/config.h.cmake.in
  ${OpenCL_EX_BINARY_DIR}/src/config.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(cl_gen_header cl_gen_header.c)

foreach(kernel ${OPENCL_KERNELS})
  add_custom_command (
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${kernel}.h
    COMMAND cl_gen_header ${CMAKE_CURRENT_SOURCE_DIR}/${kernel}.cl
                           ${CMAKE_CURRENT_BINARY_DIR}/${kernel}.h
    DEPENDS cl_gen_header ${kernel}.cl
    )
endforeach(kernel)

add_executable(opencl_ex opencl_ex.c opencl_ex.h)

target_link_libraries(
  opencl_ex
  ${OPENCL_LIBRARIES}
)
