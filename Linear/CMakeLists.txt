add_library(linear
SPMM_row.cu
cusp.cu
SPMM_tile.cu
Linear.cu
SPMM_col.cu
SPMM1x16.cu
)
target_link_libraries(linear
    mycublas
    mat_type
    culib
    cusparse
)

add_subdirectory(test)