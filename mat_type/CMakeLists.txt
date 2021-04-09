add_library(mat_type
    tile_mat/tile_mat.cu
    row_mat/row_mat.cu
    csr_mat/csr_mat.cpp
    col_mat/col_mat.cu
    mat1x16/mat1x16.cu
)
set(CMAKE_CXX_STANDARD 17)
target_link_libraries(mat_type
culib
)