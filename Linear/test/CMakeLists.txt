add_executable(test_cublas
test_cublas.cu
)

add_executable(test_row
test_row.cu 
)

add_executable(test_cusp
test_cusp.cu 
)

add_executable(test_tile
test_tile.cu 
)

add_executable(test_col
test_col.cu 
)

add_executable(benchmark
benchmark.cu 
)

add_executable(test_dense
test_dense.cu 
)

target_link_libraries(test_tile
culib
linear
)

target_link_libraries(test_cublas
culib
linear
)


target_link_libraries(test_row
culib
linear
)


target_link_libraries(test_cusp
culib
linear
)

target_link_libraries(test_col
culib
mat_type
linear
)

target_link_libraries(benchmark
culib
linear
mat_type
)

target_link_libraries(test_dense
culib
linear
mat_type
)

# add_executable(test_1x16
# test_1x16.cu 
# )

# target_link_libraries(test_1x16
# culib
# linear
# )


add_executable(test_concurrent
test_concurrent.cu 
)

target_link_libraries(test_concurrent
culib
linear
)


add_executable(test_outprod
test_outprod.cu 
)

target_link_libraries(test_outprod
culib
linear
)



add_executable(test_algo_col
benchmark/test_col
)

target_link_libraries(test_algo_col
culib
linear
mat_type
mycublas
)
