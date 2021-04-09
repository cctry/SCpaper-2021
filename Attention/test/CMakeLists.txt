set(CMAKE_CUDA_ARCHITECTURES 70)

add_executable(test_attn
test_attn.cu
)
target_link_libraries(test_attn
attention
linear
culib
)

add_executable(test_prune_attn
test_prune_attn.cu
)
target_link_libraries(test_prune_attn
attention
linear
culib
)

add_executable(test_attn_kernels
test_attn_kernels.cu
)
target_link_libraries(test_attn_kernels
attention
linear
culib
)
