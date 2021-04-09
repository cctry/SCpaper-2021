add_executable(test_sparsity_encoder
test_sparsity.cu 
)
target_link_libraries(test_sparsity_encoder
    culib
    attention
    linear
    encoder
)



add_executable(test_sparsity_encoder_single_QKV
test_sparsity_single.cu 
)
target_link_libraries(test_sparsity_encoder_single_QKV
    culib
    attention
    linear
    encoder
)
