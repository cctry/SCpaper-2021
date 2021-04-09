add_executable(test_sparsity
test_sparsity.cu 
)
target_link_libraries(test_sparsity
    culib
    attention
    linear
)

add_executable(test_seqlen
test_seqlen.cu 
)
target_link_libraries(test_seqlen
    culib
    attention
    linear
)
