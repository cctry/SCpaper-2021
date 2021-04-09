add_executable(testVO
    testVO.cu
)


add_executable(testfused
    testfused.cu
)

target_link_libraries(testVO
    linear 
    culib
    mycublas
    fused_attention
)

target_link_libraries(testfused
    linear 
    culib
    mycublas
    fused_attention
)


add_executable(test_prune
test_prune.cu
)

target_link_libraries(test_prune
    linear 
    culib
    mycublas
    fused_attention
)


add_executable(test_prune_encoder
testfusedEncoder.cu
)

target_link_libraries(test_prune_encoder
    linear 
    culib
    mycublas
    fused_attention
    skiplayernorm
)