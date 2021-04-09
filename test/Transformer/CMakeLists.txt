add_executable(test_transformer
    main.cu
)

target_link_libraries(test_transformer 
    linear
    culib
    attention
    layernorm
    skiplayernorm
    encoder
    fused_attention
)