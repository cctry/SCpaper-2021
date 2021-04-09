add_executable(table2
    main.cu
)

target_link_libraries(table2 
    linear
    culib
    attention
    layernorm
    skiplayernorm
    encoder
    fused_attention
)