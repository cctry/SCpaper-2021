add_executable(test
    main.cu
)

target_link_libraries(test 
    linear
    culib
    attention
    layernorm
    skiplayernorm
    encoder
)

add_subdirectory(Attention)
add_subdirectory(Encoder)
add_subdirectory(Table2)
add_subdirectory(Transformer)