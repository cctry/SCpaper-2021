add_executable(transformer_infer main.cu)
target_link_libraries(transformer_infer
    culib
    linear
    attention
    layernorm
    mat_type
    skiplayernorm
    cnpy
    z
)
