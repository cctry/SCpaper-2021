add_library(layernorm
LayerNorm.cu
)

target_link_libraries(layernorm
culib
)