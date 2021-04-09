add_library(attention
Attention.cu
kernels.cu 
prune_attn.cu
mixed-precision/mixed_attn.cu
kernel_fuseO.cu
Attention_single.cu
)

target_link_libraries(attention
culib
mycublas
)
add_subdirectory(test)