add_library(fused_attention
Fused_attention.cu
Linear_VO.cu
)

target_link_libraries(fused_attention
culib
mycublas
attention
)
add_subdirectory(test)