add_library(skiplayernorm
SkipLayerNorm.cu
)

target_link_libraries(skiplayernorm
culib
)
