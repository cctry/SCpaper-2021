add_library(encoder
Encoder.cu
)

target_link_libraries(encoder
culib
linear
attention
skiplayernorm
)