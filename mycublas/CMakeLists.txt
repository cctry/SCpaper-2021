add_library(mycublas STATIC
    cublas.cu
)

target_link_libraries(mycublas
cublas
)