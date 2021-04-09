add_library(cusp
    cusp.cpp   
)

target_link_libraries(cusp
    culib
    cublas
    cusparse
)
