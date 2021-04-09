add_library(culib STATIC
    utils.cu
    CUDA_ptr.cu
)

target_compile_options(culib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       >)