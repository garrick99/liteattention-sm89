// Auto-generated INT8 skipable SM90 instantiation (PackGQA=true variant)
#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM256
template void run_mha_fwd_<90, int8_t, 256, 256, false, false, false, true>(Flash_fwd_params &params, cudaStream_t stream);
#endif
