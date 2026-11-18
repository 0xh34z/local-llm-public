// Isolated gfx1152 experiment for the contiguous Q8_0 vocabulary head.
// One wave computes one row using the stock dot and reduction order.
template<int rows>
static __global__ void vocab_rows(const void * weights, const block_q8_1 * input, float * out) {
    const int lane = threadIdx.x;
    const int row = blockIdx.x * rows + threadIdx.y;
    constexpr int blocks_per_row = 2048 / QK8_0;
    constexpr int lanes_per_block = QI8_0 / VDR_Q8_0_Q8_1_MMVQ;
    const int quant_index = (lane % lanes_per_block) * VDR_Q8_0_Q8_1_MMVQ;
    float sum = 0;
    #pragma unroll
    for (int k = lane / lanes_per_block; k < blocks_per_row; k += 32 / lanes_per_block) {
        sum += vec_dot_q8_0_q8_1(weights, input + k, row * blocks_per_row + k, quant_index);
    }
    const float value = warp_reduce_sum<32>(sum);
    if (lane == 0) out[row] = value;
}

template<int rows>
static void launch_vocab_rows(const void * weights, const void * input, float * out, cudaStream_t stream) {
    static_assert(248320 % rows == 0);
    const auto params = ggml_cuda_kernel_launch_params(dim3(248320 / rows), dim3(32, rows), 0, stream);
    ggml_cuda_kernel_launch(vocab_rows<rows>, params, weights, static_cast<const block_q8_1 *>(input), out);
}
