#include <ATen/ATen.h>


namespace at {
namespace native {
namespace preprocessing {

/**
 * This function will take nested query, key, and value
 * and will preprocess it in order to run with either
 * the flash-attention or efficient-attention kernels.
 * @return A tuple containing all the necessary data for running the fused
 * kernels
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, int64_t, Tensor>
sdpa_nested_preprocessing(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value);

/**
 * This function will take the output information from the forward function
 * and will preprocess it in order to run with either
 * the backward pass for flash-attention.
 * Memory efficient attention currently does not support ragged backwards.
 * @return A tuple containing all the necessary data for running
 * FlashAttention backward
 */
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
sdpa_nested_preprocessing_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_kv,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_kv);

/**
 * This function will take the the dense outputs from backward function
 * and will post process it to the correct nested shape.
 * Memory efficient attention currently does not support ragged backwards.
 * @return A tuple containing all the necessary data for running
 * FlashAttention backward
 */
std::tuple<Tensor, Tensor, Tensor>
sdpa_nested_postprocessing_backward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& grad_q_buffer,
    const at::Tensor& grad_k_buffer,
    const at::Tensor& grad_v_buffer);
} // namespace preprocessing
} // namespace native
} // namespace at