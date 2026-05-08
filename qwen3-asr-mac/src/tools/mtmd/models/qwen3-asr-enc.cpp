// Qwen3-ASR Audio Encoder Graph
// Architecture: 3x Conv2D + conv_out Linear + 24-layer Transformer + 2-stage MLP projector
// Reference: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
//
// Convolution analysis (derived from conv_out weight [1024, 7680]):
//   7680 = 480 channels * 16 mel bins  -> conv stack halves mel 3 times: 128→64→32→16
//   Time dimension T is preserved through all conv2d ops (stride_W=1 for all 3)
//   Strides: all conv2d use stride_H=2 (ggml s1=2), stride_W=1 (ggml s0=1)
//
// Input  (from Whisper preprocessor): clip_image_f32 with nx=T_frames, ny=128_mel
// build_inp_raw(1) → tensor ne[0]=T, ne[1]=128, ne[2]=1  (W=T, H=128, C=1)
//
// After 3× conv2d (s0=1, s1=2, pad=1, kernel=3×3):
//   ne[0]=T, ne[1]=16, ne[2]=480, ne[3]=1  (W=T, H=16, C=480, N=1)
// Permute to [16, 480, T, 1] → cont → reshape to [7680=480×16, T]
// conv_out  mul_mat [7680→1024]:  [7680,T] × [7680,1024]^T = [T, 1024]
// Transpose → inp for build_vit: [1024, T]
// build_vit (24 transformer layers, no position embeddings)
// ln_post LayerNorm
// build_ffn  proj1[1024→1024]+GELU  then  proj2[1024→2048]
// Output: [n_pos=T, 2048]

#include "models.h"
#include "../clip-impl.h"

ggml_cgraph * clip_graph_qwen3a_enc::build() {
    const int n_frames = img.nx; // mel time frames (W dimension)
    // img.ny == 128 (mel bins)

    // Input tensor: ne[0]=T, ne[1]=128, ne[2]=1
    // Treated as 2D conv input: W=T (time), H=128 (mel), C=1
    ggml_tensor * inp = build_inp_raw(1);

    // -------------------------------------------------------------------------
    // Conv2D block — 3 layers, each halves the mel (H) dimension
    // stride: s0=1 (time unchanged), s1=2 (mel halved)
    // -------------------------------------------------------------------------
    auto add_conv_bias = [&](ggml_tensor * cur, ggml_tensor * bias) {
        // conv2d output is [T, OH, OC, 1]  →  bias [OC] needs shape [1,1,OC,1]
        ggml_tensor * b4d = ggml_reshape_4d(ctx0, bias, 1, 1, bias->ne[0], 1);
        return ggml_add(ctx0, cur, b4d);
    };

    // conv2d1: [480, 1, 3, 3] weight,  [480] bias
    inp = ggml_conv_2d(ctx0, model.conv2d_1_w, inp,
                       /*s0*/1, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
    inp = add_conv_bias(inp, model.conv2d_1_b);
    inp = ggml_gelu_erf(ctx0, inp);
    cb(inp, "conv2d_1", -1);

    // conv2d2: [480, 480, 3, 3] weight,  [480] bias
    inp = ggml_conv_2d(ctx0, model.conv2d_2_w, inp,
                       /*s0*/1, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
    inp = add_conv_bias(inp, model.conv2d_2_b);
    inp = ggml_gelu_erf(ctx0, inp);
    cb(inp, "conv2d_2", -1);

    // conv2d3: [480, 480, 3, 3] weight,  [480] bias
    inp = ggml_conv_2d(ctx0, model.conv2d_3_w, inp,
                       /*s0*/1, /*s1*/2, /*p0*/1, /*p1*/1, /*d0*/1, /*d1*/1);
    inp = add_conv_bias(inp, model.conv2d_3_b);
    inp = ggml_gelu_erf(ctx0, inp);
    cb(inp, "conv2d_3", -1);
    // inp: ne[0]=T, ne[1]=16, ne[2]=480, ne[3]=1

    // -------------------------------------------------------------------------
    // Flatten mel×channels → [7680=480×16, T]
    // -------------------------------------------------------------------------
    // Permute [T, 16, 480, 1] → [16, 480, T, 1]
    // ggml_permute(inp, ax0,ax1,ax2,ax3): ne[axi]=inp.ne[i]
    // To get [16,480,T,1] from [T,16,480,1]: ax1=0, ax2=1, ax0=2, ax3=3 → (2,0,1,3)
    ggml_tensor * perm = ggml_permute(ctx0, inp, 2, 0, 1, 3);
    inp = ggml_cont(ctx0, perm);
    // ne[0]=16, ne[1]=480, ne[2]=T, ne[3]=1
    const int64_t n_seq = inp->ne[2]; // T (actual sequence length)
    inp = ggml_reshape_2d(ctx0, inp, 480 * 16, n_seq); // ne[0]=7680, ne[1]=T
    cb(inp, "flattened", -1);

    // -------------------------------------------------------------------------
    // conv_out linear: [7680 → 1024], no bias
    // mul_mat(W[7680,1024], x[7680,T]) → [T, 1024]
    // -------------------------------------------------------------------------
    inp = ggml_mul_mat(ctx0, model.conv_out_w, inp); // ne[0]=1024, ne[1]=T
    cb(inp, "after_conv_out", -1);

    const int64_t n_pos = inp->ne[1]; // T

    // -------------------------------------------------------------------------
    // 24-layer Whisper-style transformer encoder (no position embeddings)
    // -------------------------------------------------------------------------
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);

    ggml_tensor * cur = build_vit(
        inp,
        n_pos,
        NORM_TYPE_NORMAL,
        hparams.ffn_op,
        nullptr,   // no learned position embeddings
        nullptr);  // no position-add callback

    cb(cur, "after_transformer", -1);

    // -------------------------------------------------------------------------
    // post-LayerNorm (ln_post)
    // -------------------------------------------------------------------------
    cur = ggml_norm(ctx0, cur, hparams.eps);
    cur = ggml_mul(ctx0, cur, model.a_post_ln_w);
    cur = ggml_add(ctx0, cur, model.a_post_ln_b);
    cb(cur, "after_post_ln", -1);

    // -------------------------------------------------------------------------
    // Two-stage MLP projector: proj1[1024→1024]+GELU  →  proj2[1024→2048]
    // build_ffn handles the matmul shape correctly
    // -------------------------------------------------------------------------
    cur = build_ffn(cur,
        model.mm_proj1_w, model.mm_proj1_b,
        nullptr, nullptr,
        model.mm_proj2_w, model.mm_proj2_b,
        FFN_GELU_ERF,
        -1);

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

