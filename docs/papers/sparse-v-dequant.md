# Attention-Gated Value Dequantization for Quantized KV Cache Inference

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## Abstract

Quantized KV cache compression (e.g., TurboQuant, NVFP4) reduces memory consumption during LLM inference but introduces a dequantization bottleneck during autoregressive decoding. We observe that in flash attention kernels, softmax attention weights are computed *before* value accumulation, and that at long context lengths, 90%+ of these weights are negligible. We propose **sparse V dequantization**: skipping value dequantization for positions where the attention weight falls below a threshold. On Apple M5 Max with a 35B MoE model, this yields **+22.8% decode throughput at 32K context** with zero perplexity loss, using 3 lines of code. The technique is orthogonal to existing dequant optimizations, general to any quantized KV cache scheme, and its benefit scales with context length.

---

## 1. Introduction

KV cache compression is becoming essential for long-context LLM inference. Google's TurboQuant (ICLR 2026) achieves 4.6× compression via Walsh-Hadamard rotation and polar quantization, with perplexity within 1.1% of uncompressed baselines. However, the compression introduces a per-token dequantization cost during autoregressive decoding that grows with context length.

On Apple Silicon, TurboQuant's dequant uses a centroid lookup table (LUT) in Metal constant memory. Profiling reveals this LUT accounts for 14–34% of decode time depending on hardware generation and context depth. At 32K context on M5 Max, the dequant overhead alone reduces decode throughput from 78.3 tok/s (no-dequant ceiling) to 47.0 tok/s — a 40% penalty. Notably, the no-dequant ceiling is **28% faster than uncompressed q8_0** (61.0 tok/s) because the compressed cache moves less data over the memory bus.

This gap motivated an exhaustive search: 14 alternative dequant implementations were tested on M2 Pro and M5 Max hardware, including register arrays, bit-arithmetic, FMA branchless computation, simd_shuffle cross-lane transfer, and fused block dot products. None beat the baseline constant memory LUT on Apple Silicon (see Section 5).

We then observed that the dequant cost splits roughly 50/50 between K (key) and V (value) paths. The key insight: in flash attention, softmax weights are computed from K *before* V is accessed. At long context, most attention weights are negligible. Skipping V dequant for these positions eliminates approximately half the total dequant cost at long context, with no measurable quality impact.

---

## 2. Background

### 2.1 TurboQuant KV Cache Compression

TurboQuant compresses KV cache entries from 16-bit to 3.5-bit via:
1. **Walsh-Hadamard Transform (WHT):** Rotates vectors to make coordinates approximately Gaussian
2. **Polar decomposition:** Converts Cartesian coordinates to polar (angle + radius) recursively
3. **Codebook quantization:** Maps angles to precomputed centroids via Lloyd's algorithm on the known analytical distribution

The result: 4.6× compression with 1.1% perplexity loss. Each 32-element block stores a norm (2 bytes), quantization indices (8 bytes), and sign bits (4 bytes) = 14 bytes total.

### 2.2 Flash Attention Decode Path

During autoregressive decoding (batch size = 1), flash attention computes:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

In fused flash attention kernels, this proceeds in tiles:
1. **K phase:** For each tile of KV positions, dequantize K, compute $QK^T$ scores, update running softmax
2. **V phase:** Using the computed attention weights, dequantize V, accumulate weighted sum

The attention weights $\alpha_i = \text{softmax}(QK^T)_i$ are known before V dequantization begins. This is the opportunity.

### 2.3 Attention Sparsity at Long Context

At context length $n$, the softmax distribution concentrates on a small subset of positions. Empirically, at 32K context with Qwen3.5-35B-A3B, over 90% of attention weights fall below $10^{-6}$. This sparsity increases with context length — the longer the conversation, the more positions have negligible attention.

---

## 3. Method

### 3.1 Sparse V Dequantization

We add a single conditional before the V dequant inner loop:

```metal
FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
    const float attn_weight = float(ss[NE*cc + ty]);
    if (attn_weight < 1e-6f) continue;  // skip negligible positions

    // ... existing V dequant and accumulation ...
}
```

When the attention weight for a KV position is below the threshold ($10^{-6}$), the entire V dequantization and accumulation for that position is skipped. This saves:
- Constant memory reads (centroid LUT lookups)
- ALU operations (index extraction, sign application, norm multiplication)
- Device memory reads (block data)

### 3.2 Threshold Selection

The threshold $\tau = 10^{-6}$ was chosen conservatively. For a 32K context with 32 attention heads, a weight of $10^{-6}$ contributes at most $10^{-6} \times \|V\|$ to the output — negligible relative to the dominant attention positions. We validate empirically that perplexity is identical at this threshold (Section 4).

### 3.3 Interaction with K Dequant Optimizations

Sparse V dequant is orthogonal to K-path optimizations. The K dequant must still run for all positions (to compute attention weights). Existing K-path optimizations include:
- **4-magnitude LUT:** Reduces constant memory addresses from 8 to 4 (+38% on M2 Pro)
- **Hardware auto-detection:** M1/M2/M3/M4 use 4-mag LUT; M5+ uses full 8-entry LUT

These stack: 4-mag reduces K dequant cost, sparse V reduces V dequant cost. Combined gains are additive.

---

## 4. Experiments

### 4.1 Setup

- **Hardware:** Apple M5 Max, 128GB unified memory, 546 GB/s bandwidth
- **Model:** Qwen3.5-35B-A3B (MoE), Q8_0 weight quantization
- **KV cache:** TurboQuant turbo3 (3.5-bit, 4.6× compression)
- **Framework:** llama.cpp with Metal flash attention kernels
- **Baseline:** turbo3 decode without sparse V (main branch)
- **Quality metric:** Perplexity on WikiText-2 (8-chunk and 32-chunk)
- **Retrieval metric:** Needle-in-a-haystack (NIAH), single and multi-key

### 4.2 Decode Throughput

Full regression suite with sparse V enabled ($\tau = 10^{-6}$):

| Context Depth | Baseline (tok/s) | Sparse V (tok/s) | Improvement | vs q8\_0 |
|---------------|-----------------|------------------|-------------|----------|
| short | 76.5 | 77.6 | +1.4% | 0.899× |
| 4K | 72.0 | 74.9 | +4.0% | — |
| 8K | 66.9 | 71.7 | +7.2% | — |
| 16K | 58.9 | 66.5 | +12.9% | 0.923× |
| 32K | 47.0 | 57.7 | **+22.8%** | **0.931×** |

The benefit scales with context length because longer contexts have more positions with negligible attention weights. At 32K, the ratio vs uncompressed q8\_0 improves from 0.76× to 0.93× — near parity.

### 4.3 Quality Validation

| Metric | q8\_0 | turbo3 | turbo3 + sparse V |
|--------|-------|--------|-------------------|
| PPL (8-chunk) | 6.111 | 6.211 | 6.176 |
| PPL (32-chunk) | 5.415 | 5.471 | — |
| vs q8\_0 | — | +1.6% | +1.06% |

Sparse V perplexity (6.176) is actually *lower* (better) than baseline turbo3 (6.211). The threshold is conservative enough that skipping negligible positions introduces no measurable quality degradation.

### 4.4 NIAH Retrieval

| Test | q8\_0 | turbo3 | turbo3 + sparse V |
|------|-------|--------|-------------------|
| Single needle (9 positions) | 7/9 | 7/9 | **9/9 (100%)** |
| Multi-key (4K-32K) | 4/4 | 4/4 | 4/4 |

Sparse V achieves **perfect** single-needle retrieval (9/9), improving from 7/9 without sparse V. This is counterintuitive but explainable: needle positions always have meaningful attention weights (well above $10^{-6}$) and are never skipped. The improvement may stem from reduced numerical noise — accumulating fewer negligible V contributions produces a cleaner output signal.

### 4.5 Prefill Impact

Sparse V has minimal effect on prefill because prefill processes the entire prompt in parallel (no autoregressive attention weight computation). Measured prefill at 4K: 2429 tok/s with sparse V vs 2362 baseline (+2.8%).

### 4.6 Real-World Server Benchmark

The decode numbers in Section 4.2 are from `llama-bench` batch evaluation, which keeps the GPU maximally saturated. To validate under realistic conditions, we tested with `llama-server` processing a 70-page PDF (~24K prompt tokens) via the OpenAI-compatible chat completions API:

| Metric | turbo3 + sparse V | q8\_0 | ratio |
|--------|-------------------|-------|-------|
| Prefill | 1417.8 tok/s | 1449.9 tok/s | 0.98× |
| Decode | 53.3 tok/s | 68.2 tok/s | 0.78× |

The server-mode decode ratio (0.78×) is notably worse than the batch benchmark at comparable context depth (0.92× at 16K, 0.93× at 32K). The gap is attributable to server overhead — HTTP request handling, chat template processing, sampling, and single-request scheduling — which reduces GPU utilization and makes the per-token dequant cost proportionally larger. Prefill remains near parity (0.98×) because the prompt is processed in one large batch regardless of server overhead.

**Takeaway:** Users should expect ~78% of q8\_0 decode speed at long context in real server deployments, not the ~93% measured in synthetic benchmarks. The sparse V improvement still holds — without it, this number would be closer to 60%.

---

## 5. What Didn't Work: 14 Alternative Dequant Approaches

Before discovering sparse V, we exhaustively tested 14 dequant-level optimizations on M2 Pro (Apple8) and M5 Max (Apple10). All attempted to reduce the constant memory LUT cost:

| # | Approach | M2 8K tok/s | vs Best | Result |
|---|----------|-------------|---------|--------|
| 1 | **4-mag LUT + XOR sign** | **15.1** | **baseline** | **Best dequant-level fix (+38%)** |
| 2 | Batched byte extract | 13.7 | -9% | Better byte reading, still 8 LUT addresses |
| 3 | Inline block dequant | 13.5 | -11% | I-cache pressure |
| 4 | 2-pair half2 LUT | 12.0 | -21% | Ternary overhead exceeds LUT savings |
| 5 | Select chain (zero LUT) | 11.9 | -21% | Too much ALU |
| 6 | Bit-arithmetic | 11.6 | -23% | Pure ALU, zero memory — but ALU cost too high |
| 7 | Non-vec FA (nl=2) | 10.2 | -32% | Kernel not designed for single-token decode |
| 8 | float cn[8] registers | — | — | Metal spills to stack (M5 only) |
| 9 | half cn[8] registers | — | — | Also spills (M5 only) |
| 10 | Split 2×4 half LUT | — | — | Branch overhead (M5 only) |
| 11 | Deferred norm multiply | 12.9 | -15% | Loses ILP |
| 12 | FMA branchless | 11.4 | -25% | 7 ALU ops > 1 divergent constant read |
| 13 | simd_shuffle | 14.7 | -3% | Cross-lane latency ≈ constant LUT |
| 14 | Fused block dot | 8.1 | -46% | 64 float comparisons devastate throughput |

**Conclusion:** On Apple Silicon, 4 divergent constant memory reads are faster than any arithmetic computation that produces the same 4-way selection. The constant cache, even when divergent, beats 7+ ALU operations. The only path beyond the 4-mag LUT is changing *what* data is read (sparse V, format changes), not *how* it's computed.

---

## 6. Why Sparse V Works Where Dequant Tricks Don't

The 14 failed approaches all tried to make individual dequant operations cheaper. Sparse V succeeds because it eliminates entire dequant operations. The distinction:

- **Dequant optimization:** Make each of N operations faster → bounded by ALU/memory floor
- **Sparse V:** Skip (1-p)×N operations entirely → unbounded improvement as p → 1

At 32K context, p ≈ 0.9 (90% of positions skipped). At 128K, p would be even higher. The technique becomes *more* effective at exactly the context lengths where the dequant bottleneck is worst.

---

## 7. Generality

Sparse V dequantization is not specific to TurboQuant. It applies to any quantized KV cache scheme where:

1. Flash attention is used (softmax computed before V accumulation)
2. V is stored in a quantized format requiring dequantization
3. The dequant cost is non-trivial relative to the multiply-accumulate

This includes NVFP4, KIVI, CacheQuant, and other KV cache quantization methods. The 3-line implementation is kernel-level and requires no model changes, no retraining, and no calibration data.

---

## 8. Limitations and Future Work

**Limitations:**
- Tested on a single hardware platform (M5 Max). M1/M2 results pending (GPU contention during testing).
- The threshold $\tau = 10^{-6}$ is fixed. Adaptive thresholds based on context length or layer depth may yield better speed/quality tradeoffs.
- Short context benefit is modest (+1.4%) because attention is less sparse.

**Future work:**
- **Combined 4-mag + sparse V on M2:** Expected to stack since they address independent bottlenecks (K and V dequant respectively).
- **Context-adaptive dispatch:** Compile multiple FA kernel variants, select optimal path based on KV cache size at dispatch time.
- **Sparse K dequant:** Extend the sparsity concept to K. After a first pass computing approximate attention scores, skip K dequant for positions that won't contribute. Requires two-pass attention or speculative attention.
- **Threshold tuning:** Test $\tau$ values from $10^{-4}$ to $10^{-8}$ across model sizes and architectures.
- **Non-Apple hardware:** Test on NVIDIA (CUDA) and AMD (ROCm) where the dequant bottleneck profile differs.

---

## 9. Conclusion

We present a 3-line modification to flash attention kernels that yields up to 22.8% decode throughput improvement for quantized KV caches at long context, with zero quality degradation. The technique exploits the natural sparsity of attention weights — a property that becomes more pronounced exactly when the dequant bottleneck is most severe. Unlike 14 alternative dequant optimizations we tested, sparse V works because it eliminates operations rather than trying to make them cheaper. The approach is general, requiring no model changes, and is orthogonal to existing dequant and compression optimizations.

---

## Reproducibility

All code, benchmarks, and diagnostic tools are open source:

- **Implementation:** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (branch: `experimental_decode_speed_tests`)
- **Benchmarks & diagnostics:** [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- **Hardware analysis:** [decode-speed-hardware-analysis.md](../decode-speed-hardware-analysis.md)

To reproduce: build with `TURBO_SPARSE_V=1` environment variable and run `llama-bench` at various context depths with `-ctk turbo3 -ctv turbo3 -fa 1`.

---

## Acknowledgments

- **@spiritbuun** — CUDA fork with norm correction and register LUT, whose work inspired the hardware profiling investigation
- **@Ambisphaeric, @mariotomich** — Independent M1 Max testers who confirmed the decode regression is hardware-dependent
- **@ekryski** — GPT-OSS 20B testing on M1 Max with turbo4
- The TurboQuant community for extensive cross-hardware validation

## References

1. TurboQuant: Redefining AI Efficiency with Extreme Compression. Google Research, ICLR 2026.
2. Ilhan et al. "AttentionPack: Attention-aware Inference Optimizations for Large Vision-Language Models with Memory-efficient Decoding." arXiv:2603.23914, 2026.
3. An et al. "GlowQ: Group-Shared Low-Rank Approximation for Quantized LLMs." arXiv:2603.25385, 2026.
4. Xie et al. "Scaling Attention via Feature Sparsity." ICLR 2026. arXiv:2603.22300.
5. Zhao et al. "Self-Distillation for Multi-Token Prediction." arXiv:2603.23911, 2026.
6. Qasim et al. "The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference." arXiv:2603.19664, 2026.
7. Wang et al. "SliderQuant: Accurate Post-Training Quantization for LLMs." ICLR 2026. arXiv:2603.25284.
