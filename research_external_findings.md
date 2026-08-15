# External optimization research findings

## Mamba / selective SSM

Source: https://arxiv.org/abs/2312.00752

The Mamba paper states that selective state-space models target linear-time sequence processing and addresses a key weakness of earlier subquadratic models: weak content-based reasoning on discrete modalities. It makes SSM parameters input-dependent and introduces a hardware-aware parallel algorithm for recurrent-mode execution. The paper reports fast inference and linear scaling with sequence length, but these are properties of a specialized scan kernel and hardware-aware implementation, not of an arbitrary Python recurrent loop.

## FlashAttention

Source attempted: https://proceedings.neurips.cc/paper_files/paper_2022/hash/67d57c32e20fd0a7a302cb81d36e40f5-Abstract.html

The selected proceedings URL returned Page Not Found in the browser. Do not cite it as evidence until a valid source URL is obtained. The search result identified the FlashAttention contribution as an exact-attention method focused on reducing HBM access and improving memory efficiency; this must be cross-checked against a valid paper or publisher page before inclusion in the final report.

## FlashAttention

Source: https://arxiv.org/abs/2205.14135

FlashAttention describes exact attention as quadratic in sequence length and argues that wall-clock performance is also governed by I/O between GPU HBM and on-chip SRAM. Its proposed tiled, fused algorithm reduces HBM reads/writes and is reported to deliver substantial end-to-end speedups while retaining exact attention. The relevant lesson for CDI is that asymptotic linear state does not guarantee high throughput: memory traffic, kernel fusion, and hardware mapping determine realized speed.

## GRU

Source: https://d2l.ai/chapter_recurrent-modern/gru.html

The D2L GRU reference describes GRU as a simplified gated recurrent cell with reset and update gates. The update gate controls how much of the prior state is retained versus replaced by new content, and the reset gate controls how much prior state participates in the candidate. The source characterizes GRU as faster to compute than LSTM while retaining an internal state and multiplicative gating. The relevant CDI comparison point is that GRU remains a sequential recurrence but typically maps to optimized fused library kernels, while CDI currently performs a Python-level token loop around many small tensor operations.

## PyTorch GRU implementation

Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html

The official PyTorch documentation gives the reset/update/new gate equations and notes that the implementation is intentionally arranged for efficiency. It also documents the packed sequence interface and optimized backend conditions. CDI's matched pilot uses a hand-written `GRUCell` loop rather than `torch.nn.GRU`, so the current GRU baseline is not a best-case fused-library reference; a fair systems comparison should include both the repository baseline and an optimized `nn.GRU` baseline.

## PyTorch compilation

Source: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

The official tutorial states that `torch.compile` traces Python code into optimized kernels, while unsupported Python code causes graph breaks and lost optimization opportunities. `fullgraph=True` can expose unsupported constructs instead of silently splitting graphs. This directly applies to CDI: its per-token Python loop and runtime checks are likely barriers unless the recurrence is rewritten as a fixed-shape tensor kernel, a compiled scan, or an external custom operator.
