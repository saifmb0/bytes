"""Small CUDA-graph timing helper shared by system microbenchmarks."""

import torch


def benchmark_with_cuda_graphs(attn_func, *args, num_warmups=20, num_runs=200):
    """Measure a warmed-up GPU function with CUDA graph replay, in milliseconds."""
    for _ in range(num_warmups):
        attn_func(*args)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attn_func(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_runs):
        graph.replay()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs


def benchmark_captures(attn_func, *args, captures=15, num_warmups=20, num_runs=100, **kwargs):
    """Independent CUDA-graph captures, retaining one mean latency per capture."""
    return [benchmark_with_cuda_graphs(lambda: attn_func(*args, **kwargs),
                                       num_warmups=num_warmups, num_runs=num_runs)
            for _ in range(captures)]
