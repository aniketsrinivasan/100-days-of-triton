import time
import torch


def benchmark(function, *args, warmup_steps: int = 10, iteration_steps: int = 100):
    assert warmup_stpes >= 0, "Invalid number of warmup_steps."
    assert iteration_steps > 0, "Invalid number of iteration_steps."

    # warmup steps allow the GPU to set up one-time contents (e.g. JiT for Triton)
    for _ in range(warmup_steps):
        function(*args)
    # ensure all warmup kernels have finished (synchronize them)
    torch.cuda.synchronize()

    # start timed benchmark
    start_t = time.perf_counter()
    for _ in range(iteration_steps):
        function(*args)
    torch.cuda.synchronize()
    end_t = time.perf_counter()

    # compute average time 
    average_time_ms = (end_t - start_t) / (n_iters * 1000)
    return average_time_ms