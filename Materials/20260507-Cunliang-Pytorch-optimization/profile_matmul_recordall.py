import torch
from torch.profiler import profile, ProfilerActivity, record_function
import time


def create_large_tensor_cpu(size, operations=10):
    """Create a large tensor on CPU with some computational work to take time"""
    # Create initial tensor
    tensor = torch.randn(size, size, device="cpu")

    # Perform some operations to make creation take time
    for _ in range(operations):
        tensor = tensor + torch.randn(size, size, device="cpu") * 0.01
        tensor = torch.sin(tensor)

    return tensor


# Profile the entire process: CPU creation, GPU transfer, and matmul
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             on_trace_ready=torch.profiler.tensorboard_trace_handler("trace_matmul", worker_name="matmul_recordall", use_gzip=True),
             record_shapes=True,
             profile_memory=True,
             with_stack=True) as prof:

    # Step 1: Create tensors on CPU (this will take time)
    with record_function("cpu_tensor_creation"):
        x_cpu = create_large_tensor_cpu(1024, 5)
        w_cpu = create_large_tensor_cpu(1024, 5)

    # Step 2: Transfer tensors to GPU
    with record_function("cpu_to_gpu_transfer"):
        x_gpu = x_cpu.to("cuda")
        w_gpu = w_cpu.to("cuda")

    # Step 3: Matrix multiplication on GPU
    with record_function("gpu_matmul"):
        y_gpu = torch.matmul(x_gpu, w_gpu)
        # Ensure computation is complete within the timing context
        torch.cuda.synchronize()

# Save memory profile to a file
prof.export_memory_timeline("trace_matmul/matmul_recordall_memory_timeline.json")

print("=== PROFILING RESULTS BY CPU TIME===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

print("\n=== PROFILING RESULTS BY CUDA TIME ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
