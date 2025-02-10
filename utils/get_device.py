import torch
import pynvml
def get_free_gpu():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_memory = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory.append((i, info.free))

    pynvml.nvmlShutdown()
    free_memory.sort(key=lambda x: x[1], reverse=True)
    return free_memory[0][0]


import torch


def get_device(threshold_gb=1.0):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        max_free_memory = 0
        best_device_index = -1

        for i in range(num_gpus):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            reserved_memory = torch.cuda.memory_reserved(i)
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = total_memory - reserved_memory - allocated_memory

            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_device_index = i

        # from B to GB
        max_free_memory_gb = max_free_memory / (1024 ** 3)

        if max_free_memory_gb > threshold_gb:
            return torch.device(f'cuda:{best_device_index}')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')






if __name__ == '__main__':
    # print(get_free_gpu())
    # threshold of memory 1GB
    threshold_gb = 1
    device = get_device(threshold_gb)

    print(f"Selected device: {device}")
