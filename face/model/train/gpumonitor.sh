nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5 >gpu.log 2>&1 &
