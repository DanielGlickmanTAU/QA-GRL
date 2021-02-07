from utils.compute import get_index_of_free_gpus
import time

while True:
    gpu_memory = get_index_of_free_gpus()
    print(gpu_memory)
    time.sleep(60 * 30 )


