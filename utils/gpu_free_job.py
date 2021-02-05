from utils.compute import get_index_of_free_gpus
import time

while True:
    get_index_of_free_gpus()
    time.sleep(60 * 30 )


