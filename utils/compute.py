import os
import time
import torch


def get_free_gpu():
    lines = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()
    memory_available = [int(x.split()[2]) for x in lines]
    return {index: mb for index, mb in enumerate(memory_available)}

last_write = 0
def write_gpus_to_file(dict):
    global last_write
    t = time.time()
    if t - last_write > 20:
        last_write = t
        with open(str(t) + '_gpu', 'w+') as f:
            f.write(str(dict))


def get_index_of_free_gpus(minimum_free_giga=4):
    gpus = get_free_gpu()
    write_gpus_to_file(gpus)
    return [index for index, mega in gpus.items() if mega > minimum_free_giga * 1000]


def get_torch():
    gpus = get_free_gpu()
    write_gpus_to_file(gpus)



def get_device():
    return torch.device('cuda:' + str(get_index_of_free_gpus()[0]) if torch.cuda.is_available() else 'cpu')


def get_device_and_set_as_global():
    d = get_device()
    torch.cuda.set_device(d)
    return d
