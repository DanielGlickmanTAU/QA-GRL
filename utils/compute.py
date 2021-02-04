import os
import torch


def get_index_of_free_gpus(minimum_free_giga=4):
    def get_free_gpu():
        lines = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()
        memory_available = [int(x.split()[2]) for x in lines]
        return {index: mb for index, mb in enumerate(memory_available)}

    return [index for index, mega in get_free_gpu().items() if mega > minimum_free_giga * 1000]


def get_device():
    return torch.device('cuda:' + str(get_index_of_free_gpus()[0]) if torch.cuda.is_available() else 'cpu')


def get_device_and_set_as_global():
    d = get_device()
    torch.cuda.set_device(d)
    return d
