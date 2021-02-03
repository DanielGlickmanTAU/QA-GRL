import os
import torch


def get_index_of_free_gpus(minimum_free_giga=4):
    def get_free_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_mem_allo')
        memory_available = [int(x.split()[2]) for x in open('tmp_mem_allo', 'r').readlines()]
        return {index: mb for index, mb in enumerate(memory_available)}

    return [index for index, mega in get_free_gpu().items() if mega > minimum_free_giga * 1000]


def get_device():
    return torch.device('cuda:' + get_index_of_free_gpus()[0] if torch.cuda.is_available() else 'cpu')
