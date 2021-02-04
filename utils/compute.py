import os
import time


last_write = 0


home = '/specific/scratches/scratch/glickman1/'
def write_gpus_to_file(dict):
    if len(dict) == 0:
        return
    global last_write
    t = time.time()
    if t - last_write > 20:
        last_write = t
        try:
            with open(home + '/gpu_usage/' + str(t) + '_gpu', 'w+') as f:
                f.write(str(dict))
        except: pass


def get_index_of_free_gpus(minimum_free_giga=4):
    def get_free_gpu():
        try:
            lines = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()
        except:
            return {}

        memory_available = [int(x.split()[2]) for x in lines]
        return {index: mb for index, mb in enumerate(memory_available)}

    gpus = get_free_gpu()
    write_gpus_to_file(gpus)
    return [str(index) for index, mega in gpus.items() if mega > minimum_free_giga * 1000]


def get_torch():
    gpus = get_index_of_free_gpus()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)
    import torch
    return torch


def get_device():
    # todo here probably should just use device 0, as in get torch we are disabling devices in OS level, so if only gpu 4 is free, it will be regraded as gpu 0.
    torch = get_torch()
    return torch.device('cuda:' + get_index_of_free_gpus()[0] if torch.cuda.is_available() else 'cpu')


def get_device_and_set_as_global():
    d = get_device()
    get_torch().cuda.set_device(d)
    return d
