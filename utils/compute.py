import os
import time

minimum_free_giga = 8
max_num_gpus = 3

last_write = 0
home = '/specific/netapp5_3/ML_courses/students/DL2020/glickman1'

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def write_gpus_to_file(dict):
    if len(dict) == 0:
        return
    global last_write
    t = time.time()
    if t - last_write > 20:
        last_write = t
        try:
            server_name = os.environ['HOST']
            filename = home + '/gpu_usage/' + str(t) + '__' + server_name + '__' + '_gpu'
            with open(filename, 'w+') as f:
                f.write(str(dict))
            print('print to gpu usage to ' + filename, dict)
        except:
            print('fail to save file')


def get_index_of_free_gpus(minimum_free_giga=minimum_free_giga):
    def get_free_gpu():
        try:
            lines = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').readlines()
        except Exception as e:
            print('error getting free memory', e)
            return {0: 10000, 1: 10000, 2: 0, 3: 10000, 4: 0, 5: 0, 6: 0, 7: 0}

        memory_available = [int(x.split()[2]) for x in lines]
        return {index: mb for index, mb in enumerate(memory_available)}

    gpus = get_free_gpu()
    write_gpus_to_file(gpus)
    return {index: mega for index, mega in gpus.items() if mega >= minimum_free_giga * 1000}
    # return [str(index) for index, mega in gpus.items() if mega > minimum_free_giga * 1000]


force_cpu = False


def get_torch(forcing_cpu=False):
    if forcing_cpu:
        print('using cpu')
        import torch
        return torch

    global force_cpu
    force_cpu = force_cpu
    gpus = get_index_of_free_gpus()
    gpus = list(map(str, gpus))[:max_num_gpus]
    join = ','.join(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = join
    print('setting CUDA_VISIBLE_DEVICES=' + join)
    if max_num_gpus == 1:
        print('remember you are working with 1 gpu:(.. probably should fix gpu index indent')
    import torch
    return torch


def print_size_of_model(model):
    get_torch().save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


def get_device():
    torch = get_torch()
    if force_cpu:
        return torch.device('cpu')
    gpus = get_index_of_free_gpus()
    print(gpus)
    return torch.device(compute_gpu_indent(gpus) if torch.cuda.is_available() else 'cpu')


def compute_gpu_indent(gpus):
    try:
        # return 'cuda'
        best_gpu = max(gpus, key=lambda gpu_num: gpus[int(gpu_num)])
        indented_gpu_index = list(gpus.keys()).index(best_gpu)
        return 'cuda:' + str(indented_gpu_index)
    except:
        return 'cuda'


def get_device_and_set_as_global():
    d = get_device()
    get_torch().cuda.set_device(d)
    return d


def is_university_server():
    try:
        return 'gamir' in os.environ['HOST']
    except:
        return False


def get_cache_dir():
    if is_university_server():
        return '/specific/netapp5_3/ML_courses/students/DL2020/glickman1/cache'
    return None
