import time
import torch


def load_session():
    # PyTorch 不需要显式加载会话，但可以设置 GPU 配置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or total == 0:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def generate_out_folder(out_folder, training_data_path, div_path, method_name):
    params = training_data_path.strip('/').split('/')
    print(out_folder, training_data_path, params, div_path, method_name)
    path = params[-1]
    folder = out_folder + method_name + '/' + path + "/" + div_path + str(time.strftime("%Y%m%d%H%M%S")) + "/"
    print("results output folder:", folder)
    return folder

# # 示例用法：
# device = load_session()
# print(f'Using device: {device}')