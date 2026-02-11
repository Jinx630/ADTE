import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import argparse
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from data.datautils import prepare_imagenet_classnames
from data.imagenet_prompts import imagenet_classes

from utils.tools import (
    Summary, AverageMeter, ProgressMeter, 
    accuracy, display_results, print, arg_in_results, seed_everything
)
from utils.best_view_experimental import *
from data.cls_to_names import *
from data import prepare_dataset

from ttas import get_tta_module

import torch

def acc(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def compute_P(memory_bank, logit_gap):
    K = memory_bank.shape[0]
    P = torch.zeros(K, K).to(memory_bank.device)
    for j in range(K):
        valid_indices = (logit_gap[j] != -1).nonzero(as_tuple=True)[0]
        valid_samples = memory_bank[j][valid_indices]
        Nj = len(valid_samples)
        if Nj > 0:
            softmax_values = torch.softmax(valid_samples, dim=1)
            P[:, j] = torch.sum(softmax_values, dim=0) / Nj
    return P


def estimate_q(P, epsilon=1e-4, n=500):
    K = P.shape[0]
    q = torch.ones(K).to(P.device) / K
    I = torch.eye(K).to(P.device)
    for _ in range(n):
        # 加入L2正则化项
        q_new = torch.matmul(P, q)
        q_new = q_new / torch.norm(q_new)  # 归一化，确保q的范数为1
        if torch.norm(q_new - q) < epsilon:
            break
        q = q_new
    return q

def compute_q(memory_bank, logit_gap):

    P = compute_P(memory_bank, logit_gap)
    q = estimate_q(P)

    return torch.log(q)

def create_results_filename(tta_module, args):
    alg_name = str(tta_module)
    name = f"{alg_name}_{args.arch.replace('/', '-')}_{args.pretrained}"
    if args.maple:
        name += f"_maple"
    name += f"_seed{args.seed}"
    name = name.replace("/", "_")
    if args.reward_arch is not None:
        name += f"_r{args.reward_arch.replace('/', '-')}"
    if args.templates:
        name += f"_templates"
    return name


def augment_results(results, args):
    results = arg_in_results(results, "seed", args.seed)
    results = arg_in_results(results, "arch", args.arch)
    # results = arg_in_results(results, "pretrained", args.pretrained)
    results = arg_in_results(results, "templates", bool(args.templates))
    # results = arg_in_results(results, "maple", bool(args.maple))
    results = arg_in_results(results, "scale_w", args.scale_w)
    return results

def min_max_normalize(data, a=0.1, b=1.0):
    """
    对输入的 PyTorch 张量进行 Min - Max 归一化，将数据缩放到 [a, b] 区间
    :param data: 输入的 PyTorch 张量
    :param a: 缩放区间的下限
    :param b: 缩放区间的上限
    :return: 归一化后的 PyTorch 张量
    """
    min_val = torch.min(data)
    max_val = torch.max(data)
    # 避免分母为零
    if max_val - min_val == 0:
        return torch.full_like(data, a)
    normalized_data = a + (data - min_val) * (b - a) / (max_val - min_val)
    return normalized_data

def sigmoid_normalize(x, a=0.1, b=1.0):
    z = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y1 = 1 / (1 + torch.exp(-z))
    y = a + (b - a) * y1
    return y

def z_score_normalize(data, a=0.1, b=1.0):
    # 计算均值和标准差
    mean = torch.mean(data)
    std = torch.std(data)
    # 进行 Z - Score 标准化
    z = (data - mean) / std
    # 找出标准化后数据的最小值和最大值
    z_min = torch.min(z)
    z_max = torch.max(z)
    # 将标准化后的数据映射到指定范围 [a, b]
    normalized_data = a + (b - a) * (z - z_min) / (z_max - z_min)
    return normalized_data


def decimal_scaling_custom_normalize(data, a=0.1, b=1.0):
    # 计算缩放因子 j
    j = torch.ceil(torch.log10(torch.max(torch.abs(data))))
    # 进行小数定标归一化
    scaled_data = data / (10 ** j)
    # 找出缩放后数据的最小值和最大值
    scaled_min = torch.min(scaled_data)
    scaled_max = torch.max(scaled_data)
    # 将缩放后的数据映射到指定范围 [a, b]
    normalized_data = a + (b - a) * (scaled_data - scaled_min) / (scaled_max - scaled_min)
    return normalized_data

def main(args):
    
    # reproducibility
    torch.use_deterministic_algorithms(True)
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True) # warn_only needed to allow for torch.topk
    cudnn.benchmark = True
    set_id = args.set_id

    if set_id in ['A', 'R', 'K', 'V', 'I']:
        classnames = prepare_imagenet_classnames(set_id, imagenet_classes)
    elif len(set_id) > 1: 
        classnames = eval("{}_classes".format(set_id.lower()))
    
    classnames = [name.replace("_", " ") for name in classnames]

    # create TTA-ed module
    tta_module = get_tta_module(
        args.tuner, 
        **dict(model=args.model, 
        arch=args.arch, 
        use_templates=args.templates,
        use_cupl=args.cupl,
        pretrained=args.pretrained,
        gpu=args.gpu, 
        ctx_init=args.ctx_init,
        maple_weights=args.maple,
        reward_arch=args.reward_arch,
        reward_pretrained=args.reward_pretrained,
        seed=args.seed,
        classnames=classnames,
        pre_compute_q=args.pre_compute_q)
    )
    tta_module = tta_module.to(args.gpu)
    
    # iterating through eval datasets
    results = {}
    print("=> Evaluating on testset [{}]".format(set_id))

    # create dataset
    val_dataset = prepare_dataset(tta_module, set_id, args.num_views, args.resolution, args.dataset_mode)
        
    # create dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, # episodic TTA!
        shuffle=True, # irrelevant 
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=False
    )

    batch_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, # episodic TTA!
        shuffle=False, # irrelevant 
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=False
    )
    print("=> Number of test samples {} (# classes = {})".format(len(val_loader), len(val_dataset.classes)))

    # prepare the model for the current dataset
    # (set the class names and their embeddings for the current dataset)
    tta_module.prepare_for_training(set_id, args.arch)
    
    # run evaluation over the dataset
    results[set_id] = test_time_adapt_eval(val_loader, batch_loader, tta_module, args)
    
    # log and release memory
    accs = [results[set_id][k] for k in results[set_id] if "Acc" in k]
    print(f"=> Testset [{set_id}]: Acc@1 {accs[0]:.2f}/ Acc@5 {accs[1]:.2f} / Acc@10 {accs[2]:.2f}\n")
    del val_dataset, val_loader

    # create the folder
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # create the filename (including parameters)
    name = create_results_filename(tta_module, args)
    output_path = os.path.join(results_dir, f"{name}.csv")
    
    results = augment_results(results, args)
    display_results(results, save_to=output_path)


def test_time_adapt_eval(val_loader, batch_loader, tta_module, args):
    
    # initialize meters
    batch_time = AverageMeter(' Time', ':4.3f', Summary.AVERAGE)
    top1 = AverageMeter(' Acc@1', ':4.2f', Summary.AVERAGE)
    top5 = AverageMeter(' Acc@5', ':4.2f', Summary.AVERAGE)
    top10 = AverageMeter(' Acc@10', ':4.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        num_batches=len(val_loader),
        meters=[batch_time, top1, top5, top10],
        prefix='Test: '
    )

    # reset model and switch to evaluate mode
    tta_module.eval()

    memory_size = 20
    memory_bank = torch.zeros(tta_module.class_num, memory_size, tta_module.class_num).to(tta_module.memory_bank.device)
    logit_gap = torch.full((tta_module.class_num, memory_size), -1., dtype=torch.float).to(tta_module.memory_bank.device)

    if args.pre_compute_q:
        if os.path.exists(args.q_path):
            q = torch.load(args.q_path, map_location=tta_module.memory_bank.device, weights_only=True)
        else:
            print("Computing Bias...")
            for i, (images, target) in tqdm(enumerate(batch_loader)):
                images = [img.to(args.gpu, non_blocking=True) for img in images]
                images = torch.cat(images, dim=0)
                with torch.amp.autocast(device_type='cuda'):
                    logits = tta_module(images, target, is_zero=True)
                    # full_idx = torch.argsort(logits.logsumexp(1), descending=False)[:32]
                    origin_output = logits.mean(0)
                    top_1, top_2 = torch.topk(origin_output, 2)[0]
                    logit_diff = top_1 - top_2

                    if logit_diff >= 1.0:
                        pseudo_label = torch.argmax(origin_output)
                        min_index = torch.argmin(logit_gap[pseudo_label])
                        memory_bank[pseudo_label][min_index] = origin_output
                        logit_gap[pseudo_label][min_index] = logit_diff

            q = compute_q(memory_bank, logit_gap)
            torch.save(q, args.q_path)
        tta_module.q = q
        
    # min-max norm
    if args.pre_compute_q:
        copy_q = copy.deepcopy(q)
        scale_w = min_max_normalize(copy_q, a=0.01, b=0.9).to(tta_module.memory_bank.device)
        if args.same_q:
            tta_module.scale_w = args.scale_w
        else:
            tta_module.scale_w = scale_w

    if args.best_views:
        comb = combination(32, 3) - 1
        criterion = nn.CrossEntropyLoss(reduction='none')

    # iterate through the validation set
    for i, (images, target) in enumerate(val_loader):
        
        # move the data to the GPU
        target = target.to(args.gpu, non_blocking=True)
        images = [img.to(args.gpu, non_blocking=True) for img in images]
        images = torch.cat(images, dim=0)

        # tta with zero temp is implemented in the forward pass of the model
        with torch.amp.autocast(device_type='cuda'):
            
            # measure tta time with cuda events
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()

            # actual tta here
            min_loss = 100000
            best_idx = -1
            if args.best_views:
                output = tta_module(images, target, args.is_zero) - q
                new_outputs = output[comb].mean(dim=1)
                new_o_loss = criterion(new_outputs, target.repeat(new_outputs.shape[0]))
                best_idx = torch.argmin(new_o_loss, dim=0)
                output = output[comb[best_idx], :].mean(0)[None]
            else: 
                output = tta_module(images, target, args.is_zero, args.use_adte)[:1]

            # finish measuring time
            end_event.record()
            torch.cuda.synchronize()
            
        # measure accuracy
        acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
        top1.update(val=acc1[0], n=1)
        top5.update(val=acc5[0], n=1)
        top10.update(val=acc10[0], n=1)

        # print(time.time() - t2)

        # measure elapsed time and display updates
        batch_time.update(start_event.elapsed_time(end_event)/1000, n=1)
        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            progress.display(i+1)

    # when evaluation over the shard of each rank is finished, we must gather the AverageMeters from the different ranks
    print("=> Finished evaluation.")
    results_dict = dict(zip([m.name for m in progress.meters], [m.avg for m in progress.meters]))
    return results_dict

if __name__ == '__main__':
    import ttas as ttas
    parser = argparse.ArgumentParser(description='Test-Time Adaptation with Zero Temperature for Vision-Language Models.')

    parser.add_argument('--best_views', action='store_true', help='tmd is getting few best view (experimental)')

    # parameters for the TTA method
    parser.add_argument('-t', '--tuner', type=str, default='Zero', help='tuner to use: TPT/MEMO', choices=ttas.__all__)
    parser.add_argument('--ctx_init', default="a_photo_of_a", type=str, help="underscore separated context, such as 'a_photo_of_a' ")
    parser.add_argument('--num_views', default=64, type=int, help='number of views for TTA')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--pre_compute_q', type=int, default=1)
    parser.add_argument('--same_q', action='store_true')
    parser.add_argument('--is_zero', action='store_true')
    parser.add_argument('--use_adte', action='store_true')
    parser.add_argument('--q_path', type=str, default="bias_pt/imagenet_A_q_1_10_ensemble_B16.pt")
    parser.add_argument('--scale_w', type=float, default=0.5)

    # model parameters
    parser.add_argument('-m', '--model', type=str, default='clip', help='model to use: clip/vit', choices=['clip'])
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('--pretrained', type=str, default="openai", help="Pretrained Keyword for the OpenCLIP repo. \
                        Default: \"openai\", will also use OpenAI's implementation of CLIP.")
    parser.add_argument('--templates', action="store_true", help="Use textual templates (+Ensemble in the paper).")
    parser.add_argument('--cupl', action="store_true", help="Use cupl text descriptions")
    
    # data parameters
    from data.datautils import ID_to_DIRNAME
    parser.add_argument('--set_id', type=str, default="I", help='ID of the Test Dataset (case sensitive).', choices=list(ID_to_DIRNAME.keys()))
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')

    # hardware arguments
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--results_dir', type=str, default='results', help='directory to save results')

    # development arguments
    parser.add_argument('-p', '--print_freq', default=500, type=int, metavar='N', help='print frequency (default: 10)')

    # arguments for Reinforcement Learning from CLIP Feedback (optional)
    parser.add_argument('--reward_arch', type=str, default=None, help='reward model to use (optional)')
    parser.add_argument('--reward_pretrained', type=str, default=None, help="Enables using OpenCLIP models with ZeroRLCF. Please see the --pretrained flag for more details.")

    parser.add_argument('--maple', action="store_true", help='Use MaPLe weights. Will load a different pretraining based on the seed.')
    args = parser.parse_args()
    
    # setup tensor cores 
    torch.set_float32_matmul_precision("medium")

    main(args)