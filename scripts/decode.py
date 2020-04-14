import argparse
import os
import sys
import time
import warnings
import numpy as np
from struct import unpack
from kaldi_io import read_mat_scp

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from datasets import SequenceDataset
from model import NeuralSpeakerModel

def printf(format, *args):
    sys.stdout.write(format % args)

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--spk_num', type=int, help='number of speakers')
parser.add_argument('--arch', type=str, required=True, help='feature extractor model type')
parser.add_argument('--input-dim', type=int, required=True, help='input feature dimension')
parser.add_argument('--pooling', type=str, required=True, help='mean or mean+std')
parser.add_argument('--model-path', help='trained model (.h5)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpu-num', default=-1, type=int,
                    help='GPU nums to use.')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--decode-scp', help='decode.scp')
parser.add_argument('--out-path', help='output file path')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def SequenceGenerator(model, file_name, out_file, args):
    model.eval()
    device = torch.device("cpu")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    f = open(out_file, 'w')
    with torch.no_grad():
        for lab, feature in read_mat_scp(file_name):
            x = np.array(feature)
            x = x.T
            print('{} feature size: {}'.format(lab, x.shape))
            if args.gpu is not None:
                x = torch.from_numpy(x).to(device)
                x = x.view(1, x.size(0), x.size(1))
                #x = x.cuda(args.gpu, non_blocking=True)
            pred = model.predict(x)
            pred = pred.cpu().data.numpy().flatten()
            #pred=model.predict(torch.from_numpy(np.array(y,dtype=np.float32)).to(device)).cpu().data.numpy().flatten()
            f.write(lab+' [ '+' '.join(map(str, pred.tolist()))+' ]\n')
            #return
    f.close()

def main():
    args = parser.parse_args() 

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.gpu_num == -1:
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = min(torch.cuda.device_count(), args.gpu_num)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        print("not implemented")
        return
        #model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        #model = models.__dict__[args.arch]()
        #model = NeuralSpeakerModel(input_dim=args.input_dim, spk_num=args.spk_num, pooling='mean')
        model = NeuralSpeakerModel(spk_num=args.spk_num, feat_dim=args.input_dim, pooling=args.pooling)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('===> Model total parameter: {}'.format(model_params))
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            #args.batch_size = int(args.batch_size / ngpus_per_node)
            #args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            #print("=> workers per gpu: {}, batch size per gpu: {}".format(args.workers, args.batch_size))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()


    #checkpoint = torch.load(args.model_path, lambda a,b:a)

    #val = SequenceDataset(scp_file=args.cv_list, utt2spkid_file=args.utt2spkid, chunk_size=[args.max_chunk_size])
    #val_loader = torch.utils.data.DataLoader(
    #    val, batch_size=args.batch_size, shuffle=False,
    #    num_workers=args.workers, pin_memory=True)
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> loading checkpoint '{}'".format(args.model_path))
            if args.gpu is None:
                checkpoint = torch.load(args.model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.model_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

    SequenceGenerator(model, args.decode_scp, args.out_path, args)


if __name__ == '__main__':
    main()
