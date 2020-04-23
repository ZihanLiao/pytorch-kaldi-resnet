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

from datasets import SequenceDataset, EmbeddingDataset
from model import NeuralSpeakerModel

def printf(format, *args):
    sys.stdout.write(format % args)

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--spk_num', type=int, help='number of speakers')
parser.add_argument('--arch', type=str, required=True, help='feature extractor model type')
parser.add_argument('--input-dim', type=int, required=True, help='input feature dimension')
parser.add_argument('--pooling', type=str, required=True, help='mean or mean+std')
parser.add_argument('--chunk-size', default=-1, type=int,
                    help='minimum feature map length')
parser.add_argument('--model-path', help='trained model (.h5)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', type=int,
                    help='GPU id to use.')
parser.add_argument('--gpu-num', default=-1, type=int,
                    help='GPU nums to use.')
parser.add_argument('--decode-scp', help='decode.scp')
parser.add_argument('--out-path', help='output file path')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

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
    print("=> creating model '{}'".format(args.arch))
    model = NeuralSpeakerModel(spk_num=args.spk_num, feat_dim=args.input_dim, pooling=args.pooling)

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
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            #model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

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
        #model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.loadParameters(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model_path))
        return

    embed_dataset = EmbeddingDataset(scp_file=args.decode_scp, chunk_size=args.chunk_size)
    if args.distributed:
        embed_sampler = torch.utils.data.distributed.DistributedSampler(embed_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    else:
        embed_sampler = None

    embed_loader = torch.utils.data.DataLoader(
        embed_dataset, batch_size=args.batch_size, shuffle=(embed_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=embed_sampler)
    print("=> args.world_size: {}, args.rank: {}, loaded embedding samples num: {}".format(args.world_size, args.rank, len(embed_loader)))

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    SequenceGenerator(embed_loader, model, args.out_path, args)


def SequenceGenerator(loader, model, out_path, args):
    model.eval()
    if args.gpu is not None:
        #f = open(os.path.join(out_path, arg.gpu), 'w')
        f = open(out_path+'/'+str(args.gpu), 'w')
    else:
        f = open(out_path+'/'+'alone', 'w')
    with torch.no_grad():
        #for i, (utt, audios) in enumerate(loader):
        for i, (audios, utts) in enumerate(loader):
            if args.gpu is not None:
                audios = audios.cuda(args.gpu, non_blocking=True)
            #print('{} feature size: {}'.format(utts, audios.shape))
            pred = model.predict(audios)
            pred = pred.cpu().data.numpy()
            #print('{} predicts size: {}'.format(utts, pred.shape))
            #pred=model.predict(torch.from_numpy(np.array(y,dtype=np.float32)).to(device)).cpu().data.numpy().flatten()
            for i in range(pred.shape[0]):
                out = pred[i,:].flatten()
                utt = utts[i][0] 
                #print('{} predict size: {}'.format(utt, out.shape))
                f.write(utt +' [ '+' '.join(map(str, out))+' ]\n')
            #return
    f.close()


if __name__ == '__main__':
    main()
