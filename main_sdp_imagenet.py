import argparse
import os
import shutil
import time
import math

from math import cos, pi

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import models
import logging

from models import BinarizeConv2dSDP
import copy

from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging

def fast_collate(batch, memory_format):
    """Based on fast_collate from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

def parse():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                        help='evaluate model FILE on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--disable_dali', default=False, action='store_true',
                        help='Disable DALI data loader and use native PyTorch one instead.')
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument('--fp16-mode', default=False, action='store_true',
                        help='Enable half precision mode.')
    parser.add_argument('--loss-scale', type=float, default=1)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    
    parser.add_argument('-K', default=2, type=int, help='K value')
    parser.add_argument('-scale', default=1, type=float, help='variance scale hyper-parameter')
    parser.add_argument('-L', type=float, default=10, metavar='sampling frequency', help='sample 2K+L*K')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
    parser.add_argument('-save', metavar='SAVE', default='',
                        help='saved folder')
    
    parser.add_argument('--lr_decay', type=str, default='MSteps')
    parser.add_argument('--binary_opt', type=str, default='ReCU')

    args = parser.parse_args()
    return args

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels

def sample_uniform_int(a, b):
    # Sample a single integer from a uniform distribution between a and b (inclusive)
    sampled_number = torch.randint(a, b + 1, (1,))
    
    return sampled_number.item()


writer = SummaryWriter()
train_loss_idx_value = 0
val_loss_idx_value = 0

def main():
    global best_prec1, args, writer, train_loss_idx_value, val_loss_idx_value
    best_prec1 = 0
    args = parse()

    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logging(os.path.join(save_path, 'log.txt'))

    logging.info("saving to %s", save_path)
    logging.info(f"run arguments: {args}")

    if not len(args.data):
        raise Exception("error: No data set provided")

    if args.test:
        logging.info("Test mode - only 10 iterations")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

    if args.local_rank == 0:
        logging.info("fp16_mode = {}".format(args.fp16_mode))
        logging.info("loss_scale = {}".format(args.loss_scale))
        logging.info("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    if args.local_rank == 0:
        logging.info(f"creating model: {args.arch}")
    model = models.__dict__[args.arch]
    model_config = {'K': args.K, 'scale': args.scale}
    model = model(**model_config)

    if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
    else:
        model = model.cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.distributed:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        torch.cuda.current_stream().wait_stream(s)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logging.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                val_loss_idx_value, train_loss_idx_value = checkpoint['epoch'], checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.local_rank == 0:
                    logging.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
            else:
                if args.local_rank == 0:
                    logging.info("=> no checkpoint found at '{}'".format(args.resume))
        resume()
    else:
        if args.evaluate:
            if not os.path.isfile(args.evaluate):
                print('invalid checkpoint: {}'.format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded checkpoint '%s' (epoch %s)",
                        args.evaluate, checkpoint['epoch'])

    # Data loading code
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
        valdir = os.path.join(args.data[0], 'val')
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if args.arch == "inception_v3":
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    train_loader = None
    val_loader = None
    if not args.disable_dali:
        train_pipe = create_dali_pipeline(batch_size=args.batch_size,
                                          num_threads=args.workers,
                                          device_id=args.local_rank,
                                          seed=12 + args.local_rank,
                                          data_dir=traindir,
                                          crop=crop_size,
                                          size=val_size,
                                          dali_cpu=args.dali_cpu,
                                          shard_id=args.local_rank,
                                          num_shards=args.world_size,
                                          is_training=True)
        train_pipe.build()
        train_loader = DALIClassificationIterator(train_pipe, reader_name="Reader",
                                                  last_batch_policy=LastBatchPolicy.PARTIAL,
                                                  auto_reset=True)

        val_pipe = create_dali_pipeline(batch_size=args.batch_size,
                                        num_threads=args.workers,
                                        device_id=args.local_rank,
                                        seed=12 + args.local_rank,
                                        data_dir=valdir,
                                        crop=crop_size,
                                        size=val_size,
                                        dali_cpu=args.dali_cpu,
                                        shard_id=args.local_rank,
                                        num_shards=args.world_size,
                                        is_training=False)
        val_pipe.build()
        val_loader = DALIClassificationIterator(val_pipe, reader_name="Reader",
                                                last_batch_policy=LastBatchPolicy.PARTIAL,
                                                auto_reset=True)
    else:
        train_dataset = datasets.ImageFolder(traindir,
                                             transforms.Compose([transforms.RandomResizedCrop(crop_size),
                                                                 transforms.RandomHorizontalFlip()]))
        val_dataset = datasets.ImageFolder(valdir,
                                           transforms.Compose([transforms.Resize(val_size),
                                                               transforms.CenterCrop(crop_size)]))

        train_sampler = None
        val_sampler = None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        collate_fn = lambda b: fast_collate(b, memory_format)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   collate_fn=collate_fn)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler,
                                                 collate_fn=collate_fn)

    if args.evaluate:
        top1, top5 = validate(val_loader, model, criterion)
        print(f'top1: {top1}, top5: {top5}')
        return

    scaler = torch.cuda.amp.GradScaler(init_scale=args.loss_scale,
                                       growth_factor=2,
                                       backoff_factor=0.5,
                                       growth_interval=100,
                                       enabled=args.fp16_mode)
    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # weight_histograms(writer, epoch, model)
        if args.local_rank == 0:
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        # train for one epoch
        avg_train_time = train(train_loader, model, criterion, scaler, optimizer, epoch)
        total_time.update(avg_train_time)
        if args.test:
            break

        # evaluate on validation set
        [prec1, prec5] = validate(val_loader, model, criterion)
        if args.local_rank == 0:
            weight_histograms(writer, epoch, model, args.scale)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            if epoch == args.epochs - 1:
                logging.info('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(
                      prec1,
                      prec5,
                      args.total_batch_size / total_time.avg))
            writer.add_scalar("best_prec1", best_prec1, epoch)

class data_prefetcher():
    """Based on prefetcher from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def __iter__(self):
        return self

    def __next__(self):
        """The iterator was added on top of the orignal example to align it with DALI iterator
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        if input is None:
            raise StopIteration
        return input, target



def weight_histograms_W(writer, step, weights, layer_number):
  weights_shape = weights.shape
  num_kernels = weights_shape[0]
  for k in range(num_kernels):
    flattened_weights = weights[k].flatten()
    tag = f"layer_{layer_number}/M_{k}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_Z(writer, step, Z, layer_number):
    Z_shape = Z.shape
    num_kernels = Z_shape[0]
    for k in range(num_kernels):
        flattened_Z = Z[k].flatten()
        tag = f"layer_{layer_number}/Z_{k}"
        writer.add_histogram(tag, flattened_Z, global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model, scale):
    print("Visualizing model weights...")
    layer_number = 0
    for module in model.modules():
        if isinstance(module, BinarizeConv2dSDP):
            m = copy.deepcopy(module.M)
            z = copy.deepcopy(module.Z)
            m_shape, z_shape = m.shape, z.shape
            m = m.view(-1)
            z = z.view(args.K, m.shape[0])
            # visualize m and z after normalization.
            A = m*m + torch.sum(z.T**2, dim=1)/scale
            m.data = m.data / torch.sqrt(A)
            z.data = z.data / torch.sqrt(A)

            m = m.view(m_shape)
            z = z.view(z_shape)

            weight_histograms_W(writer, step, m, layer_number)
            weight_histograms_Z(writer, step, z, layer_number)
            layer_number += 1


def train(train_loader, model, criterion, scaler, optimizer, epoch):
    global writer, train_loss_idx_value
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    if args.disable_dali:
        data_iterator = data_prefetcher(train_loader)
        data_iterator = iter(data_iterator)
    else:
        data_iterator = train_loader

    for i, data in enumerate(data_iterator):
        if args.disable_dali:
            input, target = data
            train_loader_len = len(train_loader)
        else:
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
            train_loader_len = int(math.ceil(data_iterator._size / args.batch_size))

        if args.prof >= 0 and i == args.prof:
            logging.info("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        if args.lr_decay == 'cos':
            adjust_learning_rate_cos(optimizer, epoch, i, train_loader_len)
        elif args.lr_decay == 'MSteps':
            adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        if args.test:
            if i > 10:
                break

        with torch.cuda.amp.autocast(enabled=args.fp16_mode):
            output = model(input)
            loss = criterion(output, target)

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")

        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        scaler.scale(loss).backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        scaler.step(optimizer)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        scaler.update()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()

            if args.local_rank == 0:
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, train_loader_len,
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            logging.info("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

    if args.local_rank == 0:
        logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Losses {losses.avg:.3f}'
            .format(top1=top1, top5=top5, losses=losses))
        writer.add_scalar("Train Loss", losses.avg, train_loss_idx_value)
        train_loss_idx_value += 1
        
    return batch_time.avg

def validate(val_loader, model, criterion):
    global writer, val_loss_idx_value
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    if args.disable_dali:
        data_iterator = data_prefetcher(val_loader)
        data_iterator = iter(data_iterator)
    else:
        data_iterator = val_loader

    for i, data in enumerate(data_iterator):
        if args.disable_dali:
            input, target = data
            val_loader_len = len(val_loader)
        else:
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
            val_loader_len = int(math.ceil(data_iterator._size / args.batch_size))

        # compute output
        L = int(args.L)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            total_loss = loss
            total_output = output

            for j in range(L-1):
                output = model(input)
                loss = criterion(output, target)
                total_loss += loss
                total_output += output

            loss = total_loss / L
            output = total_output / L

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, val_loader_len,
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    if args.local_rank == 0:
        logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Losses {losses.avg:.3f}'
            .format(top1=top1, top5=top5, losses=losses))
        writer.add_scalar("Validation Loss", losses.avg, val_loss_idx_value)
        val_loss_idx_value += 1
        
    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best):
    save_path = os.path.join(args.results_dir, args.save)
    model_path = os.path.join(save_path, 'checkpoint.pth.tar')
    best_model_path = os.path.join(save_path, 'model_best.pth.tar')
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, best_model_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_cos(optimizer, epoch, step, len_epoch):
    # first 5 epochs for warmup
    warmup_iter = 5 * len_epoch
    current_iter = step + epoch * len_epoch
    max_iter = args.epochs * len_epoch
    lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    if epoch < 5:
        lr = args.lr * current_iter / warmup_iter

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()