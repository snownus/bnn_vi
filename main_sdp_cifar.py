import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from torchsummary import summary
import math
import numpy as np
import random
import copy

from math import cos, pi

from models import BinarizeConv2dSDP
from torch.utils.tensorboard import SummaryWriter

seed_value = 2020   

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  

torch.manual_seed(seed_value)     
torch.cuda.manual_seed(seed_value)      
torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--fp_regime', default="{0: {'optimizer': 'Adam','lr':1e-3}}", type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--binarization', default='det', type=str, help='binarization function:det or threshold (default: det)')
parser.add_argument('-t', '--threshold', default=1e-8, type=float,
                    help='optimization threshold (default: 1e-8)')
parser.add_argument('-K', default=2, type=int, help='K value')
parser.add_argument('-scale', default=1, type=float, help='variance scale hyper-parameter')
parser.add_argument('-iters', default=1, type=int, help='init iters for gradients')

parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 3e-4)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='BayesBiNN momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='weight decay',
                        help='weight decay (default: 1e-4)')
parser.add_argument('-L', type=float, default=10, metavar='sampling frequency', 
                    help='sample 2K+L*K')
parser.add_argument('--seed', type=int, default=2020, metavar='sampling frequency', help='seed value')
parser.add_argument('--milestones', metavar='N', type=int, nargs='+', help='milestones')

parser.add_argument('--lr_decay', type=str, default='MSteps')


writer = SummaryWriter()
train_loss_idx_value = 0
val_loss_idx_value = 0


def main():
    global args, best_prec1, writer, train_loss_idx_value, val_loss_idx_value
    args = parser.parse_args()
    seed_value = args.seed
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  

    torch.manual_seed(seed_value)     
    torch.cuda.manual_seed(seed_value)      
    torch.cuda.manual_seed_all(seed_value)   

    best_prec1 = 0

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.info("run arguments: %s", args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus 
    cudnn.benchmark = True

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    print(f'args.input_size: {args.input_size}')
    model_config = {'K': args.K, 'scale': args.scale, 'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config != '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    model= nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)
    elif args.pretrained:
        checkpoint_file = args.pretrained
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.pretrained)
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }

    transform = getattr(model, 'input_transform', default_transform)
    fp_regime = getattr(model, 'fp_regime', eval(args.fp_regime))

    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        loss, top1, top5 = validate(val_loader, model, criterion, 0)
        print(f'loss: {loss}, top1: {top1}, top5: {top5}, L: {args.L}')
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    binary_num_parameters = 0
    fp_num_parameters = 0
    for l in list(model.parameters()):
        if hasattr(l,'binary'):
            binary_num_parameters = binary_num_parameters + l.nelement()
        else:
            fp_num_parameters = fp_num_parameters + l.nelement()
    logging.info("number of parameters: %d", num_parameters)
    logging.info("number of binary parameters: %d", binary_num_parameters)
    logging.info("number of full precision parameters: %d", fp_num_parameters)

    fp_optimizer = torch.optim.SGD([{'params':model.parameters()}], lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(train_loader, model, criterion, epoch, fp_optimizer)
        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)

        writer.add_scalar("LR", fp_optimizer.param_groups[0]['lr'], epoch)
        logging.info(f"lr:{fp_optimizer.param_groups[0]['lr']} at epoch:{epoch}")

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        writer.add_scalar("best_prec1", best_prec1, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'fp_regime': fp_regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \t'
                     'Best Prec@1 {best_prec1:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5, best_prec1=best_prec1))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=train_prec1, val_error1=val_prec1,
                    train_error5=train_prec5, val_error5=val_prec5)
        results.save()


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


def sample_uniform_int(a, b):
    # Sample a single integer from a uniform distribution between a and b (inclusive)
    sampled_number = torch.randint(a, b + 1, (1,))
    
    return sampled_number.item()


def forward(data_loader, model, criterion, epoch=0, training=True, fp_optimizer=None):
    global writer, train_loss_idx_value, val_loss_idx_value
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    iters = 0
    for i, (inputs, target) in enumerate(data_loader):
        train_loader_len = len(data_loader)
        if args.lr_decay == 'cos' and training:
            adjust_learning_rate_cos(fp_optimizer, epoch, i, train_loader_len)
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            inputs = inputs.cuda()
            target = target.cuda()

        L = int(args.L)
        if not training:
            # model.eval()
            with torch.no_grad():
                output = model(inputs)
                loss = criterion(output, target)
                total_loss = loss
                total_output = output

                for j in range(L-1):
                    output = model(inputs)
                    loss = criterion(output, target)
                    total_loss += loss
                    total_output += output
                total_loss /= L
                total_output /= L
        else:
            output = model(inputs)
            loss = criterion(output, target)
            loss.backward()
            total_loss = loss
            total_output = output

            fp_optimizer.step()
            fp_optimizer.zero_grad()

        params = model.named_parameters()
        for name, param in params:
            if 'sample' in name and 'downsample' not in name:
                param.zero_()

        if type(total_output) is list:
            total_output = total_output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(total_output.data, target, topk=(1, 5))
        losses.update(total_loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and i != 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))
        
        if not training:
            writer.add_scalar("Validation Loss", total_loss, val_loss_idx_value)
            val_loss_idx_value += 1
        else:
            writer.add_scalar("Training Loss", total_loss, train_loss_idx_value)
            train_loss_idx_value += 1

    # if not training:
    #     weight_histograms(writer, epoch, model, args.scale)
    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, fp_optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, fp_optimizer=fp_optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, fp_optimizer=None)


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


if __name__ == '__main__':
    main()