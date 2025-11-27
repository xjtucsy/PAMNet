import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.rppg_datasets import VIPL, UBFC, PURE, COHFACE, BUAA, MMPD, V4V
from losses.NPLoss import Neg_Pearson
from losses.CELoss import CrossEntropyKL
from archs.PAMNet import PAMNet

def _init_fn(seed=92):
    np.random.seed(seed)

def build_dataset(args, mode, batch_size):
    mode_num_rppg = {
        'train': args.num_rppg,
        'val_video': -1,
        'val_clip': args.num_rppg,
    }

    num_rppg = mode_num_rppg[mode]
    train_mode = 'train' if mode == 'train' else 'test'
    shuffle_flag = mode == 'train'

    if args.dataset == 'VIPL':
        dataset = VIPL(data_dir=args.dataset_dir, T=num_rppg, train=train_mode, w=args.img_size, h=args.img_size, fold=args.vipl_fold)
    # -------------------------------------------------------
    ## UBFC dataset
    elif args.dataset == 'UBFC':
        dataset = UBFC(data_dir=args.dataset_dir, T=num_rppg, train=train_mode, w=args.img_size, h=args.img_size)
    # -------------------------------------------------------
    ## PURE dataset
    elif args.dataset == 'PURE':
        dataset = PURE(data_dir=args.dataset_dir, T=num_rppg, train=train_mode, w=args.img_size, h=args.img_size)
    # -------------------------------------------------------
    ## COHFACE dataset
    elif args.dataset == 'COHFACE':
        dataset = COHFACE(data_dir=args.dataset_dir, T=num_rppg, train=train_mode, w=args.img_size, h=args.img_size)
    # -------------------------------------------------------
    ## BUAA dataset
    elif args.dataset == 'BUAA':
        dataset = BUAA(data_dir=args.dataset_dir, T=num_rppg, train=train_mode, w=args.img_size, h=args.img_size)
    # -------------------------------------------------------
    ## MMPD dataset
    elif args.dataset == 'MMPD':
        dataset = MMPD(data_dir=args.dataset_dir, T=num_rppg, train=train_mode, w=args.img_size, h=args.img_size)
    # -------------------------------------------------------
    ## V4V dataset
    elif args.dataset == 'V4V':
        dataset = V4V(data_dir=args.dataset_dir, T=num_rppg, train=train_mode, w=args.img_size, h=args.img_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=_init_fn,
    )
    
def build_optimizer(args, model):
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError
    return optimizer

def build_scheduler(args, optimizer):
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.factor,
            patience=args.patience,
            verbose=True,
            threshold=args.threshold,
            threshold_mode='rel',
            cooldown=0,
            min_lr=args.min_lr,
            eps=args.eps,
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max,
            eta_min=args.eta_min,
        )
    else:
        raise NotImplementedError
    return scheduler

def build_criterion(args):
    def _build_criterion(criterion_name):
        if criterion_name == 'np_loss':
            return Neg_Pearson()
        elif criterion_name == 'ce_loss':
            return CrossEntropyKL()
        else:
            raise NotImplementedError
    criterion_list = eval(args.loss)
    criterion_dict = {}
    for criterion_name in criterion_list:
        criterion_dict[criterion_name] = _build_criterion(criterion_name)
    return criterion_dict

def build_model(args):
    if args.model == 'PAMNet':
        model = PAMNet(image_size=(args.num_rppg, args.img_size, args.img_size), patches=(4,4,4),\
            dim=32, ff_dim=64, num_layers=6, dropout_rate=0.1, device=f'cuda:{args.gpu}')
    else:
        raise NotImplementedError
    return model  


        
        