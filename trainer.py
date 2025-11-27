
'''
# ! docx for trainer.py
## ? NOTE_1 : change archs:
- 在 archs/MODEL.PY 中定义模型，在 forward 中首先取出 dict 值，返回值包装成 dict，然后在 utils/engine.py 中的 build_model 中调用
- 在 trainer.py -> train_one_epoch() 中，传递给模型的参数值先包装为 dict

## ? NOTE_2 : change losses:
- 在 losses/LOSS.PY 中定义损失函数，所有损失函数封装成类，在 __call__ 中定义前向
- 在 utils/engine.py 中的 build_criterion 中调用，注意 loss_func 的名字
- 在 trainer.py -> args 中，传递 args.loss 和 args.loss_weight
- 在 trainer.py 中，注意 args.loss, train_one_epoch(), train_losses 的一致性

## ? NOTE_3 : change datasets:
- 在 datasets/DATASET.PY 中定义数据集，所有数据集封装成类，__getitem__ 返回 dict
- 在 utils/engine.py 中的 build_dataset 中调用
- 在 trainer.py -> args 中，传递 args.dataset 和 args.num_rppg，注意 train_one_epoch() 中 unpack dict 的方式
'''

import torch, os
import numpy as np

import random
import time
import argparse
import logging
import shutil

from copy import deepcopy
from scipy import io as sio
from scipy import signal
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.engine import build_dataset, build_optimizer, build_scheduler, build_criterion, build_model
from utils.util import AvgrageMeter, pearson_correlation_coefficient, update_avg_meters, cal_psd_hr

from datasets.rppg_datasets import VIPL, UBFC, PURE, COHFACE


def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class RppgEstimatorTrainer:
    def __init__(self, args) -> None:
        self.args = args
        
        self.gpu_list = [int(i) for i in args.gpu.split(',')]
        self.gpu_num = len(self.gpu_list)
        self.actual_batch_size = args.batch_size * self.gpu_num
        self.is_multi_gpu = self.gpu_num > 1
        self.device = torch.device(f'cuda:{self.gpu_list[0]}')
        
        self.rppg_estimator = build_model(args).to(self.device)
        if self.is_multi_gpu:
            self.rppg_estimator = torch.nn.DataParallel(self.rppg_estimator, device_ids=self.gpu_list)

        ## generate save path
        self.run_date = time.strftime('%m%d_%H%M', time.localtime(time.time()))
        self.save_path = f'{args.save_path}/{self.run_date}'
        
        ## dataloader NOTE: SELECT YOUR DATASET
        self.train_dataloader = build_dataset(args, mode='train', batch_size=self.actual_batch_size)
        self.val_dataloader_video = build_dataset(args, mode='val_video', batch_size=1)
        self.val_dataloader_clip = build_dataset(args, mode='val_clip', batch_size=1)

        ## optimizer
        self.optimizer = build_optimizer(args, self.rppg_estimator)
        self.scheduler = build_scheduler(args, self.optimizer)
        self.loss_funcs = build_criterion(args)
        self.loss_funcs_weight = dict(zip(eval(args.loss), eval(args.loss_weight)))
        
        ## loss & metrics saver
        self.loss_meters = dict([(key, AvgrageMeter()) for key in self.loss_funcs.keys()])
        self.metrics_meters = {
            'mae': AvgrageMeter(),
        }

        ## constant
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).to(self.device)
        self.best_epoch = 0
        self.best_val_mae = 1000    # mean absolute error
        self.best_val_rmse = 1000   # root mean square error
        self.best_sd = 1000         # standard deviation
        self.best_r = 0             # Pearson’s correlation coefficient
        self.frame_rate = 30
    
    def prepare_train(self, start_epoch, continue_log):
        """Prepares the training process.

        Sets up the necessary directories for saving checkpoints and logs. Initializes the logger for logging training progress.
        Copies the current file to the save path. Loads the checkpoint if starting from a specific epoch.
        Sets the rppg_estimator to train mode.

        Args:
            start_epoch (int): The starting epoch for training.
            continue_log (str): The name of the log to continue from.

        Raises:
            Exception: If the rppg_estimator checkpoint file for the previous epoch is not found.

        Returns:
            None
        """
        if start_epoch != 0:
            self.save_path = f'{self.args.save_path}/{continue_log}'
            self.run_date = continue_log

        self.save_ckpt_path = f'{self.save_path}/ckpt'
        self.save_rppg_path = f'{self.save_path}/rppg'
        if not os.path.exists(self.save_ckpt_path):
            os.makedirs(self.save_ckpt_path)
        if not os.path.exists(self.save_rppg_path):
            os.makedirs(self.save_rppg_path)

        logging.basicConfig(filename=f'./logs/{self.args.model}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}.log',\
                            format='%(message)s', filemode='a')
        self.logger = logging.getLogger(f'./logs/{self.args.model}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}')
        self.logger.setLevel(logging.INFO)

        ## save proj_file to save_path
        cur_file = os.getcwd()
        cur_file_name = cur_file.split('/')[-1]
        shutil.copytree(f'{cur_file}', f'{self.save_path}/{cur_file_name}', dirs_exist_ok=True)

        if start_epoch != 0:
            if not os.path.exists(f'{self.save_ckpt_path}/rppg_estimator_{start_epoch - 1}.pth'):
                raise Exception(f'rppg_estimator ckpt file {start_epoch - 1} not found')
            self.rppg_estimator.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_{start_epoch - 1}.pth'))

        ## block gradient and set train
        self.rppg_estimator.train()
    
    def draw_rppg_ecg(self, rPPG, ecg, save_path_epoch, train=False, mini_batch=0):
        """Draws rPPG and ECG signals, saves the results, and plots the power spectral density.

        Filters the rPPG signal using a bandpass filter. Saves the filtered rPPG and ECG signals as a .mat file.
        Plots the rPPG and ECG signals, as well as the power spectral density, and saves the figure as a .jpg file.

        Args:
            rPPG (Tensor): The rPPG signal.
            ecg (Tensor): The ECG signal.
            save_path_epoch (str): The path to save the results.
            train (bool, optional): Whether it is in training mode. Defaults to False.
            mini_batch (int, optional): The mini-batch number. Defaults to 0.

        Returns:
            None
        """
        rPPG_sample, ecg_sample = rPPG[0], ecg[0]
        ## save the results
        b, a = signal.butter(2, [0.67 / 15, 3 / 15], 'bandpass')
        # 使用 lfilter 函数进行滤波 
        rPPG_np = rPPG_sample.cpu().data.numpy()
        rPPG_np = signal.lfilter(b, a, rPPG_np)
        y1 = rPPG_np
        y2 = ecg_sample.cpu().data.numpy()
        results_rPPG = [y1, y2]
        if not train:
            sio.savemat(
                os.path.join(save_path_epoch, 'test_rPPG.mat'),
                {'results_rPPG': results_rPPG},
            )
        else:
            sio.savemat(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.mat'), {'results_rPPG': results_rPPG})
        # show the ecg images
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        psd_pred = cal_psd_hr(rPPG_sample, self.frame_rate, return_type='psd')
        psd_gt = cal_psd_hr(ecg_sample, self.frame_rate, return_type='psd')
        ax[0].set_title('rPPG')
        ax[0].plot(y1, label='rPPG')
        ax[0].plot(y2, label='ecg')
        ax[0].legend()
        ax[1].set_title('psd')
        ax[1].plot(psd_pred.cpu().data.numpy(), label='pred')
        ax[1].plot(psd_gt.cpu().data.numpy(), label='gt')
        ax[1].legend()
        if not train:
            fig.savefig(os.path.join(save_path_epoch, 'test_rPPG.jpg'))
        else:
            fig.savefig(os.path.join(save_path_epoch, f'minibatch_{mini_batch+1:0>4}_rPPG.jpg'))
        plt.close(fig)

    def update_best(self, epoch, hr_pred, hr_gt, val_type='video'):
        """Updates the best validation metrics and saves the model if the current metrics are better.

        Calculates the mean absolute error (MAE), root mean squared error (RMSE), standard deviation (SD),
        and Pearson correlation coefficient (R) between the predicted and ground truth heart rates.
        If the current MAE is lower than the best MAE, updates the best MAE, RMSE, epoch, SD, and R,
        and saves the model.

        Args:
            epoch (int): The current epoch number.
            hr_pred (List[float]): The predicted heart rates.
            hr_gt (List[float]): The ground truth heart rates.
            val_type (str, optional): The type of validation. Defaults to 'video'.

        Returns:
            None
        """
        cur_mae = np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))
        cur_rmse = np.sqrt(np.mean(np.square(np.array(hr_gt) - np.array(hr_pred))))
        cur_sd = np.std(np.array(hr_gt) - np.array(hr_pred))
        cur_r = pearson_correlation_coefficient(np.array(hr_gt), np.array(hr_pred))

        if cur_mae < self.best_val_mae:
            self.best_val_mae = cur_mae
            self.best_val_rmse = cur_rmse
            self.best_epoch = epoch
            self.best_sd = cur_sd
            self.best_r = cur_r
            # save the model
            torch.save(
                self.rppg_estimator.state_dict(),
                os.path.join(self.save_ckpt_path, 'rppg_estimator_best.pth'),
            )

        self.logger.info(f'evaluate epoch {epoch}, total val {len(hr_gt)} ----------------------------------')
        self.logger.info(f'{val_type}-level mae of vq: {np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))}')
        self.logger.info(f'{val_type}-level cur mae: {cur_mae:.2f}, cur rmse: {cur_rmse:.2f}, cur sd: {cur_sd:.2f}, cur r: {cur_r:.4f}')
        self.logger.info(f'{val_type}-level best mae of vq: {self.best_val_mae:.2f}, best rmse: {self.best_val_rmse:.2f}, best epoch: {self.best_epoch}, ' \
                         f'best sd: {self.best_sd:.2f}, best r: {self.best_r:.4f}')
        self.logger.info(
            '------------------------------------------------------------------'
        )

    def evaluate_video(self, epoch = 0):
        """Evaluates the video data and saves the results.

        Evaluates the video data by processing it in clips and calculating the power spectral density (PSD) for each clip.
        Saves the PSD results and updates the best validation metrics.

        Args:
            epoch (int, optional): The current epoch number. Defaults to 0.

        Returns:
            None
        """
        save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
        hr_gt = []
        hr_pred = []

        with torch.no_grad():
            clip_len = 300
            for sample_batched in tqdm(self.val_dataloader_video):
                # get the inputs
                inputs, ecg, clip_average_HR = sample_batched['video'].to(self.device),\
                    sample_batched['ecg'].to(self.device), sample_batched['clip_avg_hr'].to(self.device)

                
                input_len = inputs.shape[2]
                num_clip = input_len // clip_len
                if num_clip == 0:
                    num_clip = 1
                    clip_len = clip_len - clip_len % 32
                # input_len = input_len - input_len % (num_clip * 32)
                # clip_len = input_len // num_clip
                inputs = inputs[:, :, :input_len, :, :]
                ecg = ecg[:, :input_len]

                new_args = deepcopy(self.args)
                new_args.num_rppg = clip_len
                val_rppg_estimator = build_model(new_args).to(self.device)
                if self.is_multi_gpu:
                    val_rppg_estimator = torch.nn.DataParallel(val_rppg_estimator, device_ids=self.gpu_list)
                val_rppg_estimator.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_{epoch}.pth'))
                val_rppg_estimator.eval()
                psd_gt_total = 0
                psd_pred_total = 0
                for idx in range(num_clip):

                    inputs_iter = inputs[:, :, idx*clip_len : (idx+1)*clip_len, :, :]
                    ecg_iter = ecg[:, idx*clip_len : (idx+1)*clip_len]

                    psd_gt = cal_psd_hr(ecg_iter, self.frame_rate, return_type='psd')
                    psd_gt_total += psd_gt.view(-1).max(0)[1].cpu() + 40

                    ## for rppg_estimator:
                    all_inputs = {
                        'input_clip': inputs_iter,
                        'epoch': epoch
                    }
                    outputs = val_rppg_estimator(all_inputs)
                    rPPG = outputs['rPPG']

                    psd_pred = cal_psd_hr(rPPG[0], self.frame_rate, return_type='psd')
                    psd_pred_total += psd_pred.view(-1).max(0)[1].cpu() + 40

                hr_pred.append(psd_pred_total / num_clip)
                hr_gt.append(clip_average_HR.cpu().numpy())
        ## save the results
        self.draw_rppg_ecg(rPPG, ecg_iter, save_path_epoch)
        self.update_best(epoch, hr_pred, hr_gt, val_type='video')

    def evaluate_clip(self, epoch = 0, optional_cliplen = None):
        """Evaluates the clip data and saves the results.

        Evaluates the clip data by calculating the power spectral density (PSD) for each clip.
        Saves the PSD results and updates the best validation metrics.

        Args:
            epoch (int, optional): The current epoch number. Defaults to 0.

        Returns:
            None
        """
        save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
        hr_gt = []
        hr_pred = []
        
        if optional_cliplen is not None:
            self.args.num_rppg = optional_cliplen
            self.val_dataloader_clip = build_dataset(self.args, mode='val_clip', batch_size=1)            

        val_rppg_estimator = build_model(self.args).to(self.device)
        if self.is_multi_gpu:
            val_rppg_estimator = torch.nn.DataParallel(val_rppg_estimator, device_ids=self.gpu_list)
        val_rppg_estimator.load_state_dict(torch.load(f'{self.save_ckpt_path}/rppg_estimator_{epoch}.pth'))
        val_rppg_estimator.eval()
        
       

        with torch.no_grad():
            for sample_batched in tqdm(self.val_dataloader_clip):
                # get the inputs
                inputs, ecg, clip_average_HR = sample_batched['video'].to(self.device),\
                    sample_batched['ecg'].to(self.device), sample_batched['clip_avg_hr'].to(self.device)
                ## for gt:
                hr_gt.append(clip_average_HR[0].item())

                ## for rppg_estimator:
                all_inputs = {
                    'input_clip': inputs,
                }
                outputs = val_rppg_estimator(all_inputs)
                rPPG = outputs['rPPG']
                psd_pred = cal_psd_hr(rPPG[0], self.frame_rate, return_type='psd')
                hr_pred.append(psd_pred.view(-1).max(0)[1].cpu() + 40)

        ## save the results
        self.draw_rppg_ecg(rPPG, ecg, save_path_epoch)
        self.update_best(epoch, hr_pred, hr_gt, val_type='clip')
        
    def train_one_epoch(self, epoch, save_path_epoch):
        """Trains the model for one epoch.

        Performs forward and backward propagation, calculates losses, and updates the model parameters.
        Updates the loss and metrics savers, and saves the ECG images periodically during training.

        Args:
            epoch (int): The current epoch number.
            save_path_epoch (str): The path to save the results for the current epoch.

        Returns:
            dict: The training losses.
        """
        # sourcery skip: merge-dict-assign
        with tqdm(range(len(self.train_dataloader))) as pbar:
            for iter_idx, sample_batched in zip(pbar, self.train_dataloader):
                inputs, ecg, clip_average_HR = sample_batched['video'].to(self.device), \
                    sample_batched['ecg'].to(self.device), sample_batched['clip_avg_hr'].to(self.device)

                self.optimizer.zero_grad()
                # forward + backward + optimize
                ## for backbone (PhysNet or others):
                all_inputs = {
                    'input_clip': inputs,
                    'epoch': epoch
                }
                outputs = self.rppg_estimator(all_inputs)            # estimate rPPG signal
                rPPG = outputs['rPPG']
                ## calculate loss
                train_losses = {}

                # train_losses['np_loss'] = self.loss_funcs['np_loss'](rPPG, ecg)   # calculate the loss of rPPG signal

                fre_loss, kl_loss, train_mae = self.loss_funcs['ce_loss'](rPPG, clip_average_HR)  # calculate the loss of KL divergence
                train_losses['ce_loss'] = fre_loss + kl_loss

                # total_loss = train_losses['np_loss'] * self.loss_funcs_weight['np_loss'] + \
                total_loss = train_losses['ce_loss'] * self.loss_funcs_weight['ce_loss']
                
                total_loss.backward()
                self.optimizer.step()
                ## update loss saver and metrics saver
                train_metrics = {
                    'mae': train_mae,
                }
                update_avg_meters(self.loss_meters, train_losses, self.actual_batch_size)
                update_avg_meters(self.metrics_meters, train_metrics, self.actual_batch_size)

                mini_batch_info = f'epoch : {epoch:0>3}, mini-batch : {iter_idx:0>4}, lr = {self.optimizer.param_groups[0]["lr"]:.5f}'
                loss_info = ', '.join([f'{key} = {self.loss_meters[key].avg:.4f}' for key in self.loss_meters.keys()])
                metrics_info = ', '.join([f'{key} = {self.metrics_meters[key].avg:.4f}' for key in self.metrics_meters.keys()])

                if iter_idx % self.args.echo_batches == self.args.echo_batches - 1:  # info every mini-batches
                    self.logger.info(', '.join([mini_batch_info, loss_info, metrics_info]))
                    # save the ecg images
                    self.draw_rppg_ecg(rPPG, ecg, save_path_epoch, train=True, mini_batch=iter_idx)

                pbar.set_description(', '.join([mini_batch_info, loss_info, metrics_info]))
        self.scheduler.step()
        return train_losses
             
    def train(self, start_epoch=0, continue_log=''):
        """Trains the model.

        Prepares the training process by setting up necessary directories and loading checkpoints.
        Performs training for each epoch, saving the model and evaluating it periodically.

        Args:
            start_epoch (int, optional): The starting epoch number. Defaults to 0.
            continue_log (str, optional): The name of the log to continue from. Defaults to ''.

        Returns:
            None
        """
        self.prepare_train(start_epoch=start_epoch, continue_log=continue_log)
        self.logger.info(f'prepare train, load ckpt and block gradient, start_epoch: {start_epoch}, gpu: {self.gpu_list}.\n'\
            f'dataset: {self.args.dataset}, num_rppg: {self.args.num_rppg}, model: {self.args.model}, loss: {self.loss_funcs_weight}.\n'\
            f'batch_size: {self.actual_batch_size}, lr: {self.args.lr}, optim: {self.args.optim}, scheduler: {self.args.scheduler}.')

        for epoch in range(start_epoch, self.args.epochs):
            save_path_epoch = f'{self.save_rppg_path}/{epoch:0>3}'
            if not os.path.exists(save_path_epoch):
                os.makedirs(save_path_epoch)
            self.logger.info(f'train epoch: {epoch} lr: {self.optimizer.param_groups[0]["lr"]:.5f}')
            self.train_one_epoch(epoch, save_path_epoch)
            
            # save the model
            torch.save(self.rppg_estimator.state_dict(), os.path.join(self.save_ckpt_path, f'rppg_estimator_{epoch}.pth'))

            # delete the model
            if epoch > 0 and self.args.save_mode == 'best':
                os.remove(os.path.join(self.save_ckpt_path, f'rppg_estimator_{epoch-1}.pth'))

            # evaluate the model
            if epoch %self.args.eval_step == self.args.eval_step - 1:
                self.evaluate_video(epoch) if self.args.eval_type == 'video' else self.evaluate_clip(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## ! general params.
    parser.add_argument('--num_rppg', type=int, default=160, help='the number of rPPG')
    parser.add_argument('--dataset', type=str, default='VIPL', help='dataset = [VIPL, UBFC, PURE, COHFACE]')
    parser.add_argument('--dataset_dir', type=str, default='/data/chushuyang/VIPL', help='dataset dir')
    parser.add_argument('--vipl_fold', type=int, default=1, help='the fold of VIPL dataset')
    parser.add_argument('--save_path', type=str, default='/data2/chushuyang/PAMNet', help='the path to save the model [ckpt, code, visulization]')
    parser.add_argument('--save_mode', type=str, default='all', help='save mode [all, best]')

    ## ! train params.
    parser.add_argument('--gpu', type=str, default="0", help='gpu id list')
    parser.add_argument('--img_size', type=int, default=128, help='the length of clip')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size per gpu')
    parser.add_argument('--eval_step', type=int, default=1, help='the number of **epochs** to eval')
    parser.add_argument('--epochs', type=int, default=200, help='the number of epochs to train')
    parser.add_argument('--echo_batches', type=int, default=500, help='the number of **mini-batches** to print the loss')
    ### loss
    parser.add_argument('--loss', type=str, default='["np_loss", "ce_loss"]', help='loss = [np_loss, ce_loss]')
    parser.add_argument('--loss_weight', type=str, default='[0.1, 1]', help='loss_weight = [1, 1]')
    ### eval_option
    parser.add_argument('--eval_type', type=str, default='video', help='eval_type = [clip, video]')
    
    ## ! model params.
    parser.add_argument('--model', type=str, default='PAMNet', help='model = [Efficientphys, Physnet, Physformer, PAMNet]')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    ### optim
    parser.add_argument('--optim', type=str, default='adam', help='optimizer = [adam, sgd]')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    ### scheduler
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler = [step]')
    parser.add_argument('--step_size', type=int, default=50, help='learning rate decay step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay')

    args = parser.parse_args()

    set_seed(92)

    rppg_estimator_trainer = RppgEstimatorTrainer(args)

    rppg_estimator_trainer.train(start_epoch=0, continue_log='0920_2328') # NOTE: WHETHER TO CONTINUE TRAINING
    # rppg_estimator_trainer.prepare_train(start_epoch=1, continue_log='0920_2328')
    # rppg_estimator_trainer.evaluate_video(epoch=0)