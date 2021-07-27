import argparse
import time
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from DNL_3DCNN import DNL_3DCNN
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
# import  os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--save", default='./log', type=str, help="Save path")
#parser.add_argument("--resume", default="/media/smartcity/E6AA1145AA1113A1/CaiFeifan/D3Dnet/code/log/model4_epoch2.pth.tar", type=str, help="Resume path (default: none)")
parser.add_argument("--resume", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--train_dataset_dir", default='./data/Vimeo', type=str, help="train_dataset")
parser.add_argument("--inType", type=str, default='y', help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=24, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=35, help="Number of epochs to train for")
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument("--gpu", default=7, type=int, help="gpu ids (default: 0)")
parser.add_argument("--lr", type=float, default=4e-4, help="Learning Rate. Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument("--step", type=int, default=6, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")

global opt, model
opt = parser.parse_args()
#gpus = [0,1,2,3,4,5,6,7]
torch.cuda.set_device(opt.gpu)

#DDP
# # 1) 初始化
# dist.init_process_group(backend="nccl")
# print("Use GPU: {} for training".format(opt.local_rank))

# # 2） 配置每个进程的gpu
# local_rank = dist.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)



def train(train_loader, scale_factor, epoch_num):

    #torch.cuda.set_device(opt.local_rank)  # 当前卡
    #one GPU
    net = DNL_3DCNN(scale_factor).cuda()
    #DP
    #net = torch.nn.DataParallel(DNL_3DCNN(scale_factor).cuda(), device_ids=gpus)
    #DDP
    #net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[opt.local_rank], output_device=opt.local_rank)# 数据并行

    #print the construction of DNL_3DCNN
    #print(net)

    epoch_state = 0
    loss_list = []
    psnr_list = []
    loss_epoch = []
    psnr_epoch = []

    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        loss_list = ckpt['loss']
        psnr_list = ckpt['psnr']

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    criterion_MSE = torch.nn.MSELoss().cuda()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)
    print("Begin Train")
    for idx_epoch in range(epoch_state, epoch_num):
        print(idx_epoch)
        for idx_iter, (LR, HR) in tqdm(enumerate(train_loader),desc="Training each epoch: ", total = len(train_loader),position = 0, leave = True):
            print(idx_iter)
            LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
            print(idx_iter)
            SR = net(LR)
            print(idx_iter)

            loss = criterion_MSE(SR, HR[:, :, 3, :, :])
            loss_epoch.append(loss.detach().cpu())
            psnr_epoch.append(psnr(SR, HR[:, :, 3, :, :]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print("Test")
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, loss_epoch---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path=opt.save, filename='model' + str(scale_factor) + '_epoch' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []
            psnr_epoch = []
            valid(net)

def valid(net):
    valid_set = ValidSetLoader(opt.train_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    valid_loader = DataLoader(dataset=valid_set, num_workers=opt.threads, batch_size=8, shuffle=True)
    psnr_list = []
    for idx_iter, (LR, HR) in enumerate(valid_loader):
        LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
        SR = net(LR)
        psnr_list.append(psnr(SR.detach(), HR[:, :, 3, :, :].detach()))
    print('valid PSNR---%f' % (float(np.array(psnr_list).mean())))

def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path,filename))

def main():
    # train_sampler = DistributedSampler(opt.train_dataset_dir)
    # train_set = TrainSetLoader(opt.train_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, shuffle=False,
    #                                            num_workers=opt.workers, pin_memory=True, sampler=train_sampler)

    print("Step 1 TrainSetLoader bigin")
    train_set = TrainSetLoader(opt.train_dataset_dir, scale_factor=opt.scale_factor, inType=opt.inType)
    print("Step 2 TrainLoader bigin")
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    print("Step 3 Train bigin")
    train(train_loader, opt.scale_factor, opt.nEpochs)

if __name__ == '__main__':
    main()

