import argparse,os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataset_utils import DIV2K, Set5_val
import utils
import skimage.color as sc
import random
from model.model import PCRN
from adan import Adan

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="PCRN")
parser.add_argument("--batch_size", type=int, default=64,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-3,
                    help="Learning Rate. Default=2e-3")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="checkpoint_x2", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=32,
                    help="number of threads for dataset_utils loading")
parser.add_argument("--root", type=str, default="/root/autodl-tmp/npy",
                    help='dataset_utils directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=96,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained PCRN")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)
#python train_PConvSR.py > train.log 2>&1
cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = DIV2K.div2k(args)
print(len(trainset))
testset = Set5_val.DatasetFromFolderVal("Test_Datasets/Set5/",
                                        "Test_Datasets/Set5_LR/x{}/".format(args.scale),
                                        args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building PCRN")
args.is_train = True

model = PCRN(upscale_factor=args.scale)

l1_criterion = nn.L1Loss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)


print("===> Setting Optimizer")

optimizer = Adan(model.parameters(), lr=args.lr)



def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)

    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1

        loss_sr.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print(
                "===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                loss_l1.item()
                                                                ))


def valid():
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader),
                                                          avg_ssim / len(testing_data_loader)))


def save_checkpoint(epoch):
    model_folder = "checkpoint_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path, _use_new_zipfile_serialization=False)
    print("===> Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(model)

for epoch in range(args.start_epoch, args.nEpochs + 1):
    valid()
    train(epoch)
    save_checkpoint(epoch)
