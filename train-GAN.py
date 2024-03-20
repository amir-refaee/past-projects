import numpy as np
import time
from pathlib import Path
from matplotlib import pyplot as plt
import PIL

import tifffile

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import transforms
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter

from ABUSio import readBIN
from utils  import save_RGB_tiff
from tqdm import tqdm
from view3D import msv

import networks
import pix2pix as p2p

################################################################################
##                                                                            ##
##    This script assumes the input data has been normalized to [-1,1]        ##
##                                                                            ##
################################################################################

# Training Options
class options():
    def __init__(self):

        # Most Commonly Changed to Least

        self.ngf = 64
        self.ndf = 64

        self.patch_size  = 128
        self.batch_size  = 5
        self.gpu_ids   = [0]
        self.lr        = 0.0002
        self.beta1     = 0.5
        self.lambda_L1 = 100.0

        self.epoch_count = 1 # the starting epoch count
        self.n_epochs = 800  # number of epochs with the initial learning rate
        self.n_epochs_decay = 200 # number of epochs to linearly decay learning rate to zero

        self.name = 'wire-phantom-training'
        self.checkpoints_dir = './checkpoints/'

        self.raw_data ='path/to/data/x.npy'
        self.ground_truth ='path/to/data/y.npy'

        self.print_freq = 100
        self.save_latest_freq = 5 # epoch saving frequency


        self.input_nc  = 1
        self.output_nc = 1

        self.shuffle_training_data = False
        self.norm        = 'batch'
        self.no_dropout = True
        self.init_type   = 'normal'
        self.init_gain   = 0.02
        self.n_layers_D  = 3 # only used if netD = n_layers
        self.gan_mode    = 'lsgan'       # [vanilla| lsgan | wgangp]
        self.netG        = 'unet_128'      # [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
        self.netD        = 'basic'         # [basic | n_layers | pixel]

        self.direction = 'AtoB'
        self.lr_policy = 'linear'

        self.epoch = 'latest'
        self.continue_train = False
        self.load_iter = 0

        self.isTrain = True
        self.verbose = True

opt = options()

folder = Path(opt.checkpoints_dir+opt.name)
folder.mkdir(parents=True, exist_ok=True)

with open(str(folder)+'/train_options.txt', 'w') as f:
    print(opt.__dict__, file=f)

# Loading input data
X_RF = np.load(opt.raw_data)
Y_RF = np.load(opt.ground_truth)

n_elements = X_RF.shape[1]*X_RF.shape[2]
n_samples  = X_RF.shape[0]
n_angles   = n_elements//384

# Preparing the train pairs
X_RF = X_RF[:n_samples//opt.patch_size*opt.patch_size,:,:]
Y_RF = Y_RF[:n_samples//opt.patch_size*opt.patch_size,:,:]

n_data_samples = (n_elements//opt.patch_size)*(n_samples//opt.patch_size)

X = np.zeros((n_data_samples, opt.patch_size, opt.patch_size))
Y = np.zeros((n_data_samples, opt.patch_size, opt.patch_size))

# diving RF data into patches    ###############################################
for angle in tqdm(range(n_angles)):
    for i in range(384//opt.patch_size):
        for j in range(n_samples//opt.patch_size):

            patch_idx = angle*(384//opt.patch_size)*(n_samples//opt.patch_size) + i*(n_samples//opt.patch_size)+j

            X[patch_idx,:,:] = X_RF[j*opt.patch_size:(j+1)*opt.patch_size, angle, i*opt.patch_size:(i+1)*opt.patch_size]
            Y[patch_idx,:,:] = Y_RF[j*opt.patch_size:(j+1)*opt.patch_size, angle, i*opt.patch_size:(i+1)*opt.patch_size]

#Setting up the network
device = torch.device("cuda:%d"%(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

if opt.shuffle_training_data:
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

model = p2p.Pix2PixModel(opt)
model.setup(opt)

G_GAN  = np.zeros(opt.n_epochs + opt.n_epochs_decay + 1)
G_L1   = np.zeros(opt.n_epochs + opt.n_epochs_decay + 1)
D_real = np.zeros(opt.n_epochs + opt.n_epochs_decay + 1)
D_fake = np.zeros(opt.n_epochs + opt.n_epochs_decay + 1)

# Training the model=============================================
total_iters = 0

for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

    epoch_start_time = time.time()
    epoch_iter       = 0
    errors = np.zeros((4,1))

    for i in range(n_data_samples//opt.batch_size):

        total_iters += opt.batch_size
        epoch_iter  += opt.batch_size


        inpt_x = X[i*opt.batch_size:(i+1)*opt.batch_size,:,:]
        inpt_y = Y[i*opt.batch_size:(i+1)*opt.batch_size,:,:]


        inpt_x = Variable(torch.from_numpy(inpt_x)\
                          .to(device, dtype=torch.float)).view(opt.batch_size,1,opt.patch_size,opt.patch_size)
        inpt_y = Variable(torch.from_numpy(inpt_y)\
                          .to(device, dtype=torch.float)).view(opt.batch_size,1,opt.patch_size,opt.patch_size)

        model.set_input(inpt_x, inpt_y)
        model.optimize_parameters()

        if True: #total_iters % opt.print_freq == 0:
            losses = model.get_current_losses()

            message = '(epoch: %d, step_in_epoch: %d:) ' % (epoch, epoch_iter)

            error_  = np.zeros((4,1))
            i = 0
            for k, v in losses.items():
                error_[i,0] = v
                i += 1
                message += '%s: %.3f ' % (k, v)
                #writer.add_scalar('step_losses/'+k, v, total_iters)

            errors = np.append(errors, error_, axis=1)

    G_GAN[epoch]  = np.sum(errors[0,:])/errors.shape[1]
    G_L1[epoch]   = np.sum(errors[1,:])/errors.shape[1]
    D_real[epoch] = np.sum(errors[2,:])/errors.shape[1]
    D_fake[epoch] = np.sum(errors[3,:])/errors.shape[1]

    if epoch % opt.save_latest_freq == 0:
        print('Epoch %d: G_GAN: %f  G_L1: %f' %(epoch, G_GAN[epoch], G_L1[epoch]))
        model.save_networks('latest')
        model.save_networks(epoch)

    model.update_learning_rate()    # update learning rates at the end of every epoch.

# End of training===============================================================

plt.figure()
plt.plot(G_GAN)
plt.title('G_GAN')
plt.savefig(str(folder)+'/G_GAN.png')

plt.figure()
plt.plot(G_L1)
plt.title('G_L1')
plt.savefig(str(folder)+'/G_L1.png')

plt.figure()
plt.plot(D_real)
plt.title('D_real')
plt.savefig(str(folder)+'/D_real.png')

plt.figure()
plt.plot(D_fake)
plt.title('D_fake')
plt.savefig(str(folder)+'/D_fake.png')
