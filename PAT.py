"""#############################################################################
# Authors:      Amir Refaee, Corey Kelly
#
# Description:  This script reconstructs 3D Photoacoustic images from
#               Radiofrequency data using GPU Programming and convex
#               optimization.
#
# Notes:        If you end up using this code for your project please cite:
#               https://doi.org/10.1364/BOE.431997
#               and
#               https://doi.org/10.1117/1.JBO.25.11.116010
#
#############################################################################"""

import numpy as np
import tifffile
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
from pathlib import Path
import time
from skimage import restoration

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.autoinit import context

from ABUSio import readBIN, ABUScoords, abus_plane_3d_multiple
import ABUSillum
from view3D import msv

gpu_functions = SourceModule("""

__global__ void sa_calc(float* Omega, float* Y_pos, float* Y_orn, int iter,
                        float x0, float y0, float z0, float freq, float c, int n_samples, float dpMin,
                        int angleCheck, float sigmaE, float sigmaL, float res)
    {

    int index = blockIdx.z + gridDim.z * (blockIdx.y + gridDim.y * blockIdx.x);
    int el_idx = iter*blockDim.x+threadIdx.x;

    float voxel_x_pos_min = fmaf(blockIdx.x,res, x0);
    float voxel_y_pos_min = fmaf(blockIdx.y,res, y0);
    float voxel_z_pos_min = fmaf(blockIdx.z,res, z0);

    float voxel_x_pos_max = fmaf(blockIdx.x+1,res, x0);
    float voxel_y_pos_max = fmaf(blockIdx.y+1,res, y0);
    float voxel_z_pos_max = fmaf(blockIdx.z+1,res, z0);

    float xs[] = {voxel_x_pos_min - Y_pos[el_idx*3 + 0], 0, Y_pos[el_idx*3 + 0]-voxel_x_pos_max};
    float ys[] = {voxel_y_pos_min - Y_pos[el_idx*3 + 1], 0, Y_pos[el_idx*3 + 1]-voxel_y_pos_max};
    float zs[] = {voxel_z_pos_min - Y_pos[el_idx*3 + 2], 0, Y_pos[el_idx*3 + 2]-voxel_z_pos_max};

    float rx_min = fmaxf(fmaxf(xs[0],xs[1]),xs[2]);
    float rx_max = fminf(fminf(xs[0],xs[1]),xs[2]);

    float ry_min = fmaxf(fmaxf(ys[0],ys[1]),ys[2]);
    float ry_max = fminf(fminf(ys[0],ys[1]),ys[2]);

    float rz_min = fmaxf(fmaxf(zs[0],zs[1]),zs[2]);
    float rz_max = fminf(fminf(zs[0],zs[1]),zs[2]);

    float r_min = norm3df (rx_min, ry_min, rz_min);
    float r_max = r_min + res;

    float x_A = Y_orn[el_idx*9 + 0];
    float y_A = Y_orn[el_idx*9 + 1];
    float z_A = Y_orn[el_idx*9 + 2];

    float x_E = Y_orn[el_idx*9 + 3];
    float y_E = Y_orn[el_idx*9 + 4];
    float z_E = Y_orn[el_idx*9 + 5];

    float x_L = Y_orn[el_idx*9 + 6];
    float y_L = Y_orn[el_idx*9 + 7];
    float z_L = Y_orn[el_idx*9 + 8];

    int num_samples_between = abs(floor((freq/c)*r_min)-ceil((freq/c)*r_max));

    float r_avg = (r_min+r_max)/2;

    float dx = (voxel_x_pos_min+0.5*res - Y_pos[el_idx*3 + 0]);
    float dy = (voxel_y_pos_min+0.5*res - Y_pos[el_idx*3 + 1]);
    float dz = (voxel_z_pos_min+0.5*res - Y_pos[el_idx*3 + 2]);

    float dotPE = (x_E*dx + y_E*dy + z_E*dz)/r_avg;
    float dotPA = (x_A*dx + y_A*dy + z_A*dz)/r_avg;
    float dotPL = (x_L*dx + y_L*dy + z_L*dz)/r_avg;

    if (dotPA >  dpMin){

        if(abs(dotPA)>1.0)
            dotPA = dotPA/abs(dotPA);

        if(abs(dotPE)>1.0)
            dotPE = dotPE/abs(dotPE);

        if(abs(dotPL)>1.0)
            dotPL = dotPL/abs(dotPL);

        float angE = asin(dotPE);
        float angL = asin(dotPL);

        angE = asin(dotPE);
        angL = asin(dotPL);

        float omegai = dotPA/(r_avg*r_avg);
        float weight=1;

        // use experimentally-measured transducer directivity
        if(angleCheck){
            float weightE = exp(-0.5*(pow(angE / sigmaE, 2)));
            float weightL = exp(-0.5*(pow(angL / sigmaL, 2)));

            weight = weightE * weightL;
        }
        if((int)ceil((freq/c)*r_min) < n_samples)
            atomicAdd(&Omega[index], weight*omegai);
    }
}


__global__ void AT(float* Img, float* RF, float* Omega, float* Y_pos, float* Y_orn, int iter,
                        float x0, float y0, float z0, float freq, float c, int n_samples, float dpMin,
                        int angleCheck, float sigmaE, float sigmaL, float res)
    {

    int index = blockIdx.z + gridDim.z * (blockIdx.y + gridDim.y * blockIdx.x);
    int el_idx = iter*blockDim.x+threadIdx.x;

    float voxel_x_pos_min = fmaf(blockIdx.x,res, x0);
    float voxel_y_pos_min = fmaf(blockIdx.y,res, y0);
    float voxel_z_pos_min = fmaf(blockIdx.z,res, z0);

    float voxel_x_pos_max = fmaf(blockIdx.x+1,res, x0);
    float voxel_y_pos_max = fmaf(blockIdx.y+1,res, y0);
    float voxel_z_pos_max = fmaf(blockIdx.z+1,res, z0);

    float xs[] = {voxel_x_pos_min - Y_pos[el_idx*3 + 0], 0, Y_pos[el_idx*3 + 0]-voxel_x_pos_max};
    float ys[] = {voxel_y_pos_min - Y_pos[el_idx*3 + 1], 0, Y_pos[el_idx*3 + 1]-voxel_y_pos_max};
    float zs[] = {voxel_z_pos_min - Y_pos[el_idx*3 + 2], 0, Y_pos[el_idx*3 + 2]-voxel_z_pos_max};

    float rx_min = fmaxf(fmaxf(xs[0],xs[1]),xs[2]);
    float rx_max = fminf(fminf(xs[0],xs[1]),xs[2]);

    float ry_min = fmaxf(fmaxf(ys[0],ys[1]),ys[2]);
    float ry_max = fminf(fminf(ys[0],ys[1]),ys[2]);

    float rz_min = fmaxf(fmaxf(zs[0],zs[1]),zs[2]);
    float rz_max = fminf(fminf(zs[0],zs[1]),zs[2]);

    float r_min = norm3df (rx_min, ry_min, rz_min);
    float r_max = r_min + res;

    float x_A = Y_orn[el_idx*9 + 0];
    float y_A = Y_orn[el_idx*9 + 1];
    float z_A = Y_orn[el_idx*9 + 2];

    float x_E = Y_orn[el_idx*9 + 3];
    float y_E = Y_orn[el_idx*9 + 4];
    float z_E = Y_orn[el_idx*9 + 5];

    float x_L = Y_orn[el_idx*9 + 6];
    float y_L = Y_orn[el_idx*9 + 7];
    float z_L = Y_orn[el_idx*9 + 8];

    int num_samples_between = abs(floor((freq/c)*r_min)-ceil((freq/c)*r_max));

    float r_avg = (r_min+r_max)/2;

    float dx = (voxel_x_pos_min+0.5*res - Y_pos[el_idx*3 + 0]);
    float dy = (voxel_y_pos_min+0.5*res - Y_pos[el_idx*3 + 1]);
    float dz = (voxel_z_pos_min+0.5*res - Y_pos[el_idx*3 + 2]);

    float dotPE = (x_E*dx + y_E*dy + z_E*dz)/r_avg;
    float dotPA = (x_A*dx + y_A*dy + z_A*dz)/r_avg;
    float dotPL = (x_L*dx + y_L*dy + z_L*dz)/r_avg;

    if (dotPA >  dpMin){

        if(abs(dotPA)>1.0)
            dotPA = dotPA/abs(dotPA);

        if(abs(dotPE)>1.0)
            dotPE = dotPE/abs(dotPE);

        if(abs(dotPL)>1.0)
            dotPL = dotPL/abs(dotPL);

        float angE = asin(dotPE);
        float angL = asin(dotPL);

        angE = asin(dotPE);
        angL = asin(dotPL);

        float omegai = dotPA/(r_avg*r_avg);
        float weight=1;

        // use experimentally-measured transducer directivity
        if(angleCheck){
            float weightE = exp(-0.5*(pow(angE / sigmaE, 2)));
            float weightL = exp(-0.5*(pow(angL / sigmaL, 2)));

            weight = weightE * weightL;
        }

        int ind = (int)ceil((freq/c)*r_min);

        float val = weight*omegai/Omega[index];
        float RF_sum = 0;

        for (int i=0; i<num_samples_between; i++){
            int curr_idx = (el_idx)*n_samples + ind+i;

            if(ind + i < n_samples)
                RF_sum += RF[curr_idx];

        }
        atomicAdd(&Img[index], val*RF_sum);
    }
}

__global__ void A(float* Img, float* RF, float* Omega, float* Y_pos, float* Y_orn, int iter,
                        float x0, float y0, float z0, float freq, float c, int n_samples, float dpMin,
                        int angleCheck, float sigmaE, float sigmaL, float res)
    {

    int index = blockIdx.z + gridDim.z * (blockIdx.y + gridDim.y * blockIdx.x);
    int el_idx = iter*blockDim.x+threadIdx.x;

    float voxel_x_pos_min = fmaf(blockIdx.x,res, x0);
    float voxel_y_pos_min = fmaf(blockIdx.y,res, y0);
    float voxel_z_pos_min = fmaf(blockIdx.z,res, z0);

    float voxel_x_pos_max = fmaf(blockIdx.x+1,res, x0);
    float voxel_y_pos_max = fmaf(blockIdx.y+1,res, y0);
    float voxel_z_pos_max = fmaf(blockIdx.z+1,res, z0);

    float xs[] = {voxel_x_pos_min - Y_pos[el_idx*3 + 0], 0, Y_pos[el_idx*3 + 0]-voxel_x_pos_max};
    float ys[] = {voxel_y_pos_min - Y_pos[el_idx*3 + 1], 0, Y_pos[el_idx*3 + 1]-voxel_y_pos_max};
    float zs[] = {voxel_z_pos_min - Y_pos[el_idx*3 + 2], 0, Y_pos[el_idx*3 + 2]-voxel_z_pos_max};

    float rx_min = fmaxf(fmaxf(xs[0],xs[1]),xs[2]);
    float rx_max = fminf(fminf(xs[0],xs[1]),xs[2]);

    float ry_min = fmaxf(fmaxf(ys[0],ys[1]),ys[2]);
    float ry_max = fminf(fminf(ys[0],ys[1]),ys[2]);

    float rz_min = fmaxf(fmaxf(zs[0],zs[1]),zs[2]);
    float rz_max = fminf(fminf(zs[0],zs[1]),zs[2]);

    float r_min = norm3df (rx_min, ry_min, rz_min);
    float r_max = r_min + res;

    float x_A = Y_orn[el_idx*9 + 0];
    float y_A = Y_orn[el_idx*9 + 1];
    float z_A = Y_orn[el_idx*9 + 2];

    float x_E = Y_orn[el_idx*9 + 3];
    float y_E = Y_orn[el_idx*9 + 4];
    float z_E = Y_orn[el_idx*9 + 5];

    float x_L = Y_orn[el_idx*9 + 6];
    float y_L = Y_orn[el_idx*9 + 7];
    float z_L = Y_orn[el_idx*9 + 8];

    int num_samples_between = abs(floor((freq/c)*r_min)-ceil((freq/c)*r_max));

    float r_avg = (r_min+r_max)/2;

    float dx = (voxel_x_pos_min+0.5*res - Y_pos[el_idx*3 + 0]);
    float dy = (voxel_y_pos_min+0.5*res - Y_pos[el_idx*3 + 1]);
    float dz = (voxel_z_pos_min+0.5*res - Y_pos[el_idx*3 + 2]);

    float dotPE = (x_E*dx + y_E*dy + z_E*dz)/r_avg;
    float dotPA = (x_A*dx + y_A*dy + z_A*dz)/r_avg;
    float dotPL = (x_L*dx + y_L*dy + z_L*dz)/r_avg;

    if (dotPA >  dpMin){

        if(abs(dotPA)>1.0)
            dotPA = dotPA/abs(dotPA);

        if(abs(dotPE)>1.0)
            dotPE = dotPE/abs(dotPE);

        if(abs(dotPL)>1.0)
            dotPL = dotPL/abs(dotPL);

        float angE = asin(dotPE);
        float angL = asin(dotPL);

        angE = asin(dotPE);
        angL = asin(dotPL);

        float omegai = dotPA/(r_avg*r_avg);
        float weight=1;

        // use experimentally-measured transducer directivity
        if(angleCheck){
            float weightE = exp(-0.5*(pow(angE / sigmaE, 2)));
            float weightL = exp(-0.5*(pow(angL / sigmaL, 2)));

            weight = weightE * weightL;
        }

        int ind = (int)ceil((freq/c)*r_min);

        float val = weight*omegai*Img[index]/Omega[index];

        for (int i=0; i<num_samples_between; i++){
            int curr_idx = (el_idx)*n_samples + ind+i;

            if(ind + i <n_samples)
                atomicAdd(&RF[curr_idx], val);
        }
    }
}

""")
################################################################################
######################## Options ###############################################
################################################################################
# Location of the RF bin file
elements = readBIN('data/test/50-20-proc.bin')
num_avg = 20 # set this number according to how many frames per acquistion angle were acquired

# possible cases:   x    -> raw data
#                   svd  -> svd denoised
#                   avg  -> temporal averaging denoised
#                   yhat -> GAN denoised
#                   y    -> temporal averaged -> svd denoised
case = 'y'
data = np.load('50-20-proc/50-20-proc-%s.npy'%case)

res             = 2.0
thetaMax        = 15
SOS             = 1.48e6
freq            = 40.0e6
sigmaE, sigmaL  = 0.1057, 0.4280
angleCheck      = True  # Account for directivity
region3d        = np.array([[-12,12], [-10, 10], [-10, 0]])

L           = 1         # Initial Guess for Lipschitz constant
n           = 1.5       # backtracking line-search resolution
t           = 1         # FISTA paramter, should be 1
lam         = 0.00005   # Regularizer parameter
prox        = 'tv'      # Rugularizer type, possible cases: l1  tv
n_iterations = 25      # Number of iterations for FISTA

# Directory to save the results
folder = Path('%s-50-20-knot-3D-%d-%.2f-res-%s-lambda-%.4f/' % (case,thetaMax, res, prox, lam))
################################################################################
################################################################################
################################################################################

folder.mkdir(exist_ok=True, parents=True)
thetaMax = thetaMax* (np.pi / 180.0)
dpMin = np.cos(thetaMax)
################################################################################
def soft_thresh(X, alpha):
    return np.maximum(np.abs(X)-alpha,0)*np.sign(X)
################################################################################
def tv_norm(X):
    Y = np.zeros_like(X)
    Y[0:Y.shape[0]-1, :, :] = (X[0:X.shape[0]-1, :, :]-X[1:X.shape[0], :, :])**2
    Y[:, 0:Y.shape[1]-1, :] = Y[:, 0:Y.shape[1]-1, :] + (X[:, 0:X.shape[1]-1, :]-X[:, 1:X.shape[1], :])**2
    Y[:, :, 0:Y.shape[2]-1] = Y[:, :, 0:Y.shape[2]-1] + (X[:, :, 0:X.shape[2]-1]-X[:, :, 1:Y.shape[2]])**2
    return np.sum(np.sqrt(Y))
################################################################################

n_elements = len(elements)
n_samples  = len(elements[0].RF)
n_angles   = n_elements//384

x0 = region3d[0,0]
y0 = region3d[1,0]
z0 = region3d[2,0]

xn = region3d[0,1]
yn = region3d[1,1]
zn = region3d[2,1]

xx = np.arange(x0, xn, res)
yy = np.arange(y0, yn, res)
zz = np.arange(z0, zn, res)

XX, YY, ZZ = np.meshgrid(xx, yy, zz, indexing='ij')
sx = len(xx)
sy = len(yy)
sz = len(zz)

Omega = np.zeros((sx, sy, sz),dtype=np.float32, order='C')

Y_pos = np.zeros((n_elements,3))
Y_orn = np.zeros((n_elements, 9))

n_elements = data.shape[1]*data.shape[2]
n_samples  = data.shape[0]
n_angles   = data.shape[1]
b     = np.zeros((n_elements, n_samples),dtype=np.float32)

# This is to get the correct positions
idx = 0
els = elements[:n_elements].copy()

for i in range(n_angles):
    for j in range(384):
        els[i*384+j] = elements[(i*num_avg+idx)*384+j]

elements=els.copy()

for i in range(n_elements):

    b[i,:data.shape[0]]           = np.abs(data[:,i//384,i%384])
    Y_pos[i,:]   =  elements[i].pos
    Y_orn[i,:3]  =  elements[i].vAxial
    Y_orn[i,3:6] =  elements[i].vElev
    Y_orn[i,6:]  =  elements[i].vLat
b = b - b.mean()
AT = gpu_functions.get_function("AT")
A  = gpu_functions.get_function("A")
sa = gpu_functions.get_function("sa_calc")

b_gpu = gpuarray.to_gpu(b.astype(np.float32))
Omega_gpu = gpuarray.to_gpu(Omega.astype(np.float32))
Y_pos_gpu = gpuarray.to_gpu(Y_pos.astype(np.float32))
Y_orn_gpu  = gpuarray.to_gpu( Y_orn.astype(np.float32))
################################################################################
# Calculating Omega before-hand so we can do the division in the kernel

for iter in tqdm(range(n_angles), desc="Omega"):
    sa(Omega_gpu, Y_pos_gpu, Y_orn_gpu,
        np.int32(iter), np.float32(x0), np.float32(y0), np.float32(z0),
        np.float32(freq), np.float32(SOS), np.int32(n_samples), np.float32(dpMin),
        np.int32(angleCheck), np.float32(sigmaE), np.float32(sigmaL),np.float32(res),
        block=(384,1,1), grid=(Omega.shape[0], Omega.shape[1], Omega.shape[2]))

    context.synchronize()

_,mask = ABUSillum.lightMap3(XX,YY,ZZ,0, maskOnly=True)
Omega = Omega_gpu.get()*mask
Omega[np.where(Omega==0)]=1
np.save((folder/'Omega.npy'), Omega)
######################### FISTA ################################################
errors      = np.empty((n_iterations,3)) # tracking f g F
errors[:]   = np.NaN

x = np.zeros_like(Omega)
x_gpu = gpuarray.to_gpu(x.astype(np.float32))
Omega_gpu = gpuarray.to_gpu(Omega.astype(np.float32))

for iter in tqdm(range(n_angles), desc='x0'):
    AT(x_gpu, b_gpu, Omega_gpu, Y_pos_gpu, Y_orn_gpu,
        np.int32(iter), np.float32(x0), np.float32(y0), np.float32(z0),
        np.float32(freq), np.float32(SOS), np.int32(n_samples), np.float32(dpMin),
        np.int32(angleCheck), np.float32(sigmaE), np.float32(sigmaL),np.float32(res),
        block=(384,1,1), grid=(Omega.shape[0], Omega.shape[1], Omega.shape[2]))

    context.synchronize()

x = x_gpu.get()*mask
yk = x.copy()

np.save((folder/'x0.npy'), x)

F_old = float("inf")

for k in range(n_iterations):

    begin = time.time()

    Ayk     = np.zeros_like(b)
    Ayk_gpu = gpuarray.to_gpu(Ayk.astype(np.float32))
    yk_gpu  = gpuarray.to_gpu(yk.astype(np.float32))
    Omega_gpu = gpuarray.to_gpu(Omega.astype(np.float32))

    for iter in tqdm(range(n_angles), desc='Ayk'):
        A(yk_gpu, Ayk_gpu, Omega_gpu, Y_pos_gpu, Y_orn_gpu,
            np.int32(iter), np.float32(x0), np.float32(y0), np.float32(z0),
            np.float32(freq), np.float32(SOS), np.int32(n_samples), np.float32(dpMin),
            np.int32(angleCheck), np.float32(sigmaE), np.float32(sigmaL),np.float32(res),
            block=(384,1,1), grid=(Omega.shape[0], Omega.shape[1], Omega.shape[2]))

        context.synchronize()

    Ayk = Ayk_gpu.get()

    Ayk_b = Ayk - b

    ATAyk_b   = np.zeros_like(yk)
    ATAyk_b_gpu   = gpuarray.to_gpu(ATAyk_b.astype(np.float32))
    Ayk_b_gpu = gpuarray.to_gpu(Ayk_b.astype(np.float32))
    Omega_gpu = gpuarray.to_gpu(Omega.astype(np.float32))

    for iter in tqdm(range(n_angles), desc='ATAyk_b'):
        AT(ATAyk_b_gpu, Ayk_b_gpu, Omega_gpu, Y_pos_gpu, Y_orn_gpu,
            np.int32(iter), np.float32(x0), np.float32(y0), np.float32(z0),
            np.float32(freq), np.float32(SOS), np.int32(n_samples), np.float32(dpMin),
            np.int32(angleCheck), np.float32(sigmaE), np.float32(sigmaL),np.float32(res),
            block=(384,1,1), grid=(Omega.shape[0], Omega.shape[1], Omega.shape[2]))

        context.synchronize()

    ATAyk_b = ATAyk_b_gpu.get()*mask

    ATAyk_b_gpu.gpudata.free()
    Ayk_gpu.gpudata.free()

    i = 0
    while(True):
        print("\tLipschitz iteration %d "% i )
        L_bar = L*n**i

        if prox=='l1':
            p_L_yk = soft_thresh(yk - 2/L_bar*ATAyk_b, lam/L_bar)
        if prox=='tv':
            p_L_yk = restoration.denoise_tv_chambolle(yk - 2/L_bar*ATAyk_b, weight = lam/L_bar)


        p_L_yk_gpu   = gpuarray.to_gpu(p_L_yk.astype(np.float32))
        Ap_L_yk     = np.zeros_like(b)
        Ap_L_yk_gpu = gpuarray.to_gpu(Ap_L_yk.astype(np.float32))
        Omega_gpu = gpuarray.to_gpu(Omega.astype(np.float32))

        for iter in tqdm(range(n_angles), desc='Ayk'):
            A(p_L_yk_gpu, Ap_L_yk_gpu, Omega_gpu, Y_pos_gpu, Y_orn_gpu,
                np.int32(iter), np.float32(x0), np.float32(y0), np.float32(z0),
                np.float32(freq), np.float32(SOS), np.int32(n_samples), np.float32(dpMin),
                np.int32(angleCheck), np.float32(sigmaE), np.float32(sigmaL),np.float32(res),
                block=(384,1,1), grid=(Omega.shape[0], Omega.shape[1], Omega.shape[2]))

            context.synchronize()

        Ap_L_yk = Ap_L_yk_gpu.get()
        Ap_L_yk_gpu.gpudata.free()

        if prox=='l1':
            g = lam*np.linalg.norm(p_L_yk.flatten(), ord=1)
        if prox=='tv':
            g = lam*tv_norm(p_L_yk)


        f = np.linalg.norm((Ap_L_yk-b).flatten())**2

        F = f + g

        Q = np.linalg.norm((Ayk-b).flatten())**2 \
        + np.dot(p_L_yk.flatten()-yk.flatten(), 2*ATAyk_b.flatten())\
        + L_bar/2 * np.linalg.norm((p_L_yk-yk).flatten())**2

        if prox == 'l1':
            Q += lam*np.linalg.norm(p_L_yk.flatten(), ord = 1)
        if prox == 'tv':
            Q += lam*tv_norm(p_L_yk)

        print("\tF : %f" % F)
        print("\tQ : %f\n" % Q)

        if(F <= Q):
            print("\tL found: %f\n" % L_bar)
            L = L_bar

            x_old = x.copy()

            zk = p_L_yk.copy()

            #MFISTA section
            print("\tNew F: %f"  % F)
            print("\tOld F: %f:" % F_old)

            if(F < F_old):
                print("\tzk was used\n")
                x = zk.copy()
            else:
                F = F_old

            t_old = t
            t = (1+np.sqrt(1+4*t**2))/2

            yk = x + (t_old/t)*(zk-x) + ((t_old - 1)/t)*(x-x_old)

            F_old = F
            break

        i = i+1
    duration = time.time()-begin
    print("Iteration %i took: %f seconds\n" % (k, duration))

    file_name = folder/('%s.npy' % k )
    np.save(file_name, x*mask)

    errors[k,0] = f
    errors[k,1] = g/lam
    errors[k,2] = F

    plt.figure()
    plt.semilogy(errors[:, 0], label ='Least Squares (f)')
    image_file_name = folder / 'errors-f.png'
    plt.legend()
    plt.savefig(str(image_file_name))
    plt.close()

    plt.figure()
    plt.semilogy(errors[:, 1], label ='Regularizer (g)')
    image_file_name = folder / 'errors-g.png'
    plt.legend()
    plt.savefig(str(image_file_name))
    plt.close()

    plt.figure()
    plt.semilogy(errors[:, 2], label ='F = f + lambda*g')
    image_file_name = folder / 'errors-Fn.png'
    plt.legend()
    plt.savefig(str(image_file_name))
    plt.close()
