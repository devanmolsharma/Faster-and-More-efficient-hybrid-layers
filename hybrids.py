import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.nn.parallel
import matplotlib.pyplot as plt

class HybridDense(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize parameters to be used in the forward method.
        self.powers = nn.Parameter(torch.ones(args[1]))
        self.muls = nn.Parameter(torch.ones(args[1]))
    
    def forward(self, inputs):
        # Pass input through the dense layer.
        lin = F.linear(inputs, self.weight,self.bias)
        # Apply ReLU activation function and add small epsilon value.
        x = nn.ReLU()(lin) + 1e-8
        # Multiply with the learned scaling parameters and raise to learned power.
        x = torch.mul(self.muls,torch.pow(x, self.powers))
        # Apply the sign of the linear output.
        x = torch.copysign(x,lin)
        return x
    
    def to(self,device):
        # Move the HybridDense object to the specified device.
        super().to(device)
        self.powers = self.powers.to(device)
        self.muls = self.muls.to(device)


class DynamicConv(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,hidden_size=16,stride = 1,padding = 0):
    super(DynamicConv,self).__init__()
    #defining a similar conv2d layer to learn the most common features (not necessary but it helps)
    self.similar_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
    
    # defining base parameters of this class
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.predictors = nn.ModuleList()# list of models to predict patches
     self.stride = stride
    self.padding = padding
    self.device = 'cpu'
    
    # creaste n models for n out channels
    for x in range(out_channels):
      self.predictors.append(nn.Sequential(HybridDense(kernel_size*kernel_size*(in_channels),hidden_size),
                                           nn.BatchNorm1d(hidden_size),
                                    HybridDense(hidden_size,hidden_size),
                                           nn.BatchNorm1d(hidden_size),
                                    nn.Linear(hidden_size,kernel_size*kernel_size*in_channels)))
   

  def extract_image_patches(self,image, patch_size, stride=None, padding=0):
        # it extracts patches for manual convolution
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if stride is None:
        stride = patch_size
    elif isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
        
    if isinstance(padding, str):
        padding = (1,0,1,0)

    image = torch.nn.functional.pad(image, padding)

    _, _, h, w = image.shape
    num_patches_h = (h - patch_size[0]) // stride[0] + 1
    num_patches_w = (w - patch_size[1]) // stride[1] + 1

    patches = image.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1])
    patches = patches.permute(0, 1, 2, 3, 5, 4).contiguous()

    return patches.view(image.shape[0], image.shape[1], num_patches_h, num_patches_w, patch_size[0], patch_size[1])


  def forward(self,inputs):
    in_size = inputs.size
    patches = self.extract_image_patches(inputs,self.kernel_size,stride=self.stride,padding=self.padding)# get patches
    #predict kernels for each patch
    kernels_list = []
    for i in range(self.out_channels):
        kernels_list.append(
            (self.predictors[i](
                patches.view(-1,self.kernel_size*self.kernel_size*self.in_channels)
            ).view(*patches.shape)).unsqueeze(-3)
        )
    kernels = torch.cat(kernels_list,axis = -3)
    
    # apply convolution and reshape
    out = torch.mul(patches.unsqueeze(-3).repeat(1,1,1,1,self.out_channels,1,1),kernels).mean(axis=(-1,-2)).mean(axis=1)
    return out.permute(0,3,1,2)+self.similar_conv(inputs) # add the normal convolution outputs to avoid learning very frequent kernels

  def to(self,device):
      super().to(device)
      self.predictors = self.predictors.to(device)
#       self.device = device

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = DynamicConv(3,4,8,stride=8,hidden_size=512)
    def forward(self,inputs):
        x = self.c1(inputs)
        return x

class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = DynamicConv(4,8,2,stride=1,padding='same',hidden_size=512)
        self.ps1 = nn.PixelShuffle(2)
        self.norm1 = nn.InstanceNorm2d(3)
        self.c2 = DynamicConv(3,13,2,stride=1,padding='same',hidden_size=64)
        self.ps2 = nn.PixelShuffle(2)
        self.norm2 = nn.InstanceNorm2d(4)
        self.c3 = DynamicConv(4,12,2,stride=1,padding='same',hidden_size=32)
        self.ps3 = nn.PixelShuffle(2)
        self.norm3 = nn.InstanceNorm2d(4)
        self.c4 = DynamicConv(4,3,2,padding='same',hidden_size=64)
        self.norm = nn.InstanceNorm2d(11)
    def forward(self,inputs):
        x = torch.cat((self.c1(inputs),inputs),axis = 1)
        x = torch.nan_to_num(x)
        x = self.ps1(x)
        x = self.norm1(x)
#         print(x.shape)
        x = torch.cat((self.c2(x),x),axis = 1)
        x = torch.nan_to_num(x)
        x = self.ps2(x)
        x = self.norm2(x)
        x = torch.cat((self.c3(x),x),axis = 1)
        x = torch.nan_to_num(x)
        x = self.ps3(x)
        x = self.norm3(x)
        x = self.c4(x)
        return torch.relu(x).clip(0,1)

    
class Autoencoder(nn.Module):
    def __init__(self,encoders:list,decoders:list):
        super().__init__()
        self.encs=encoders
        self.decs=decoders
        
    def forward(self,inputs):
        x =self.encoded= self.encode(inputs)
        x = self.decode(x)
        return x
    
    def encode(self, inputs):
        x = None
        for enc in self.encs:
            if x == None:
                x = enc(inputs)
            else:
                x+= enc(inputs)
        return x
                
    def decode(self,inputs):
        x = None
        for dec in self.decs:
            if x == None:
                x = dec(inputs)
            else:
                x+= dec(inputs)
        return x
    
    def show(self,inputs):
        with torch.no_grad():
            out = self.forward(inputs)
            plt.imshow(out[0].permute(1,2,0).cpu().clip(0,1))
            plt.show()
            plt.imshow(inputs[0].permute(1,2,0).cpu())
            plt.show()
    def parameters(self):
        params = []
        for enc in self.encs:
            params+=enc.parameters()
        for dec in self.decs:
            params+=dec.parameters()
            
        return params
