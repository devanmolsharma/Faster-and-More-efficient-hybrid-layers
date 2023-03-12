
# Dynamic Convolution and dense layers


Introducing Hybrid convolution and dense layers

Same function but 10x-20x faster and lighter

Convolutional Neural Networks (CNNs) have achieved remarkable success in various computer vision tasks such as image classification, object detection, and segmentation. The Conv2d layer is a fundamental building block of CNNs, which applies a set of fixed filters (kernels) to extract features from the input image. However, storing and computing these kernels can be computationally expensive and memory-intensive, especially for large images or complex architectures. To address this issue, various techniques such as depthwise separable convolutions, dilated convolutions, and group convolutions have been proposed to reduce the number of parameters and computations. However, these methods still require storing a large number of pre-defined kernels.

In this paper, we propose a custom Conv2d class, where kernels are predicted dynamically using a hybrid dense layer for each patch of the input image. This approach saves time and storage as the model doesn't have to remember kernels. The proposed method learns to predict the kernels that best extract features from the input patch, based on the patch's content. Our approach is inspired by recent works on dynamic convolution, which has shown promising results in various tasks such as object detection and segmentation.

## Related Works

Dynamic Convolution: Dynamic Convolution is a novel design that addresses the performance degradation issue of light-weight CNNs due to their low computational budgets. Unlike traditional CNNs that use a single convolution kernel per layer, Dynamic Convolution aggregates multiple parallel convolution kernels dynamically based on their attentions, which are input dependent.While Dynamic Convolution offers several advantages over traditional CNNs, it also has some drawbacks. One potential issue is the increased complexity in implementing and training such a model due to the dynamic aggregation of multiple kernels. Moreover, the attention mechanism may not always be accurate and may introduce noise in the convolution process, which could lead to reduced accuracy. Furthermore, the increased model complexity may also result in higher memory and computational requirements, which may limit its application in resource-constrained environments.

MobileNetV3: MobileNetV3 is a lightweight CNN architecture that uses a combination of depthwise separable convolutions, squeeze-and-excitation blocks, and a hybrid convolution operation that combines a depthwise convolution and a pointwise convolution. The hybrid convolution operation reduces the number of parameters and computations compared to standard convolutions.


##Methodology

Our proposed custom Conv2d class is designed to predict kernels dynamically for each patch of the input image. Specifically, given an input patch of size HxWxC, we use N hybrid dense layers to predict a set of kernels of size KxKxC (and then concatenate them), where K is the kernel size, C is the number of input channels, and N is the number of output channels. The hybrid dense layer is composed of a depthwise convolution layer with a kernel size of 1x1 and a pointwise convolution layer with a kernel size of 3x3. This combination allows us to reduce the number of parameters and computations compared to a standard dense layer.

The predicted kernels are then applied to the input patch using the standard convolution operation to obtain the output feature map of size H'xW'xN. The predicted kernels are not stored explicitly, but are computed on-the-fly for each patch. The weights of the hybrid dense layer are learned during training using backpropagation and gradient descent. During inference, the hybrid dense layer is used to predict the kernels for each patch, and the convolution operation is applied to obtain the output feature map.



## Installation

Install use the layers

```python
  import hybrids
  A = hybrids.DynamicConv(2,4)
```
    
## Screenshots
Samples of my 10 mb encoder (trained 3-4 hours)
Contains 1 layer (4 kernels (8x8)) stride 8, hidden size=512

![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/light%20(1).png?raw=true)


![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/light.png?raw=true)

## Installation

The code defines two custom PyTorch modules, HybridDense and DynamicConv.

HybridDense is a linear layer that implements a nonlinear function of the input tensor using a power function and a scaling factor. The powers and muls parameters are learned during training.

DynamicConv is a custom convolutional layer that dynamically generates convolutional kernels for each patch of the input image using a neural network. The predictors attribute is a list of neural networks (one for each output channel of the convolution), which take as input a flattened patch of the input image and output a kernel for that patch. The extract_image_patches method is used to extract patches from the input image, which are then passed through the neural network to generate the convolutional kernels. The resulting kernels are used to convolve the patches and generate the output feature map.

The encoder and decoder classes are defined using DynamicConv layers to implement an image encoder and decoder, respectively.

```python
  from hybrids import encoder
  enc = torch.load(encoder.pt)
  from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
 
# sdvae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# sdvae = sdvae.to('cuda')
# use decoder from this
# sdvae.decode(enc(input))
```

## Authors

- [@Keep-up-sharma](https://www.github.com/Keep-up-sharma)

## Checkpoints

- [Checkpoint](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/encoder%20(1).pt?raw=true)
