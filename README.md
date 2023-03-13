
# Mind Your Own Kernels: Dynamic Convolution for Personalized Feature Extraction

Paper: https://keep-up-sharma.github.io/mind_your_own_kernel/
(not professional but i tried)


Introducing Hybrid convolution and dense layers

Same function but 10x-20x faster and lighter

Convolutional Neural Networks (CNNs) have achieved remarkable success in various computer vision tasks such as image classification, object detection, and segmentation. The Conv2d layer is a fundamental building block of CNNs, which applies a set of fixed filters (kernels) to extract features from the input image. However, storing and computing these kernels can be computationally expensive and memory-intensive, especially for large images or complex architectures. To address this issue, various techniques such as depthwise separable convolutions, dilated convolutions, and group convolutions have been proposed to reduce the number of parameters and computations. However, these methods still require storing a large number of pre-defined kernels.

In this paper, we propose a custom Conv2d class, where kernels are predicted dynamically using a hybrid dense layer for each patch of the input image. This approach saves time and storage as the model doesn't have to remember kernels. The proposed method learns to predict the kernels that best extract features from the input patch, based on the patch's content. Our approach is inspired by recent works on dynamic convolution, which has shown promising results in various tasks such as object detection and segmentation.

## Installation

Install use the layers

```python
  import hybrids
  A = hybrids.DynamicConv(2,4)
```

## Training 

I have added training  code to following notebook:
(might need some fixes)
- [Training Notebook](https://www.kaggle.com/code/keepupsharma/lw-encoder)

    
## Screenshots
Samples of my 10 mb encoder (trained 3-4 hours)
Contains 1 layer (4 kernels (8x8)) stride 8, hidden size=512

![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/light%20(1).png?raw=true)


![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/light.png?raw=true)

Samples of my 479 kb encoder (trained less than 1 hour)
Contains 1 layer (4 kernels (8x8)) stride 8, hidden size=128

![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/final%20(1).png?raw=true)


![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/final.png?raw=true)

Impressed? 

Samples of my 89 kb encoder (trained less than 20 minutes)
Contains 1 layer (4 kernels (8x8)) stride 8, hidden size=8

![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/vlight%20(1).png?raw=true)

![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/vlight%20(2).png?raw=true)

Not convinced? 

Samples of my 1 mb autoencoder (trained less than 50 minutes)

![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/result%20(3).png?raw=true)

![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/result%20(4).png?raw=true)

![App Screenshot](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/result%20(5).png?raw=true)

## Installation

```python
  from hybrids import encoder
  enc = torch.load(encoder.pt)
  
#from diffusers import AutoencoderKL
# sdvae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# sdvae = sdvae.to('cuda')
# use decoder from this
# sdvae.decode(enc(input))
```

## Authors

- [@Keep-up-sharma](https://www.github.com/Keep-up-sharma)

## Checkpoints

- [10 mb Checkpoint](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/encoder%20(1).pt?raw=true)
- [479 kb Checkpoint](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/lightencoder.pt?raw=true)
- [86 kb Checkpoint](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/lightencoder%20(3).pt?raw=true)
- [Autoencoder](https://github.com/Keep-up-sharma/Faster-and-More-efficient-hybrid-layers/raw/main/autoencoder%20(1).pt)
