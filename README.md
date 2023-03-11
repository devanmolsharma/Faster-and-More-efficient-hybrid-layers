
# Dynamic Convolution and dense layers

A brief description of what this project does and who it's for


Introducing Hybrid convolution and dense layers

Same function but 10x-20x faster and lighter 


## Authors

- [@Keep-up-sharma](https://www.github.com/Keep-up-sharma)


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

Install use the encoder

from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler


```python
  import hybrids
  enc = torch.load(encoder.pt)
  from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
 
# sdvae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# sdvae = sdvae.to('cuda')
# use decoder from this
# sdvae.decode(enc(input))
```

## Checkpoints

- [Checkpoint](https://github.com/Keep-up-sharma/Dynamic-Layers/blob/main/encoder%20(1).pt?raw=true)
