## FDLPIPS: Combining FDL and LPIPS for efficiency

https://github.com/eezkni/FDL  
https://github.com/richzhang/PerceptualSimilarity/  

FDL is a great perceptual loss function, it's extremely sharp and avoids the artifacts that are common with VGG LPIPS, but it comes with its own new artifacts. Using FDL and LPIPS together, they reduce each others' artifacts. However, loading them separately is wasteful. They both use VGG features, so why not combine them? Technically this is not a 1:1 match for LPIPS, because we're using VGG19 features here instead of VGG16, and the layer weighting is also not the same. So don't rely on this to calculate LPIPS as a metric, but it still works just as well as a loss function.

The parameters for FDL are changed from the original to more sane default values for convenience. Also, the random 2d projections used for FDL can easily be extended to 3d, which greatly improves temporal consistency on video models compared to the 2d version.

![image-comparison](./imgs/comparison.png)

### Installation:

```
git clone https://github.com/spacepxl/FDLPIPS
cd FDLPIPS
pip install -e .
```

### Usage (2D):
```python
from FDLPIPS import FDLPIPS_2D
fdlpips = FDLPIPS_2D().to(device)
# x, y: (B,C,H,W)
loss_fdl, loss_lpips = fdlpips(x, y)
```

### Usage (3D):
```python
from FDLPIPS import FDLPIPS_3D
fdlpips = FDLPIPS_3D().to(device)
# x, y: (B,C,F,H,W)
loss_fdl, loss_lpips = fdlpips(x, y)
```

### Options:

```python
fdlpips = FDLPIPS_2D(
  patch_size = [5, 5], # patch size for the random projections
  stride = 1, # stride for the random projections
  num_proj = 64, # number of random projections per layer
  phase_weight = 0.01, # weight for phase loss calculation, 0=disabled (faster)
  fdl_weights = [1.0, 1.0, 1.0, 1.0, 1.0], # per-layer weights for FDL
  lpips_weights = [1.0, 1.0, 1.0, 1.0, 1.0], # per-layer weights for LPIPS
  image_range = [0.0, 1.0], # expected data range of input images, so they can be normalized correctly
)

fdlpips(
  x, y, # predicted and ground truth images to compare
  reduce_batch=True, # average over the batch or return per-image loss
  resample=False # sample new random projections for FDL
)
```
