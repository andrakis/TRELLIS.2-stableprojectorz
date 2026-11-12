A One-click installer for Windows: [(Python 3.11, Cuda 12.8, Torch 2.8)](https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/tag/latest)

Repository for integration with the [StableProjectorz](https://stableprojectorz.com/), a free AI-texturing tool.<br>
Our Discord server: [here](https://discord.gg/aWbnX2qan2)

Optimized the code to fit better into 8GB gpus, even with 1024³ voxels.

Also, fixed the nasty vertical lines [bug](https://www.reddit.com/r/StableDiffusion/comments/1r197yy/trellis_2_3d_model_generation_problems/) thanks to the [visualbruno Bruno Fargnoli](https://github.com/visualbruno)

<h2>If you want to install manually:</h2>

1) use python 3.11
2) follow steps in `code/install.py` around line 130, method `install_dependencies()`
4) you will need to pip install wheel files, find them in `code/whl` folder. Install only whl that are mentioned in 2).
5) Lastly, from inside the `code` folder, do `git submodule deinit -f --all` and then `git submodule update --init --recursive`.
6) launch via `python app.py`


<br>
<h2>Original Trellis Repo description:</h2>

<br>

![](assets/teaser.webp)

# Native and Compact Structured Latents for 3D Generation

<a href="https://arxiv.org/abs/2512.14692"><img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/microsoft/TRELLIS.2-4B"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
<a href="https://huggingface.co/spaces/microsoft/TRELLIS.2"><img src="https://img.shields.io/badge/Hugging%20Face-Demo-blueviolet"></a>
<a href="https://microsoft.github.io/TRELLIS.2"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>

https://github.com/user-attachments/assets/63b43a7e-acc7-4c81-a900-6da450527d8f

*(Compressed version due to GitHub size limits. See the full-quality video on our project page!)*

**TRELLIS.2** is a state-of-the-art large 3D generative model (4B parameters) designed for high-fidelity **image-to-3D** generation. It leverages a novel "field-free" sparse voxel structure termed **O-Voxel** to reconstruct and generate arbitrary 3D assets with complex topologies, sharp features, and full PBR materials.


## ✨ Features

### 1. High Quality, Resolution & Efficiency
Our 4B-parameter model generates high-resolution fully textured assets with exceptional fidelity and efficiency using vanilla DiTs. It utilizes a Sparse 3D VAE with 16× spatial downsampling to encode assets into a compact latent space.

| Resolution | Total Time* | Breakdown (Shape + Mat) |
| :--- | :--- | :--- |
| **512³** | **~3s** | 2s + 1s |
| **1024³** | **~17s** | 10s + 7s |
| **1536³** | **~60s** | 35s + 25s |

<small>*Tested on NVIDIA H100 GPU.</small>

### 2. Arbitrary Topology Handling
The **O-Voxel** representation breaks the limits of iso-surface fields. It robustly handles complex structures without lossy conversion:
*   ✅ **Open Surfaces** (e.g., clothing, leaves)
*   ✅ **Non-manifold Geometry**
*   ✅ **Internal Enclosed Structures**

### 3. Rich Texture Modeling
Beyond basic colors, TRELLIS.2 models arbitrary surface attributes including **Base Color, Roughness, Metallic, and Opacity**, enabling photorealistic rendering and transparency support.

### 4. Minimalist Processing
Data processing is streamlined for instant conversions that are fully **rendering-free** and **optimization-free**.
*   **< 10s** (Single CPU): Textured Mesh → O-Voxel
*   **< 100ms** (CUDA): O-Voxel → Textured Mesh


## 🗺️ Roadmap

- [x] Paper release
- [x] Release image-to-3D inference code
- [x] Release pretrained checkpoints (4B)
- [x] Hugging Face Spaces demo
- [ ] Release shape-conditioned texture generation inference code (Current schdule: before 12/24/2025)
- [ ] Release training code (Current schdule: before 12/31/2025)


## 🛠️ Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**.
- **Hardware**: An NVIDIA GPU with at least 24GB of memory is necessary. The code has been verified on NVIDIA A100 and H100 GPUs.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain packages. Recommended version is 12.4.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 

### Installation Steps
1. Clone the repo:
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ```

2. Install the dependencies:
    
    **Before running the following command there are somethings to note:**
    - By adding `--new-env`, a new conda environment named `trellis2` will be created. If you want to use an existing conda environment, please remove this flag.
    - By default the `trellis2` environment will use pytorch 2.6.0 with CUDA 12.4. If you want to use a different version of CUDA, you can remove the `--new-env` flag and manually install the required dependencies. Refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) for the installation command.
    - If you have multiple CUDA Toolkit versions installed, `CUDA_HOME` should be set to the correct version before running the command. For example, if you have CUDA Toolkit 12.4 and 13.0 installed, you can run `export CUDA_HOME=/usr/local/cuda-12.4` before running the command.
    - By default, the code uses the `flash-attn` backend for attention. For GPUs do not support `flash-attn` (e.g., NVIDIA V100), you can install `xformers` manually and set the `ATTN_BACKEND` environment variable to `xformers` before running the code. See the [Minimal Example](#minimal-example) for more details.
    - The installation may take a while due to the large number of dependencies. Please be patient. If you encounter any issues, you can try to install the dependencies one by one, specifying one flag at a time.
    - If you encounter any issues during the installation, feel free to open an issue or contact us.
    
    Create a new conda environment named `trellis2` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    ```
    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
    ```sh
    Usage: setup.sh [OPTIONS]
    Options:
        -h, --help              Display this help message
        --new-env               Create a new conda environment
        --basic                 Install basic dependencies
        --flash-attn            Install flash-attention
        --cumesh                Install cumesh
        --o-voxel               Install o-voxel
        --flexgemm              Install flexgemm
        --nvdiffrast            Install nvdiffrast
        --nvdiffrec             Install nvdiffrec
    ```

## 📦 Pretrained Weights

The pretrained model **TRELLIS.2-4B** is available on Hugging Face. Please refer to the model card there for more details.

| Model | Parameters | Resolution | Link |
| :--- | :--- | :--- | :--- |
| **TRELLIS.2-4B** | 4 Billion | 512³ - 1536³ | [Hugging Face](https://huggingface.co/microsoft/TRELLIS.2-4B) |


## 🚀 Usage

### 1. Image to 3D Generation

#### Minimal Example

Here is an [example](example.py) of how to use the pretrained models for 3D asset generation.

```python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Can save GPU memory
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. Setup Environment Map
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. Load Image & Run
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216) # nvdiffrast limit

# 4. Render Video
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)

# 5. Export to GLB
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample.glb", extension_webp=True)
```

Upon execution, the script generates the following files:
 - `sample.mp4`: A video visualizing the generated 3D asset with PBR materials and environmental lighting.
 - `sample.glb`: The extracted PBR-ready 3D asset in GLB format.

**Note:** The `.glb` file is exported in `OPAQUE` mode by default. Although the alpha channel is preserved within the texture map, it is not active initially. To enable transparency, import the asset into your 3D software and manually connect the texture's alpha channel to the material's opacity or alpha input.

#### Web Demo

[app.py](app.py) provides a simple web demo for image to 3D asset generation. you can run the demo with the following command:
```sh
python app.py
```

Then, you can access the demo at the address shown in the terminal.

### 2. PBR Texture Generation

Will be released soon. Please stay tuned!

## 🧩 Related Packages

TRELLIS.2 is built upon several specialized high-performance packages developed by our team:

*   **[O-Voxel](o-voxel):** 
    Core library handling the logic for converting between textured meshes and the O-Voxel representation, ensuring instant bidirectional transformation.
*   **[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM):** 
    Efficient sparse convolution implementation based on Triton, enabling rapid processing of sparse voxel structures.
*   **[CuMesh](https://github.com/JeffreyXiang/CuMesh):** 
    CUDA-accelerated mesh utilities used for high-speed post-processing, remeshing, decimation, and UV-unwrapping.


## 🏷️ Commercial Use Without the Non-Commercial Components

Two components on the image → GLB path carry non-commercial licences:

- **nvdiffrast** (NVIDIA Source Code License, research / evaluation use only) is the UV
  rasteriser in the texture bake (`o_voxel.postprocess.to_glb`).
- **RMBG-2.0** (CC BY-NC 4.0) removes the background of an input image without alpha.

Both can now be left out. **The defaults are unchanged**, so nothing differs unless you opt in:

| Setting | Effect |
|---|---|
| `TRELLIS_UV_RASTERIZER=torch` | Bakes textures with `o-voxel/o_voxel/uvraster.py`, a pure-PyTorch (MIT) UV rasteriser; nvdiffrast is never imported. `auto` (default) uses nvdiffrast when it is installed and falls back to `uvraster` when it is not; `nvdiffrast` requires it. |
| `TRELLIS_NO_REMBG=1` | RMBG-2.0 is never loaded. Input images must then carry an alpha channel, or use `chroma_key` below. |
| API form fields `chroma_key`, `chroma_tolerance`, `chroma_shadows` | On `/generate_no_preview` (and the multi-image variant). `chroma_key` is `''` (off, default), `'auto'` (background colour = median of the image border) or `'#rrggbb'`. Background-coloured pixels connected to the image border become transparent, so the pipeline takes its alpha path and never needs background removal. `chroma_tolerance` (default 24) is the RGB distance counted as background. `chroma_shadows` (default true) also keys the subject's shadow on the backdrop, which otherwise survives as a floor slab under the model. |

`uvraster` reproduces nvdiffrast's texel conventions as measured with a single-triangle
probe: texel `(r, c)` is sampled at `((c + 0.5) / N, (r + 0.5) / N)`, and row 0 is `v = 0`.
Compared against nvdiffrast on real bakes, it gives identical texel coverage and at most a
2e-5 difference in surface position, so the exported GLB has the same mesh and UVs.

Not covered here: `trellis2/renderers` (preview rendering in `app.py` / `example.py`) still
uses nvdiffrast / nvdiffrec, and the model weights (TRELLIS.2, DINOv3) have their own
licences on their model cards. Check those for your use case.

## 🧠 Lower System-RAM Use

Low-VRAM mode keeps every sub-model in system RAM and moves each one to the GPU only for its
own stage, so on an 8 GB card, system RAM becomes the limit. Three changes bring it down.
Measured on Windows 11, RTX 3060 Ti (8 GB), 32 GB RAM, 512 pipeline:

| | Before | After |
|---|---|---|
| At rest (working set / private memory) | ~14 / ~24 GB | ~10 / ~14 GB |
| During a generation (private memory) | 25+ GB, growing with each stage | ~18 GB peak, flat across jobs |

- **Lean model loading** (on by default; `TRELLIS_LEAN_LOAD=0` turns it off). The original
  loader builds every weight in fp32, converts it to half precision, then loads a second
  full copy from the checkpoint. PyTorch's Windows allocator (mimalloc) keeps the freed fp32
  memory committed, so each model held about twice its real size. Models are now built on
  the meta device (buffers and RoPE tables are still built for real) and the checkpoint
  tensors are assigned directly, cast to the dtype the model expects. Every parameter,
  buffer and tensor attribute of all six models used by the 512 pipeline was hashed under
  both loaders: they are bit-identical. Anything unexpected falls back to the original loader.
- **Keep CPU weights across GPU round trips** (on by default in low-VRAM mode;
  `TRELLIS_KEEP_CPU_WEIGHTS=0` turns it off). Each stage's `.cpu()` allocated a fresh CPU
  copy of the model while the old memory stayed committed. Weights never change at
  inference time, so each model's CPU tensors are kept, and `.cpu()` points the parameters
  back at them. That removes the reallocation, and the copy back becomes instant.
- **`TRELLIS_MAX_RESOLUTION=512`** (opt-in) skips loading the two 1024 flow models (~5.2 GB)
  on machines that only serve 512. A 1024 request then fails with a clear error.

Two further changes keep the weights out of RAM almost entirely. Same machine, every model
loaded (512, 1024 and 1536 pipelines), same image and seed, 12 steps, 2K texture:

| | Before | After |
|---|---|---|
| At rest (private memory) | 20.3 GB | **1.7 GB** |
| Peak during a generation, 512 / 1024 / 1536 (private) | 27.6 / 33.4 / 34.9 GB | **11.7 / 14.6 / 18.9 GB** |
| System commit peak | 55.1 GB | **39.2 GB** |
| Generation time, 512 / 1024 / 1536 | 50 / 156 / 399 s | 52 / 159 / 412 s |

- **Weights stay in their files** (on by default; `TRELLIS_MMAP_WEIGHTS=0` turns it off).
  On a local disk, every weight is a read-only memory map of its checkpoint instead of a
  copy in RAM. Windows charges read-only file pages nothing against the commit limit (the
  limit other apps hit when they fail "out of memory"): under pressure it drops them and
  reads them back from disk the next time a stage moves that model to the GPU. (safetensors'
  own mapping is charged in full on Windows, so the loader maps the file itself.) Files on a
  network share are still copied. DINOv3 is mapped the same way once transformers has
  loaded it.
- **fp16 twins of the bf16 flow models.** The five flow models are bf16 on disk and the
  pipeline converts them to fp16, which would copy them anyway. Run once, from the repo root
  with the venv active:

  ```
  python make_fp16_twins.py           # ~12 GB of disk; --remove deletes them
  ```

  It writes `<name>.fp16.safetensors` beside each checkpoint: the weights exactly as the
  pipeline held them in RAM before. All 4,266 tensors hash identically under both loaders.
  A twin is ignored if its checkpoint changes. Without twins, the flow models are copied
  into RAM as before.
- **mimalloc returns freed memory.** `main_api.py` and `app.py` set
  `MIMALLOC_PURGE_DELAY=0` (unless it is already set) before torch loads; by default
  PyTorch's Windows allocator keeps freed memory committed.

The weights still show in the process's working set during a generation, but as file-backed
pages the OS can drop at no cost. The views are read-only: an in-place write to a weight
would be an access violation.

## ⚖️ License

This model and code are released under the **[MIT License](LICENSE)**.

Please note that certain dependencies operate under separate license terms:

- [**nvdiffrast**](https://github.com/NVlabs/nvdiffrast): Utilized for rendering generated 3D assets. This package is governed by its own [License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt).

- [**nvdiffrec**](https://github.com/NVlabs/nvdiffrec): Implements the split-sum renderer for PBR materials. This package is governed by its own [License](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt).

## 📚 Citation

If you find this model useful for your research, please cite our work:

```bibtex
@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}
```
