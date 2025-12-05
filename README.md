## ___***Light-X: Generative 4D Video Rendering with Camera and Illumination Control***___

<div align="center">
    <a href='http://tqtqliu.github.io/' target='_blank'>Tianqi Liu</a><sup>1,2,3</sup>&emsp;
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a><sup>1</sup>&emsp;       
    <a href='https://inso-13.github.io/' target='_blank'>Zihao Huang</a><sup>1,2,3</sup>&emsp;   
    <a href='https://daniellli.github.io/' target='_blank'>Shaocong Xu</a><sup>2</sup>&emsp;
    <a href='https://sainingzhang.github.io/' target='_blank'>Saining Zhang</a><sup>2,4</sup><br>
    <a href='https://hugoycj.github.io/' target='_blank'>Chongjie Ye</a><sup>5</sup>&emsp;
    <a href='https://arlo0o.github.io/libohan.github.io/' target='_blank'>Bohan Li</a><sup>6,7</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=396o2BAAAAAJ&hl=en' target='_blank'>Zhiguo Cao</a><sup>3</sup>&emsp;
    <a href='https://weivision.github.io/' target='_blank'>Wei Li</a><sup>1</sup>&emsp;
    <a href='https://sites.google.com/view/fromandto' target='_blank'>Hao Zhao</a><sup>4,2,*</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>1,*</sup>
</div>
<div>
<div align="center">
    <sup>1</sup>S-Lab, NTU&emsp;
    <sup>2</sup>BAAI&emsp;
    <sup>3</sup>HUST&emsp;
    <sup>4</sup>AIR,THU&emsp;
    <sup>5</sup>FNii, CUHKSZ&emsp;
    <sup>6</sup>SJTU&emsp;
    <sup>7</sup>EIT (Ningbo)
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2512.05115" target='_blank'>
    <img src="http://img.shields.io/badge/arXiv-2512.05115-b31b1b?logo=arxiv&logoColor=b31b1b" alt="ArXiv">
  </a>
  <a href="https://lightx-ai.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-Page-red?logo=googlechrome&logoColor=red">
  </a>
  <a href="https://youtu.be/ui9Lg2H--0c">
    <img src="https://img.shields.io/badge/YouTube-Video-blue?logo=youtube&logoColor=blue">
  </a>
  <a href='https://huggingface.co/datasets/tqliu/Light-Syn/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
  <a href='https://huggingface.co/tqliu/Light-X/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-red'></a>
  <a href="#">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=TQTQliu.Light-X" alt="Visitors">
  </a>
</p>



>**TL;DR**: <em>Light-X is a video generation framework that jointly controls camera trajectory and illumination from monocular videos.</em>

https://github.com/user-attachments/assets/2211979f-4910-4a88-bf19-19711fa07182

## 🌟 Abstract
Recent advances in illumination control extend image-based methods to video, yet still facing a trade-off between lighting fidelity and temporal consistency. Moving beyond relighting, a key step toward generative modeling of real-world scenes is the joint control of camera trajectory and illumination, since visual dynamics are inherently shaped by both geometry and lighting. To this end, we present **Light-X**, a video generation framework that enables controllable rendering from monocular videos with both viewpoint and illumination control. **1)** We propose a disentangled design that decouples geometry and lighting signals: geometry and motion are captured via dynamic point clouds projected along user-defined camera trajectories, while illumination cues are provided by a relit frame consistently projected into the same geometry. These explicit, fine-grained cues enable effective disentanglement and guide high-quality illumination. **2)** To address the lack of paired multi-view and multi-illumination videos, we introduce **Light-Syn**, a degradation-based pipeline with inverse-mapping that synthesizes training pairs from in-the-wild monocular footage. This strategy yields a dataset covering static, dynamic, and AI-generated scenes, ensuring robust training. Extensive experiments show that Light-X outperforms baseline methods in joint camera-illumination control and surpasses prior video relighting methods under both text- and background-conditioned settings.

## 🛠️ Installation
#### Clone Light-X
  ```
  git clone https://github.com/TQTQliu/Light-X.git
  cd Light-X
  ```
#### Setup environments
  ```
 conda create -n lightx python=3.10
 conda activate lightx
 pip install -r requirements.txt
  ```

## 🚀 Inference
Run inference using the following script:
```bash
bash run.sh
```
All required models will be downloaded automatically.

**Note:** We provide a list of commonly used commands along with their corresponding visual outputs in **[EXAMPLE.md](EXAMPLE.md)**. We encourage users to refer to this file to understand the purpose and effect of each argument.

The `run.sh` script executes `inference.py` with the following arguments:

```bash
python inference.py \
    --video_path [INPUT_VIDEO_PATH] \
    --stride [VIDEO_STRIDE] \
    --out_dir [OUTPUT_DIR] \
    --camera ['traj' | 'target'] \
    --mode ['gradual' | 'bullet' | 'direct' | 'dolly-zoom'] \
    --mask \
    --target_pose [THETA PHI RADIUS X Y] \
    --traj_txt [TRAJECTORY_TXT] \
    --relit_txt [RELIGHTING_TXT] \
    [--relit_vd] \
    --relit_cond_type ['ic' | 'ref' | 'hdr' | 'bg'] \
    [--relit_cond_img CONDITION_IMAGE] \
    [--recam_vd]
```
#### Key Arguments:
🎥 Camera

- `--camera`: Camera control (`traj` for trajectory, `target` for fixed view)
- `--mode`: Motion style (`gradual`, `bullet`, `direct`, `dolly-zoom`)
- `--traj_txt`: Trajectory file (required for `traj`)
- `--target_pose`: Target view `<theta phi r x y>` (required for `target`)
- `--recam_vd`: Enable re-camera
  
See [here](https://github.com/TrajectoryCrafter/TrajectoryCrafter/blob/main/docs/config_help.md) for more details on camera parameters.

💡 Relighting

- `--relit_txt`: Relighting parameter file
- `--relit_vd`: Enable relighting
- `--relit_cond_type`: Lighting condition (`ic`, `ref`, `hdr`, `bg`)
- `--relit_cond_img`: Conditioning image (required for `ref` / `hdr` modes)


## 🔥 Training

#### 1. Prepare Training Data

Download the [dataset](https://huggingface.co/datasets/tqliu/Light-Syn/tree/main) .

#### 2. Generate Metadata

Generate the metadata JSON file describing the training samples.

```bash
python tools/gen_json.py -r <DATA_PATH>
```

Then Update the `DATASET_META_NAME` in your config to the path of the newly generated JSON file.

#### 3. Start Training

Begin the training process. Checkpoints will be saved in the `output_train/` directory.

```bash
bash train.sh
```

#### 4. Convert Zero Checkpoint to fp32

Convert the DeepSpeed ZeRO sharded checkpoint to a single fp32 file for inference.

* **Example (for step 16000):**

```bash
python tools/zero_to_fp32.py output_train/checkpoint-16000 output_train/checkpoint-16000-out --safe_serialization
```

> `output_train/checkpoint-16000-out` is the resulting fp32 checkpoint directory.

#### 5. Run Inference

Pass the converted fp32 checkpoint directory to your inference script.

```bash
python inference.py --transformer_path output_train/checkpoint-16000-out
```


## 📚 Citation
If you find our work useful for your research, please consider citing our paper:

```
@article{lightx,
  title   = {Light-X: Generative 4D Video Rendering with Camera and Illumination Control},
  author  = {Liu, Tianqi and Chen, Zhaoxi and Huang, Zihao and Xu, Shaocong and Zhang, Saining and Ye, Chongjie and Li, Bohan and Cao, Zhiguo and Li, Wei and Zhao, Hao and Liu, Ziwei},
  journal = {arXiv preprint arXiv:2512.05115},
  year    = {2025}
}
```

## ♥️ Acknowledgement
This work is built on many amazing open-source projects shared by [TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter), [IC-Light](https://github.com/lllyasviel/IC-Light), and [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun). Thanks all the authors for their excellent contributions!

## 📧 Contact
If you have any questions, please feel free to contact Tianqi Liu <b>(tq_liu at hust.edu.cn)</b>.
