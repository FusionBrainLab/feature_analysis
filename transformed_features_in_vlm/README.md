
1. Generate Features Dataset

```
# train
python dataset_generation/dataset--OpenGVLab--InternViT-300M-448px_from_llm.py --mode train --images_dir /path/to/coco/train2017 --max_images 200000
# val
python dataset_generation/dataset--OpenGVLab--InternViT-300M-448px_from_llm.py --mode val --images_dir /path/to/coco/val2017 --max_images 100
```


2. Train Reconstruction Model

```
python reconstructor/train_feature_to_image.py --n_gpu 4 --batch_size 32
```

3. Explore Jupyter Notebooks

* `experiments/default_understanding.ipynb` - default VLLM understanding of reconstructed images
* `experiments/rgb-to-bgr_understanding.ipynb` - VLLM understanding of reconstructed images with RGB to BGR transformation