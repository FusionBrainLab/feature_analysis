In our work, we showed that it is possible to train a reconstructor that will, given features, obtain images corresponding to these features. Using this reconstructor, we explored some interesting properties of the feature space. For example, we showed that given features of the original image, it is possible to obtain features corresponding to the same image, but with swapped channels.

In this case, the question may arise whether the reconstructed image really reflects reality, that is, whether we can conclude that the change in colors in the reconstructed image corresponds to a change in the properties of the transformed features, and is not related to the reconstruction process.

For this, we decided to do a cross-check. We took the VLM and checked what would happen if we inserted transformed features into it. If using orthogonal rotation in the feature space it is indeed possible to obtain features corresponding to the same image, but with swapped channels, then when answering the question "what color is some object" vlm will have to name not the color in the initial image, but the color obtained after swapping the channels.

In order to provide a numerical result in the work, we came up with a test. Vlm is given a picture consisting of a single-color background and a single-color figure. Both the background and the figure can be of three colors: red, green, and blue. After which it is asked what color the background is and what color the figure is. The answer is counted if both colors are named correctly.

We conducted two experiments.

1. In the first case, we gave vlm the test described above to solve, without changing the features of the image in any way. In this setting, vlm coped with the task in 100% of cases. The result can be found in the file `experiments/default_understanding.ipynb`.

2. In the second variant, we fed LLM image features after orthogonal transformation, hoping that LLM would now start seeing red where there was blue, blue where there was red, and the green channel would remain untouched. And our hopes were justified! In 85% of cases, VLM really did start seeing red where there used to be blue, and blue where there used to be red. The result can be found in the file `experiments/rgb-to-bgr_understanding.ipynb`

Thus, cross-checking confirmed our assumption.


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

Run notebooks in following order:
1. `experiments/default_understanding.ipynb` - default VLLM understanding of our test
2. `experiments/rgb-to-bgr_understanding.ipynb` - VLLM understanding of transformed features