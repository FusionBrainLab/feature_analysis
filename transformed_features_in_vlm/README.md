In this work, we demonstrated that it is possible to train a model capable of reconstruct images from their corresponding feature representations. Using this reconstructor, we explored several interesting properties of the feature space. For example, we showed that by manipulating the features of an original image, it is possible to obtain features representing the same image but with permuted color channels.

This raises an important question: Does the reconstructed image truly reflect reality? In other words, can we conclude that the color shifts in the reconstructed image result from modifications in the feature space rather than being artifacts of the reconstruction process?

To verify this, we performed a cross-check. We took a Vision-Language Model (VLM) and tested how it responds when fed transformed features. If an orthogonal rotation in the feature space indeed produces features corresponding to the same image but with swapped channels, then when answering the question "What color is this object?", the VLM should name the post-swap colors rather than the original ones.

To provide quantitative results, we designed a test. The VLM was presented with an image consisting of a solid-color background and a solid-color shape, both of which could be red, green, or blue. The model was then asked to identify the colors of the background and the shape. An answer was considered correct only if both colors were named accurately.

We conducted two experiments:

Baseline test: The VLM was given the original images without any feature modifications. In this setting, it achieved 100% accuracy.  The result can be found in the file `experiments/default_understanding.ipynb`.

Feature transformation test: The VLM was fed image features after an orthogonal transformation designed to swap the red and blue channels while leaving the green channel unchanged. Our hypothesis was confirmed—in 85% of cases, the VLM indeed perceived red where blue was originally present and blue where red had been, while green remained unaffected. The result can be found in the file `experiments/rgb-to-bgr_understanding.ipynb`

Thus, the cross-validation confirmed our assumption.


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