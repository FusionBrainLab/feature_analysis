import json
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from utils_for_llm import IMAGENET_MEAN, IMAGENET_STD, load_image_from_pil, extract_feature

def rgb_to_bgr(im):
    gbr_im = np.array(im)[:, :, ::-1]
    gbr_im = Image.fromarray(gbr_im).convert('RGB')
    return gbr_im


def load_and_process_images(json_path, processor, number):
    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)

    images_pathes_list = []
    for i, (im_path, _) in enumerate(json_dict.items()):
        if i == number: break
        images_pathes_list.append(im_path)
    images_list = [Image.open(p).convert('RGB') for p in tqdm(images_pathes_list)]
    processed_images_list = [processor(im) for im in tqdm(images_list)]
    
    pixels_list = [load_image_from_pil(im, max_num=1).to(torch.bfloat16) for im in tqdm(images_list)]
    processed_pixels_list = [load_image_from_pil(im, max_num=1).to(torch.bfloat16) for im in tqdm(processed_images_list)]

    return images_list, pixels_list, processed_images_list, processed_pixels_list


@torch.no_grad()
def calc_feature(llm, pixel_values):
    features = extract_feature(llm, pixel_values.to(llm.device)) # [1, 32, 32, 1024]

    patches = llm.config.vision_config.image_size // llm.config.vision_config.patch_size
    hidden_size = llm.config.vision_config.hidden_size
    features = features[0].reshape(patches, patches, hidden_size)
    features = features.permute(2, 0, 1)
    return pixel_values, features

@torch.no_grad()
def calc_features_for_images_and_processed_images(llm, pixels_list, processed_pixels_list, device):
    inputs_list = []
    feature_list = []
    for pixel_values in tqdm(pixels_list):
        inputs, features = calc_feature(llm, pixel_values.to(device))
        inputs_list.append(inputs.cpu())
        feature_list.append(features.cpu())

    processed_inputs_list = []
    processed_feature_list = []
    for pixel_values in tqdm(processed_pixels_list):
        inputs, features = calc_feature(llm, pixel_values.to(device))
        processed_inputs_list.append(inputs.cpu())
        processed_feature_list.append(features.cpu())
    
    return inputs_list, feature_list, processed_inputs_list, processed_feature_list

@torch.no_grad()
def calculate_orthogonal_Q(init_F, trans_F): # [1024, 1024]
    init_F, trans_F = init_F.float(), trans_F.float()
    M = init_F.permute(1, 0) @ trans_F
    U, S, Vh = torch.linalg.svd(M)
    Q = U @ Vh
    return Q.to(torch.bfloat16)

@torch.no_grad()
def calculate_linear_Q(init_F, trans_F): # [1024, 1024]
    init_F, trans_F = init_F.float(), trans_F.float()
    Q = torch.linalg.inv(init_F.permute(1, 0) @ init_F) @ init_F.permute(1, 0) @ trans_F
    return Q.to(torch.bfloat16)

@torch.no_grad()
def apply_Q(Fe, Q, n=1):
    gbr_Fe = torch.zeros_like(Fe)

    Qn = Q
    for _ in range(n - 1):
        Qn = Qn @ Q

    for i in range(Fe.shape[1]):
        for j in range(Fe.shape[2]):
            gbr_Fe[:, i, j] = Fe[:, i, j] @ Qn

    # gbr_Fe /= gbr_Fe.norm(dim=0, keepdim=True)
    return gbr_Fe


@torch.no_grad()
def interpolate(im, Fe, model, device):
    processed_image = torch.tensor(im).to(device).float()
    processed_Fe = Fe.to(device)
    reconstructed_image, processed_Fe, check_dict = model.forward(processed_Fe, processed_image)
    return processed_image, reconstructed_image, processed_Fe, check_dict


def from_1_to_255(image, image_mean, image_std):
    global IMAGENET_MEAN, IMAGENET_STD
    image_mean = np.array(IMAGENET_MEAN)
    image_std = np.array(IMAGENET_STD)

    image_unnormed = np.asarray(image) * image_std[:, None, None] + image_mean[:, None, None]
    image_01 = image_unnormed.clip(0, 1)
    image_255 = image_01 * 255
    image_255_uint = np.array(image_255, dtype=np.uint8)
    return image_255_uint


def interpolate_all_features(model, inputs_list, features_list, Qed_features_list, 
                             processed_inputs_list, processed_features_list, device):
    global IMAGENET_MEAN, IMAGENET_STD
    image_mean = np.array(IMAGENET_MEAN)
    image_std = np.array(IMAGENET_STD)

    inter_list, rec_list, processed_inter_list, processed_rec_list, Qed_rec_list = [], [], [], [], []
    
    z = zip(tqdm(inputs_list), features_list, Qed_features_list, processed_inputs_list, processed_features_list)
    for inputs, features, Qed_features, processed_inputs, processed_features in z:
        inter, rec, _, _ = interpolate(inputs, features[None], model, device)
        _, Qed_rec, _, _ = interpolate(inputs, Qed_features[None], model, device)
        processed_inter, processed_rec, _, _ = interpolate(processed_inputs, processed_features[None], model, device)

        inter = from_1_to_255(inter[0].cpu(), image_mean, image_std)
        rec = from_1_to_255(rec[0].cpu(), image_mean, image_std)
        processed_inter = from_1_to_255(processed_inter[0].cpu(), image_mean, image_std)
        processed_rec = from_1_to_255(processed_rec[0].cpu(), image_mean, image_std)
        Qed_rec = from_1_to_255(Qed_rec[0].cpu(), image_mean, image_std)

        inter_list.append(inter)
        rec_list.append(rec)
        processed_inter_list.append(processed_inter)
        processed_rec_list.append(processed_rec)
        Qed_rec_list.append(Qed_rec)
    
    return inter_list, rec_list, processed_inter_list, processed_rec_list, Qed_rec_list


def Qed(llm, images_list, Q=None, n=1):
    inputs_list, features_list, Qed_features_list, processed_inputs_list, processed_features_list = [], [], [], [], []

    for im in tqdm(images_list):
        inputs, features = calc_feature(llm, load_image_from_pil(im, max_num=1).to(torch.bfloat16))
        processed_inputs, processed_features = calc_feature(llm, load_image_from_pil(rgb_to_bgr(im), max_num=1).to(torch.bfloat16))

        if Q is None:
            Q = torch.diag(torch.ones(features.shape[0], device=features.device, dtype=features.dtype))
            print(f'{Q.shape=}')

        Qed_me_features = apply_Q(features, Q, n)
        
        inputs_list.append(inputs)
        features_list.append(features)
        processed_inputs_list.append(processed_inputs)
        processed_features_list.append(processed_features)
        Qed_features_list.append(Qed_me_features)

    return inputs_list, features_list, Qed_features_list, processed_inputs_list, processed_features_list