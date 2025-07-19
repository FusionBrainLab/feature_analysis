import argparse
import sys
import os
from math import ceil

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
from transformers import AutoImageProcessor

from tqdm import tqdm

from model import VQVAE
from scheduler import CycleScheduler
import distributed as dist

##########################################################################################################
##########################################################################################################
##########################################################################################################
def log_metrics(mse_list, cos_list, lr_list, epoch_list, i_list, mse, cos, lr, epoch, i):
    mse_list.append(mse)
    cos_list.append(cos)
    lr_list.append(lr)
    epoch_list.append(epoch)
    i_list.append(i)


def save_metrics(metric_dict, values_save_dir, plots_save_dir, save_name_preffix, block_size=6):
    ncols = 1
    nrows = len(metric_dict)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * block_size * 4, nrows * block_size))

    for i, (metric_name, metric_list) in enumerate(metric_dict.items()):
        name = save_name_preffix + f'_{metric_name}'

        values_dir = os.path.join(values_save_dir, 'values')
        os.makedirs(values_dir, mode=0o777, exist_ok=True)
        np.save(os.path.join(values_dir, name + '.npy'), np.asarray(metric_list))

        plot_dir = os.path.join(plots_save_dir, 'plots')
        os.makedirs(plot_dir, mode=0o777, exist_ok=True)
        axes[i].plot(metric_list, label=metric_name)

        axes[i].legend()
    plt.savefig(os.path.join(plot_dir, name + '.png'))


##########################################################################################################
##########################################################################################################
##########################################################################################################
def draw_in_out_cos(images, 
                    reconstructed_images,
                    in_out_cos_list,
                    examples=5, block_size=6, save_dir=None, save_name=None):
    examples = min(examples, len(images))
    ncols = 4
    nrows = examples
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * block_size, nrows * block_size))
    
    for i in range(examples):
        ###############################################################################
        image = images[i]
        axes[i, 0].imshow(image)

        reconstructed_image = reconstructed_images[i].cpu().permute(1, 2, 0)
        axes[i, 1].imshow((reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min()))

        ###############################################################################
        in_out_cos = in_out_cos_list[i].cpu()
        axes[i, 2].imshow(in_out_cos)

        ###############################################################################
        flatten_in_out_cos = in_out_cos.view(-1)
        axes[i, 3].plot(flatten_in_out_cos, alpha=0.5, label='in_out_cos')
        axes[i, 3].legend()

        if i == 0:
            axes[i, 0].set_title('image')
            axes[i, 1].set_title(f'reconstructed_image')
            axes[i, 2].set_title(f'cosine similarity between\ninput and reconstruction')
            axes[i, 3].set_title(f'cosine similarity between\ninput and reconstruction\n(flatten version)')

        axes[i, 0].set_axis_off()
        axes[i, 1].set_axis_off()
        axes[i, 2].set_axis_off()

    if save_dir and save_name:
        plt.savefig(os.path.join(save_dir, save_name))
        

##########################################################################################################
##########################################################################################################
##########################################################################################################        
def draw_input_norm(inputs, outputs, examples=5, block_size=6, save_dir=None, save_name=None):
    ncols = 1
    examples = min(examples, len(inputs))
    nrows = examples
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * block_size * 4, nrows * block_size))
    
    for i in range(examples):
        input_norm = inputs[i].cpu().norm(dim=0).view(-1)
        output_norm = outputs[i].cpu().norm(dim=0).view(-1)

        axes[i].plot(input_norm, alpha=0.5, label='input')
        axes[i].plot(output_norm, alpha=0.5, label='output')
        axes[i].legend()
        
        if i == 0:
            axes[i].set_title(f'norms of input and reconstructed features')

    if save_dir and save_name:
        plt.savefig(os.path.join(save_dir, save_name))


##########################################################################################################
##########################################################################################################
##########################################################################################################
def val(model, eval_dataset, batch_size, device):
    model.eval()

    images = []
    interpolated_images = []
    sample = []
    for j, (im, inter_im, fe) in enumerate(eval_dataset):
        if j > batch_size: break
        images.append(im)
        interpolated_images.append(inter_im.to(device)[None])
        sample.append(fe.to(device)[None])
    interpolated_images = torch.cat(interpolated_images, dim=0)
    sample = torch.cat(sample, dim=0)
    # sample /= sample.norm(dim=1, keepdim=True)
    # sample *= 35

    with torch.no_grad():
        recovered_image, input, check_dict = model(sample, interpolated_images)
        loss = model.loss_function(interpolated_images, recovered_image)
    
    images = interpolated_images.permute(0, 2, 3, 1).cpu()
    images = (images - images.min()) / (images.max() - images.min())
    return images, interpolated_images, recovered_image, input, check_dict, loss


def train(epoch, loader, model, optimizer, scheduler, device, eval_dataset, args, train_tuple, val_tuple):
    if dist.is_primary():
        loader = tqdm(loader)

    train_mse_list, train_cos_list, train_lr_list, train_epoch_list, train_i_list = train_tuple
    val_mse_list, val_cos_list, val_lr_list, val_epoch_list, val_i_list = val_tuple
    for i, (im, fe) in enumerate(loader):
        model.zero_grad()
        fe = fe.to(device)
        # fe /= fe.norm(dim=1, keepdim=True)
        # fe *= 35
        im = im.to(device)

        recovered_image, _, _ = model(fe, im)
        loss = model.loss_function(im, recovered_image)
        loss['loss'].backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        #########################################################
        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]
            loader.set_description(
                (
                    f"epoch: {epoch + 1}; "
                    f"mse: {loss['mse'].item():.5f}; "
                    f"cos: {loss['cos'].item():.5f}; "
                    f"lr: {lr:.5f}"
                )
            )
                        
            if i % 100 == 0:
                log_metrics(train_mse_list, train_cos_list, train_lr_list, train_epoch_list, train_i_list, 
                            loss['mse'].item(), loss['cos'].item(), lr, epoch, i)

                train_values_save_dir=os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}', 'metrics')
                train_plots_save_dir=os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}', 'metrics')
                os.makedirs(train_values_save_dir, mode=0o777, exist_ok=True)
                os.makedirs(train_plots_save_dir, mode=0o777, exist_ok=True)
                save_metrics(
                    {
                        'mse': train_mse_list,
                        'cos': train_cos_list,
                        'lr': train_lr_list,
                        'epoch': train_epoch_list,
                        'i': train_i_list
                    }, 
                    values_save_dir=train_values_save_dir, 
                    plots_save_dir=train_plots_save_dir, 
                    save_name_preffix=f'train')
                

            if i % 100 == 0:
                model.eval()

                val_images, interpolated_images, val_out, val_input, val_check_dict, val_loss = val(model, eval_dataset, fe.shape[0], device)
                log_metrics(val_mse_list, val_cos_list, val_lr_list, val_epoch_list, val_i_list, 
                            val_loss['mse'].item(), val_loss['cos'].item(), lr, epoch, i)
                
                val_values_save_dir=os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}', 'metrics')
                val_plots_save_dir=os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}', 'metrics')
                os.makedirs(val_values_save_dir, mode=0o777, exist_ok=True)
                os.makedirs(val_plots_save_dir, mode=0o777, exist_ok=True)
                save_metrics(
                    {
                        'mse': val_mse_list, 
                        'cos': val_cos_list, 
                        'lr': val_lr_list, 
                        'epoch': val_epoch_list, 
                        'i': val_i_list
                    }, 
                    values_save_dir=val_values_save_dir,
                    plots_save_dir=val_plots_save_dir,
                    save_name_preffix=f'val')
                
                sample_save_dir = os.path.join(args.sample_path, f'num-hidden-layers-{model.config.num_hidden_layers}')
                os.makedirs(sample_save_dir, mode=0o777, exist_ok=True)
                
                in_out_cos_save_name = f'{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_in-out-cos.png'
                draw_in_out_cos(val_images, val_check_dict['transposed_input'], val_check_dict['in_out_cos'],
                                save_dir=sample_save_dir, save_name=in_out_cos_save_name)
                
                draw_input_norm_name = f'{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_input-reconstruction-norm.png'
                draw_input_norm(interpolated_images, val_out, save_dir=sample_save_dir, save_name=draw_input_norm_name)
            
                model.train()

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12, use_thumbnail=True):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_from_pil(image, input_size=448, max_num=12, use_thumbnail=True):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class FeaturesDataset(Dataset):
    def __init__(self, features_json_path, mode='train', side_size=None, image_processor=None):
        with open(features_json_path, 'r') as map_file:
            self.map = list(json.load(map_file).items())
        self.mode = mode

        self.side_size = side_size
        self.image_processor = image_processor

    def __len__(self):
        return len(self.map)

    def __getitem__(self, idx):
        image_path, feature_path = self.map[idx]
        feature = torch.load(feature_path, map_location='cpu').to(torch.float32)
        
        image = Image.open(image_path).convert('RGB')
        processed_image = self.image_processor(image, max_num=1)[0]
        if processed_image.shape[-1] != self.side_size:
            interpolated_image = F.interpolate(processed_image[None], 
                                               size=(self.side_size, self.side_size), 
                                               mode='bilinear', align_corners=False)[0]
        else:
            interpolated_image = processed_image
        if self.mode == 'train':
            return interpolated_image, feature.permute(2, 0, 1) # tensor[3 x 224 x 224], tensor[768 x 14 x 14]
        else:
            return np.asarray(image), interpolated_image, feature.permute(2, 0, 1) # array(427, 640, 3), tensor[3 x 224 x 224], tensor[768 x 14 x 14]

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
def calculat_num_parameters(model):
    params = 0
    for n, p in model.named_parameters():
        print(f'{n}: {np.prod(p.shape)}')
        params += np.prod(p.shape)
    print(f'NUM PARAMETERS: {params}')


def main(args):
    device = args.device

    args.distributed = dist.get_world_size() > 1
    vision_tower_name = args.vision_tower_name
    cache_dir = args.cache_dir
    side_size = args.side_size

    dataset = FeaturesDataset(args.features_json_path, mode='train', image_processor=load_from_pil, side_size=side_size)
    eval_dataset = FeaturesDataset(args.val_features_json_path, mode='val', image_processor=load_from_pil, side_size=side_size)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(dataset, batch_size=32 // args.n_gpu, sampler=sampler, num_workers=16) #32
    
    model = VQVAE(vision_tower_name, cache_dir).to(device)
    calculat_num_parameters(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )


    train_tuple = [], [], [], [], []
    val_tuple = [], [], [], [], []
    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, eval_dataset, args, train_tuple, val_tuple)

        if i % 1 == 0:
            save_path = os.path.join(args.save_path, f'num-hidden-layers-{model.config.num_hidden_layers}')
            os.makedirs(save_path, mode=0o777, exist_ok=True)
            if dist.is_primary():
                torch.save(model.state_dict(), os.path.join(save_path, f'vqvae-{str(i + 1).zfill(3)}.pt'))


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=4)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14 + 1
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    # training parameters
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4) #3e-4
    parser.add_argument("--sched", type=str, default='cycle')
    
    # data parameters
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--side_size", type=int, default=512)

    # cache dir for image_processor
    parser.add_argument("--vision_tower_name", type=str, 
                        default='OpenGVLab/Mini-InternVL-Chat-2B-V1-5')
    parser.add_argument("--cache_dir", type=str, 
                        default=os.path.join(current_dir, '..', 'weights'))
    
    # pathes for dataset
    parser.add_argument("--features_json_path", type=str, 
                        default=os.path.join(current_dir, '..', 'dataset/OpenGVLab-Mini-InternVL-Chat-2B-V1-5/map.json'))
    parser.add_argument("--val_features_json_path", type=str, 
                        default=os.path.join(current_dir, '..', 'dataset/OpenGVLab-Mini-InternVL-Chat-2B-V1-5/map_val.json'))
    
    # pathes for logging
    parser.add_argument("--save_path", type=str, 
                        default=os.path.join(current_dir, 'checkpoint'))
    parser.add_argument("--sample_path", type=str, 
                        default=os.path.join(current_dir, 'samples'))

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
