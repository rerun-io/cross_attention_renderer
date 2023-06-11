"""
checkpoint (*WITH SOFTRAS SPLIT*) under
/om2/user/sitzmann/logs/light_fields/NMR_hyper_1e2_reg_layernorm/64_256_None/checkpoints/model_epoch_0087_iter_250000.pth
"""

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../' )
import random

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset.realestate10k_dataio import RealEstate10k, get_camera_pose
import torch
import models
import training
import configargparse
from torch.utils.data import DataLoader
import loss_functions
import summaries
from utils import util
import config
from tqdm import tqdm
from imageio import imwrite, get_writer
from glob import glob
import lpips
from skimage.metrics import structural_similarity
import cv2

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--logging_root', type=str, default=config.logging_root)
p.add_argument('--data_root', type=str, default='/om2/user/egger/MultiClassSRN/data/NMR_Dataset', required=False)
p.add_argument('--val_root', type=str, default=None, required=False)
p.add_argument('--network', type=str, default='relu')
p.add_argument('--category', type=str, default='donut')
p.add_argument('--conditioning', type=str, default='hyper')
p.add_argument('--experiment_name', type=str, required=True)
p.add_argument('--num_context', type=int, default=0)
p.add_argument('--max_num_instances', type=int, default=None)
p.add_argument('--num_trgt', type=int, default=1)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--views', type=int, default=2)

# General training options
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--reconstruct', action='store_true', default=False)
p.add_argument('--nearest', action='store_true', default=False)
p.add_argument('--model', type=str, default='midas_vit')
p.add_argument('--autodecoder', action='store_true', default=False)
p.add_argument('--epochs_til_ckpt', type=int, default=10)
p.add_argument('--steps_til_summary', type=int, default=500)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None)
opt = p.parse_args()


def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

def make_circle(n, radius=0.2):
    angles = np.linspace(0, 4 * np.pi, n)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius

    coord = np.stack([x, y, np.zeros(n)], axis=-1)
    return coord

def render_data(model_input, scene, model):
    model_input = util.dict_to_gpu(model_input)

    nrender = model_input['query']['cam2world'].size(1)
    query_cam2world = model_input['query']['cam2world']

    context_cam2world = model_input['context']['cam2world']

    query_intrinsic = model_input['query']['intrinsics']
    uv = model_input['query']['uv']
    nrays = uv.size(-2)

    chunks = nrays // 4096
    z = model.get_z(model_input)

    query_cam2world = model_input['query']['cam2world']
    query_intrinsic = model_input['query']['intrinsics']
    trans = torch.Tensor(make_circle(query_cam2world.shape[1])).cuda()
    query_cam2world[0, :, :3, -1] = query_cam2world[0, :, :3, -1]

    scene = str(scene)
    scene_path = scene.split("/")[-1]

    if not os.path.exists('vis/'):
        os.makedirs('vis/')

    writer = get_writer("vis/{}.mp4".format(scene_path))
    loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
    mses = []
    ssims = []
    psnrs = []
    lpips_list = []

    with torch.no_grad():
        for i in tqdm(range(nrender)):
            model_input['query']['cam2world'] = query_cam2world[:, i:i+1]
            model_input['query']['intrinsics'] = query_intrinsic[:, i:i+1]

            uv_i = uv[:, i:i+1]

            uv_chunks = torch.chunk(uv_i, chunks, dim=2)

            model_outputs = []

            for uv_chunk in uv_chunks:
                model_input['query']['uv'] = uv_chunk
                model_output = model(model_input, z=z)
                del model_output['z']
                del model_output['coords']
                del model_output['uv']
                del model_output['pixel_val']
                del model_output['at_wts']

                model_outputs.append(model_output)

            model_output_full = {}

            for k in model_outputs[0].keys():
                outputs = [model_output[k] for model_output in model_outputs]
                print(k, [output.size() for output in outputs])
                val = torch.cat(outputs, dim=-2)
                model_output_full[k] = val

            rgb = model_output_full['rgb'].view(256, 256, 3)
            rgb_gt = model_input['query']['rgb'][0, i]

            rgb = rgb_np = rgb.detach().cpu().numpy()
            rgb_gt = target_np = rgb_gt.detach().cpu().numpy()

            rgb = np.clip(rgb, -1, 1)
            rgb_gt = np.clip(rgb_gt, -1, 1)

            rgb = (rgb + 1) / 2.
            target = (rgb_gt + 1) / 2

            rgb = torch.Tensor(rgb).cuda()
            target = torch.Tensor(target).cuda()

            rgb_lpips = ((rgb.permute(2, 0, 1) - 0.5) * 2)[None, :, :, :].cuda()
            target_lpips = ((target.permute(2, 0, 1) - 0.5) * 2)[None, :, :, :].cuda()

            mse = img2mse(rgb, target)
            psnr = mse2psnr(mse)

            mses.append(mse.item())
            psnrs.append(psnr.item())

            lpip = loss_fn_alex(rgb_lpips, target_lpips).item()
            lpips_list.append(lpip)

            ssim = structural_similarity(rgb_np, target_np, win_size=11, multichannel=True, gaussian_weights=True, channel_axis=2, data_range=2.0)
            ssims.append(ssim)

            print("mse, psnr, lpip, ssim", np.mean(mses), np.mean(psnrs), np.mean(lpips_list), np.mean(ssims))

            rgb_np = np.clip(rgb_np, -1, 1)
            rgb_np = (((rgb_np + 1) / 2) * 255).astype(np.uint8)


            writer.append_data(rgb_np)


def render(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)

    test_dataset = RealEstate10k(img_root="data_download/realestate/test",
                                 pose_root="data_download/realestate/test_poses",
                                 num_ctxt_views=1, num_query_views=1, query_sparsity=256,
                                 )

    train_scenes = [str(s).split("/")[-1] for s in test_dataset.all_scenes]

    model = models.CrossAttentionRenderer(model=opt.model, n_view=opt.views)
    old_state_dict = model.state_dict()

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['model'], strict=not opt.reconstruct)

    model = model.cuda()
    scenes = [s.stem for s in test_dataset.all_scenes]
    
    for idx in range(200):
        all_scene = test_dataset.all_scenes[idx]

        try:
            data = get_camera_pose(all_scene, "data_download/realestate/test_poses", test_dataset.uv, views=opt.views)
        except:
            continue

        render_data(data, all_scene, model)

if __name__ == "__main__":
    opt = p.parse_args()
    render(0, opt)
