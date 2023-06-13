"""
checkpoint (*WITH SOFTRAS SPLIT*) under
/om2/user/sitzmann/logs/light_fields/NMR_hyper_1e2_reg_layernorm/64_256_None/checkpoints/model_epoch_0087_iter_250000.pth
"""

import os

# Enable import from parent package
import sys

import numpy.typing as npt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../")
import random

import configargparse
import lpips
import numpy as np
import rerun as rr
import torch
import torch.distributed as dist
from imageio import get_writer
from skimage.metrics import structural_similarity
from tqdm import tqdm

import models
from dataset.realestate10k_dataio import RealEstate10k, get_camera_pose
from utils import util

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]).to(x.device))

p = configargparse.ArgumentParser()
p.add("-c", "--config_filepath", required=False, is_config_file=True)

p.add_argument("--rerun_vis", action="store_true", default=False)
p.add_argument("--gpus", type=int, default=1)
p.add_argument("--views", type=int, default=2)
p.add_argument("--reconstruct", action="store_true", default=False)
p.add_argument("--model", type=str, default="midas_vit")
p.add_argument("--checkpoint_path", default=None)
opt = p.parse_args()


def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)


def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


def make_circle(n, radius=0.2):
    angles = np.linspace(0, 4 * np.pi, n)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius

    coord = np.stack([x, y, np.zeros(n)], axis=-1)
    return coord


def log_current_outputs(model_outputs, world_from_camera, intrinsics) -> None:
    rgb_outputs = [model_output["rgb"][0,0] for model_output in model_outputs]
    rgb = torch.cat(rgb_outputs, dim=-2)
    rgb = (rgb.clip(-1, 1) + 1) / 2.0  # transform to [0, 1] range
    full_rgb = torch.zeros(256*256, 3, device=rgb.device)
    full_rgb[:len(rgb)] = rgb
    full_rgb = full_rgb.view(256, 256, 3).numpy(force=True)

    log_image(
        "world/prediction",
        full_rgb,
        world_from_camera,
        intrinsics,
    )


def log_image(
    camera_entity: str,
    rgb: npt.ArrayLike,
    world_from_camera: npt.ArrayLike,
    intrinsics: npt.ArrayLike,
    timeless=False,
) -> None:
    rgb_entity = f"{camera_entity}/rgb"
    width, height = rgb.shape[:2]
    rr.log_transform3d(
        camera_entity,
        rr.TranslationAndMat3(world_from_camera[:3, 3], world_from_camera[:3, :3]),
        timeless=timeless,
    )
    rr.log_view_coordinates(
        camera_entity,
        xyz="RDF",
        timeless=timeless,
    )
    rr.log_pinhole(
        rgb_entity,
        child_from_parent=intrinsics[:3, :3],
        width=width,
        height=height,
        timeless=timeless,
    )
    rr.log_image(rgb_entity, rgb, timeless=timeless)


def render_data(model_input, scene, model, rerun_vis):
    model_input = util.dict_to_gpu(model_input)

    nrender = model_input["query"]["cam2world"].size(1)

    uv = model_input["query"]["uv"]
    nrays = uv.size(-2)

    chunks = nrays // 4096
    z = model.get_z(model_input)

    context_intrinsics = model_input["context"]["intrinsics"]
    context_cam2worlds = model_input["context"]["cam2world"]
    context_rgbs = model_input["context"]["rgb"]

    query_cam2world = model_input["query"]["cam2world"]
    query_intrinsic = model_input["query"]["intrinsics"]
    query_cam2world[0, :, :3, -1] = query_cam2world[0, :, :3, -1]

    scene = str(scene)
    scene_path = scene.split("/")[-1]

    if not os.path.exists("vis/"):
        os.makedirs("vis/")

    if rerun_vis:
        rr.init("wide_baseline")
        rr.save(f"vis/{scene_path}.rrd")
        rr.log_view_coordinates("world", up="-Y", timeless=True)

        for i, (rgb, wfc, intrinsic) in enumerate(
            zip(context_rgbs[0], context_cam2worlds[0], context_intrinsics[0])
        ):
            rgb = rgb.numpy(force=True)
            rgb = (np.clip(rgb, -1, 1) + 1) / 2  # transform to [0, 1] range
            wfc = wfc.numpy(force=True)
            intrinsic = intrinsic.numpy(force=True)

            # log input views
            log_image(
                f"world/input_images/camera_#{i}", rgb, wfc, intrinsic, timeless=True
            )
            # TODO add visualization of depth (depth_ray key in model_outputs)

    writer = get_writer(f"vis/{scene_path}.mp4")
    loss_fn_alex = lpips.LPIPS(net="vgg").cuda()
    mses = []
    ssims = []
    psnrs = []
    lpips_list = []

    with torch.no_grad():
        for i in tqdm(range(nrender)):
            rr.set_time_sequence("frame_id", i)

            model_input["query"]["cam2world"] = query_cam2world[:, i : i + 1]
            model_input["query"]["intrinsics"] = query_intrinsic[:, i : i + 1]
            cam2world_np = model_input["query"]["cam2world"][0, 0].numpy(force=True)
            intrinsics_np = model_input["query"]["intrinsics"][0, 0].numpy(force=True)

            uv_i = uv[:, i : i + 1]

            uv_chunks = torch.chunk(uv_i, chunks, dim=2)

            model_outputs = []

            for uv_chunk in uv_chunks:
                model_input["query"]["uv"] = uv_chunk
                model_output = model(
                    model_input,
                    z=z,
                    vis_ray=(205, 105) if rerun_vis else None,
                )
                del model_output["z"]
                del model_output["coords"]
                del model_output["uv"]
                del model_output["pixel_val"]
                del model_output["at_wts"]

                model_outputs.append(model_output)

            log_current_outputs(
                model_outputs,
                cam2world_np,
                intrinsics_np,
            )

            model_output_full = {}

            for k in model_outputs[0].keys():
                outputs = [model_output[k] for model_output in model_outputs]
                val = torch.cat(outputs, dim=-2)
                model_output_full[k] = val

            rgb = model_output_full["rgb"].view(256, 256, 3)
            rgb_gt = model_input["query"]["rgb"][0, i]

            rgb = rgb_np = rgb.detach().cpu().numpy()
            rgb_gt = target_np = rgb_gt.detach().cpu().numpy()

            rgb = np.clip(rgb, -1, 1)
            rgb_gt = np.clip(rgb_gt, -1, 1)

            rgb = (rgb + 1) / 2.0
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

            ssim = structural_similarity(
                rgb_np,
                target_np,
                win_size=11,
                multichannel=True,
                gaussian_weights=True,
                channel_axis=2,
                data_range=2.0,
            )
            ssims.append(ssim)

            print(
                "mse, psnr, lpip, ssim",
                np.mean(mses),
                np.mean(psnrs),
                np.mean(lpips_list),
                np.mean(ssims),
            )

            rgb_np = np.clip(rgb_np, -1, 1)
            rgb_np = (((rgb_np + 1) / 2) * 255).astype(np.uint8)

            writer.append_data(rgb_np)

            # if i > 3:
            #     break

    writer.close()


def render(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:1492",
            world_size=opt.gpus,
            rank=gpu,
        )

    torch.cuda.set_device(gpu)

    test_dataset = RealEstate10k(
        img_root="data_download/realestate/test",
        pose_root="data_download/realestate/test_poses",
        num_ctxt_views=1,
        num_query_views=1,
        query_sparsity=256,
    )

    model = models.CrossAttentionRenderer(model=opt.model, n_view=opt.views)

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict["model"], strict=not opt.reconstruct)

    model = model.cuda()
    scenes = [s.stem for s in test_dataset.all_scenes]

    # TODO (iterate over all and add param to only eval specified scene)
    for idx in range(1):
        all_scene = test_dataset.all_scenes[idx]

        try:
            data = get_camera_pose(
                all_scene,
                "data_download/realestate/test_poses",
                test_dataset.uv,
                views=opt.views,
            )
        except:
            continue

        render_data(data, all_scene, model, rerun_vis=opt.rerun_vis)


if __name__ == "__main__":
    opt = p.parse_args()
    render(0, opt)
