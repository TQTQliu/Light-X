import gc
import os
import torch
from tools.depthcrafter.depth_infer import DepthCrafterDemo
import numpy as np
import torch
from transformers import T5EncoderModel
from PIL import Image
from core.models.crosstransformer3d import CrossTransformer3DModel
from core.models.autoencoder_magvit import AutoencoderKLCogVideoX
from core.pipeline.pipeline_cogvideox_fun_inpaint import CogVideoXFunInpaintPipeline
from tools.utils import *
from diffusers import (
    AutoencoderKL,
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import shutil
import cv2
from tools.iclight.demo import relight_image
from tools.iclight.demo_bg import relight_image_bg
from core.data.hdri_processer import HDRI_Preprocessor


class LightX:
    def __init__(self, opts, gradio=False):
        if opts.recam_vd:
            self.funwarp = Warper(device=opts.device)
            self.depth_estimater = DepthCrafterDemo(
                unet_path=opts.unet_path,
                pre_train_path=opts.pre_train_path,
                cpu_offload=opts.cpu_offload,
                device=opts.device,
            )
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(
            opts.blip_path, torch_dtype=torch.float16
        ).to(opts.device)
        self.setup_diffusion(opts)

    def generate_video(self, opts, frames, ref_id, relit_img, warped_relit_mask, cond_video, cond_masks, prompt, relit_cond_type):
      
        frames_ref = get_nearby_10_frames(frames, ref_id)
        relit_img = F.interpolate(
            relit_img, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        
        if relit_cond_type == 'hdr' or relit_cond_type == 'ref':
            relit_img = relit_img[:,:,None]
            light_video = relit_img.clone().repeat(1,1,opts.video_length,1,1)
            if relit_cond_type == 'hdr':
                relit_mask = torch.ones_like(cond_masks)*255*0.5 
            else:
                relit_mask = torch.ones_like(cond_masks)*255*0.25
        else:
            light_video = torch.zeros_like(cond_video)
            light_video[:,:,ref_id] = relit_img.clone()
            relit_img = relit_img[:,:,None]
            warped_relit_mask = F.interpolate(warped_relit_mask, size=opts.sample_size, mode='nearest')
            relit_mask = torch.ones_like(cond_masks)
            relit_mask[:, :, ref_id] = 1 - warped_relit_mask
            relit_mask = relit_mask * 255
        
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        if opts.mode != 'bullet':
            if hasattr(self, 'depth_estimater'):
                del self.depth_estimater
            del self.caption_processor
            del self.captioner
            gc.collect()
            torch.cuda.empty_cache()

        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
                light_video=light_video,
                relit_img=relit_img,
                relit_mask=relit_mask,
            ).videos
        
        return sample
    

    def process_with_recam(self, opts, frames, ref_id, relit_frame, prompt, relit_cond_type='ic', bullet_time=False):
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        assert frames.shape[0] == opts.video_length

        if opts.mode == 'direct':
            opts.cut = 20
            pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.cut)
        elif opts.mode == 'dolly-zoom':
            pose_s, pose_t, K = self.get_poses_f(opts, depths, num_frames=opts.video_length, f_new=250)
        else:
            pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            if opts.mode == 'gradual':
                warped_frame2, mask2, _, _ = self.funwarp.forward_warp(
                    frames[i : i + 1],
                    None,
                    depths[i : i + 1],
                    pose_s[i : i + 1],
                    pose_t[i : i + 1],
                    K[i : i + 1],
                    None,
                    opts.mask,
                    twice=False,
                )
            elif opts.mode == 'bullet':
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[:1],
                    None,
                    depths[:1],
                    pose_s[0:1],
                    pose_t[i : i + 1],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
            elif opts.mode == 'direct':
                if i < opts.cut:
                    warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                        frames[0:1],
                        None,
                        depths[0:1],
                        pose_s[0:1],
                        pose_t[i : i + 1],
                        K[0:1],
                        None,
                        opts.mask,
                        twice=False,
                    )
                else:
                    warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                        frames[i - opts.cut : i - opts.cut + 1],
                        None,
                        depths[i - opts.cut : i - opts.cut + 1],
                        pose_s[0:1],
                        pose_t[-1:],
                        K[0:1],
                        None,
                        opts.mask,
                        twice=False,
                    )
            elif opts.mode == 'dolly-zoom':
                warped_frame2, mask2,  _, _ = self.funwarp.forward_warp(
                    frames[i : i + 1],
                    None,
                    depths[i : i + 1],
                    pose_s[i : i + 1],
                    pose_t[i : i + 1],
                    K[0 : 1],
                    K[i : i + 1],
                    opts.mask,
                    twice=False,
                )
            else:
                raise NotImplementedError

            if i == ref_id:  # relit point clouds rendering
                if relit_cond_type == 'ic':
                    warped_relit_frame, warped_relit_mask, _, _ = self.funwarp.forward_warp(
                        relit_frame,
                        None,
                        depths[i : i + 1],
                        pose_s[i : i + 1],
                        pose_t[i : i + 1],
                        K[i : i + 1],
                        None,
                        opts.mask,
                        twice=False,
                    )
                    relit_img = (warped_relit_frame + 1.0) / 2.0
                else:
                    relit_img = (relit_frame + 1.0) / 2.0
                    warped_relit_mask = None

            warped_images.append(warped_frame2)
            masks.append(mask2)
            
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        masks = torch.cat(masks)
        cond_video = F.interpolate(
            cond_video, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        masks = F.interpolate(masks, size=opts.sample_size, mode='nearest')
        masks = masks.permute(1, 0, 2, 3).unsqueeze(0)

        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
       
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0).to(opts.device) 
        cond_masks = (1.0 - masks) * 255.0
        cond_masks = cond_masks.to(opts.device) 
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0

        output_video = self.generate_video(opts, frames, ref_id, relit_img, warped_relit_mask, cond_video, cond_masks, prompt, relit_cond_type)
       
        return output_video, frames, cond_video, masks


    def process_wo_recam(self, opts, frames, ref_id, relit_frame, prompt, relit_cond_type='ic'):
        relit_img = (relit_frame + 1.0) / 2.0
        warped_relit_mask = torch.ones(1, 1, relit_img.shape[2], relit_img.shape[3]).to(opts.device)
        
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
        )
        frames = F.interpolate(
            frames, size=opts.sample_size, mode='bilinear', align_corners=False
        )
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        cond_masks = torch.zeros_like(frames[:,:1]).to(opts.device)
        cond_video = frames.clone().to(opts.device)

        output_video = self.generate_video(opts, frames, ref_id, relit_img, warped_relit_mask, cond_video, cond_masks, prompt, relit_cond_type)
        
        return output_video, frames, cond_video, 1-cond_masks

    def infer(self, opts):

        frames = read_video_frames(
            opts.video_path, opts.video_length, opts.stride, opts.max_res
        )

        if opts.relit_vd:
            if opts.relit_cond_img is None:
                relit_params = parse_params(opts.relit_txt)
                print(relit_params)
                if opts.relit_cond_type == 'bg':
                    assert os.path.exists(relit_params['bg_source'])
                    replace_background_with_image(foreground_video_path=opts.video_path, mask_video_path=os.path.join(os.path.dirname(opts.video_path), 'seg/input_mask.mp4'), \
                        background_image_path=relit_params['bg_source'], output_video_path=os.path.join(os.path.dirname(opts.video_path), 'fuse.mp4'))
                    frames = read_video_frames(
                        os.path.join(os.path.dirname(opts.video_path), 'fuse.mp4'), opts.video_length, opts.stride, opts.max_res
                    )
                    save_specific_frame(opts.video_path, f"{opts.save_dir}/{opts.ref_id}.png", frame_id=opts.ref_id)
                    relit_frame = Image.open(f"{opts.save_dir}/{opts.ref_id}.png").convert("RGB")
                    relit_frame = relight_image_bg(input_fg_path=f"{opts.save_dir}/{opts.ref_id}.png", input_bg_path=relit_params['bg_source'])
                elif opts.relit_cond_type == 'ic':
                    save_specific_frame(opts.video_path, f"{opts.save_dir}/{opts.ref_id}.png", frame_id=opts.ref_id)
                    relit_frame = Image.open(f"{opts.save_dir}/{opts.ref_id}.png").convert("RGB")
                    relit_frame = np.array(relit_frame)
                    relit_frame = relight_image(relit_frame, **relit_params)
                cv2.imwrite(f'{opts.save_dir}/relight.png', relit_frame[..., ::-1])
                relit_frame = relit_frame.astype(np.float32) / 255.0
                relit_frame = torch.from_numpy(relit_frame).permute(2,0,1).to(opts.device)[None]
                relit_frame = F.interpolate(relit_frame, size=frames.shape[1:-1], mode='bilinear', align_corners=False)
                relit_frame = relit_frame * 2.0 - 1.0
            else:
                if opts.relit_cond_type == 'ic':
                    relit_frame = load_image_as_tensor(opts.relit_cond_img)[None]
                    relit_frame = F.interpolate(relit_frame, size=frames.shape[1:-1], mode='bilinear', align_corners=False)
                    relit_frame = relit_frame * 2.0 - 1.0
                elif opts.relit_cond_type == 'ref':
                    ref_light_img = load_image_as_tensor(opts.relit_cond_img)[None].to(opts.device)
                    relit_frame = F.interpolate(ref_light_img, size=frames.shape[1:-1], mode='bilinear', align_corners=False)
                    relit_frame = relit_frame * 2.0 - 1.0
                elif opts.relit_cond_type == 'hdr':
                    hdri_processor = HDRI_Preprocessor()
                    assert opts.relit_cond_img.endswith('.exr'), "HDR map must be an EXR file"
                    hdri_processor.load_hdri(opts.relit_cond_img)
                    hdri_cond = hdri_processor.get_rotate_hdri_cond(hdri_rot_roll=0.0)
                    relit_frame = hdri_cond.to(opts.device)
                    relit_frame = F.interpolate(relit_frame, size=frames.shape[1:-1], mode='bilinear', align_corners=False)
        else:
            relit_frame = torch.from_numpy(frames[opts.ref_id]).clone().permute(2,0,1).to(opts.device)
            relit_frame = relit_frame[None] * 2.0 - 1.0

        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        if opts.mode != 'bullet':
            if opts.recam_vd:
                output_video, input_video, cond_video, masks = self.process_with_recam(
                    opts, frames, opts.ref_id, relit_frame, prompt, opts.relit_cond_type, bullet_time=False
                )
            else:
                output_video, input_video, cond_video, masks = self.process_wo_recam(
                    opts, frames, opts.ref_id, relit_frame, prompt, opts.relit_cond_type
                )
        else:
            output_video, input_video, cond_video, masks = self.process_wo_recam(
                opts, frames.copy(), opts.ref_id, relit_frame, prompt, opts.relit_cond_type
            )
            bullet_output_video, input_video, cond_video, masks = self.process_with_recam(
                opts, frames, opts.ref_id, relit_frame, prompt, opts.relit_cond_type, bullet_time=True
            )
            output_video = torch.cat([bullet_output_video,output_video], dim=2)
        
        if opts.mode != 'direct':
            save_video(
                input_video.squeeze(0).permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'input.mp4'),
                fps=opts.fps,
            )
            save_video(
                cond_video.squeeze(0).permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'render.mp4'),
                fps=opts.fps,
            )
            save_video(
                masks.squeeze(0).repeat(3, 1, 1, 1).permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'mask.mp4'),
                fps=opts.fps,
            )
            save_video(
                output_video.squeeze(0).permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'gen.mp4'),
                fps=opts.fps,
            )

            if opts.mode != 'bullet':
                tensor_left = input_video.squeeze(0).permute(1, 2, 3, 0).to(opts.device)
                tensor_right = output_video.squeeze(0).permute(1, 2, 3, 0).to(opts.device)
                interval = torch.ones(opts.video_length, 384, 30, 3).to(opts.device)
                result = torch.cat((tensor_left, interval, tensor_right), dim=-2)
                result_reverse = torch.flip(result, dims=[0])
                final_result = torch.cat((result, result_reverse[1:, :, :, :]), dim=0)
                save_video(
                    final_result,
                    os.path.join(opts.save_dir, 'viz.mp4'),
                    fps=opts.fps * 2,
                )
            else:
                tensor_left = input_video.squeeze(0).to(opts.device)
                tensor_left_full = torch.cat(
                    [tensor_left[:, :1, :, :].repeat(1, opts.video_length, 1, 1), tensor_left], dim=1
                )
                tensor_right_full = output_video.squeeze(0).to(opts.device)
                interval = torch.ones(3, opts.video_length * 2, 384, 30).to(opts.device)
                result = torch.cat((tensor_left_full, interval, tensor_right_full), dim=-1)
                result_reverse = torch.flip(result, dims=[1])
                final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
                save_video(
                    final_result.permute(1, 2, 3, 0),
                    os.path.join(opts.save_dir, 'viz.mp4'),
                    fps=opts.fps * 4,
                )
        else:
            save_video(
                input_video.squeeze(0).permute(1, 2, 3, 0)[: opts.video_length - opts.cut],
                os.path.join(opts.save_dir, 'input.mp4'),
                fps=opts.fps,
            )
            save_video(
                cond_video.squeeze(0).permute(1, 2, 3, 0)[opts.cut :],
                os.path.join(opts.save_dir, 'render.mp4'),
                fps=opts.fps,
            )
            save_video(
                masks.squeeze(0).repeat(3, 1, 1, 1).permute(1, 2, 3, 0)[opts.cut :],
                os.path.join(opts.save_dir, 'mask.mp4'),
                fps=opts.fps,
            )
            save_video(
                output_video.squeeze(0).permute(1, 2, 3, 0)[opts.cut :],
                os.path.join(opts.save_dir, 'gen.mp4'),
                fps=opts.fps,
            )

            tensor_left = input_video.squeeze(0).permute(1, 2, 3, 0)[: opts.video_length - opts.cut].to(opts.device)
            tensor_right = output_video.squeeze(0).permute(1, 2, 3, 0)[opts.cut :].to(opts.device)
            interval = torch.ones(opts.video_length - opts.cut, 384, 30, 3).to(opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=-2)
            result_reverse = torch.flip(result, dims=[0])
            final_result = torch.cat((result, result_reverse[1:, :, :, :]), dim=0)
            save_video(
                final_result,
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def get_caption(self, opts, image):
        image_array = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
            opts.device, torch.float16
        )
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text + opts.refine_prompt

    def get_poses(self, opts, depths, num_frames):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K

    def get_poses_f(self, opts, depths, num_frames, f_new):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  
        cy = 288.0  
        f = 500
        f_values = torch.linspace(f, f_new, num_frames, device=opts.device)
        K = torch.zeros((num_frames, 3, 3), device=opts.device)
        K[:, 0, 0] = f_values
        K[:, 1, 1] = f_values
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx : opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K

    def setup_diffusion(self, opts):
        transformer = CrossTransformer3DModel.from_pretrained(opts.transformer_path).to(
            opts.weight_dtype
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            opts.model_name, subfolder="vae"
        ).to(opts.weight_dtype)
        text_encoder = T5EncoderModel.from_pretrained(
            opts.model_name, subfolder="text_encoder", torch_dtype=opts.weight_dtype
        )
        # Get Scheduler
        Choosen_Scheduler = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM_Cog": CogVideoXDDIMScheduler,
            "DDIM_Origin": DDIMScheduler,
        }[opts.sampler_name]
        scheduler = Choosen_Scheduler.from_pretrained(
            opts.model_name, subfolder="scheduler"
        )

        self.pipeline = CogVideoXFunInpaintPipeline.from_pretrained(
            opts.model_name,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=opts.weight_dtype,
        )

        if opts.low_gpu_memory_mode:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()