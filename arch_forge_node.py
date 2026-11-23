import asyncio
import concurrent.futures
import inspect
import importlib
import numpy as np
import torch
import folder_paths
import comfy.samplers
import comfy.utils
import nodes

SPECS = [('input_image', 'IMAGE', {}), ('checkpoint_name', '@checkpoints', {}), ('lora_name_1', '@loras', {}), ('lora_strength_1_model', 'FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('lora_strength_1_clip', 'FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('lora_name_2', '@loras', {}), ('lora_strength_2_model', 'FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('lora_strength_2_clip', 'FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('lora_name_3', '@loras', {}), ('lora_strength_3_model', 'FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('lora_strength_3_clip', 'FLOAT', {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('positive_prompt', 'STRING', {'default': 'photorealistic aerial architectural photograph of a futuristic cyberpunk city at night, high-angle drone view over a dense metropolitan district, preserve the original composition and camera perspective, preserve the large circular glass-roofed transit hub in the foreground, realistic modern skyscrapers with believable structural engineering, detailed windows and facade materials, wet streets reflecting magenta, cyan, and warm amber lights, atmospheric depth, realistic moonlight, subtle volumetric haze, natural bloom, physically plausible nighttime exposure, high dynamic range, cinematic but realistic color grading, sharp architectural details, realistic urban scale,\n\nrealistic cars driving on the streets, compact electric cars, sedans, taxis, delivery vans, buses, correctly proportioned vehicles, visible headlights and red taillights, vehicles aligned with lanes and road direction, natural spacing between cars, realistic reflections on wet asphalt, a few parked cars near curbs,\n\nsmall realistic pedestrians walking on sidewalks and crossing at intersections, varied clothing silhouettes, coats and jackets appropriate for nighttime, people correctly scaled to the buildings and vehicles, natural walking poses, small groups near storefronts and transit entrances, a few people waiting near crosswalks, subtle motion blur on distant pedestrians,\n\nbelievable street infrastructure, lane markings, traffic signals, crosswalks, sidewalks, road barriers, street lamps, rooftop equipment, signs with minimal mostly unreadable text, realistic windows, realistic shadows, coherent perspective, detailed but uncluttered scene, professional architectural photography, 35mm lens, f/4, natural depth of field, ultra-detailed photorealism'}), ('negative_prompt', 'STRING', {'default': 'anime, illustration, painting, concept art, 3d render, CGI, game asset, miniature city, toy cars, flying cars, malformed cars, merged cars, duplicated cars, cars on rooftops, cars facing random directions, vehicles floating, oversized vehicles, tiny vehicles, impossible roads, broken perspective, warped buildings, melting architecture, duplicated buildings, repeated windows, deformed pedestrians, giant people, tiny people, floating people, fused people, extra limbs, missing limbs, mannequin figures, faceless close-up people, crowd blobs, cloned pedestrians, distorted silhouettes, text, logos, readable gibberish, watermark, signature, excessive bloom, neon fog, overexposure, crushed blacks, oversaturated magenta, excessive contrast, blurry, low resolution, noisy, smeared details, painterly texture'}), ('canny_controlnet', '@controlnet', {}), ('canny_strength', 'FLOAT', {'default': 0.8, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('canny_start', 'FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}), ('canny_end', 'FLOAT', {'default': 0.9, 'min': 0.0, 'max': 1.0, 'step': 0.01}), ('depth_controlnet', '@controlnet', {}), ('depth_strength', 'FLOAT', {'default': 0.5, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('depth_start', 'FLOAT', {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01}), ('depth_end', 'FLOAT', {'default': 0.7, 'min': 0.0, 'max': 1.0, 'step': 0.01}), ('depth_resolution', 'INT', {'default': 1024, 'min': 256, 'max': 2048, 'step': 256}), ('upscale_model', '@upscale_models', {}), ('sharpen_radius', 'INT', {'default': 1, 'min': 0, 'max': 100, 'step': 1}), ('sharpen_sigma', 'FLOAT', {'default': 1.0, 'min': 0.01, 'max': 10.0, 'step': 0.1}), ('sharpen_alpha', 'FLOAT', {'default': 0.5, 'min': 0.0, 'max': 10.0, 'step': 0.1}), ('canny_enable_threshold', 'BOOLEAN', {'default': True}), ('canny_threshold_low', 'FLOAT', {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01}), ('canny_threshold_high', 'FLOAT', {'default': 0.8, 'min': 0.0, 'max': 1.0, 'step': 0.01}), ('seed', 'INT', {'default': 0, 'min': 0, 'max': 18446744073709551615}), ('steps', 'INT', {'default': 32, 'min': 1, 'max': 10000}), ('cfg', 'FLOAT', {'default': 5.0, 'min': 0.0, 'max': 100.0, 'step': 0.1}), ('sampler_name', '@samplers', {'default': 'dpmpp_sde'}), ('scheduler', '@schedulers', {'default': 'karras'}), ('start_at_step', 'INT', {'default': 15, 'min': 0, 'max': 10000}), ('end_at_step', 'INT', {'default': 32, 'min': 0, 'max': 10000}), ('enable_perspective_correction', 'BOOLEAN', {'default': False}), ('perspective_strength', 'FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05}), ('perspective_auto_crop', 'BOOLEAN', {'default': True}), ('depth_midas_a', 'FLOAT', {'default': 6.283185307179586, 'min': 0.0, 'max': 12.566, 'step': 0.01}), ('depth_midas_bg_threshold', 'FLOAT', {'default': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01}), ('enable_ssao', 'BOOLEAN', {'default': True}), ('ssao_resolution', 'INT', {'default': 512, 'min': 256, 'max': 2048, 'step': 256}), ('ssao_strength', 'FLOAT', {'default': 0.3, 'min': 0.0, 'max': 10.0, 'step': 0.01}), ('ssao_radius', 'FLOAT', {'default': 2.0, 'min': 0.1, 'max': 100.0, 'step': 0.1}), ('ssao_blur', 'FLOAT', {'default': 2.0, 'min': 0.0, 'max': 100.0, 'step': 0.1}), ('ssao_specular_threshold', 'INT', {'default': 200, 'min': 0, 'max': 255}), ('ssao_enable_specular', 'BOOLEAN', {'default': True}), ('ssao_tile_size', 'INT', {'default': 1, 'min': 1, 'max': 4096}), ('ssao_blend_factor', 'FLOAT', {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01}), ('ssao_blend_mode', ['multiply', 'normal'], {'default': 'multiply'}), ('enable_lucy', 'BOOLEAN', {'default': True}), ('lucy_iterations', 'INT', {'default': 3, 'min': 1, 'max': 100}), ('lucy_kernel_size', 'INT', {'default': 5, 'min': 3, 'max': 31, 'step': 2}), ('enable_final_sharpen', 'BOOLEAN', {'default': True})]

# Optional inputs are appended after the required ones so saved workflows keep their widget order.
OPTIONAL_DEFAULTS = {
    'workflow': 'SDXL',
    'lora_name_4': 'None', 'lora_strength_4_model': 1.0, 'lora_strength_4_clip': 1.0,
    'moss_depth_lora': 'None', 'moss_canny_lora': 'None',
    'keep_artifacts': False,
    'moss_canny_low': 120, 'moss_canny_high': 220, 'moss_canny_resolution': 1024,
    'moss_leftover_noise_handoff': False,
}
WORKFLOWS = ['SDXL', 'MOSS']
MOSS_EDIT_FORMAT = 'ai-toolkit'
MOSS_IMAGE_STRENGTH = 1.0


def _resolve(result):
    if inspect.isawaitable(result):
        async def _wait(aw):
            return await aw
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_wait(result))).result()
    return result


def call_node(name, **kwargs):
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"ArchForge requires registered node '{name}'. Install/enable its node pack and restart ComfyUI.")
    obj = cls()
    fn = getattr(obj, cls.FUNCTION)
    result = _resolve(fn(**kwargs))
    if isinstance(result, dict):
        if 'result' not in result:
            raise RuntimeError(f"{name} returned no result: {result}")
        result = result['result']
    elif hasattr(result, 'result'):
        result = result.result
    return result


def _file_list(*folders):
    for folder in folders:
        try:
            return folder_paths.get_filename_list(folder)
        except Exception:
            continue
    return []


def moss_encode(clip, image, prompt, vae=None):
    base = dict(clip=clip, prompt=prompt, edit_format=MOSS_EDIT_FORMAT, image_strength=MOSS_IMAGE_STRENGTH)
    if vae is not None:
        base['vae'] = vae
    attempts = (
        lambda: call_node('TextEncodeMOSSImageEdit', images={'image_0': image}, **base),
        lambda: call_node('TextEncodeMOSSImageEdit', **{'images.image_0': image}, **base),
    )
    errors = []
    for attempt in attempts:
        try:
            return attempt()[0]
        except (TypeError, KeyError, ValueError) as exc:
            errors.append(repr(exc))
    raise RuntimeError('TextEncodeMOSSImageEdit rejected both input layouts: ' + ' | '.join(errors))


def apply_loras(model, clip, p, indices):
    for i in indices:
        name = p.get(f'lora_name_{i}', 'None')
        if name and name != 'None':
            model, clip = call_node('LoraLoader', model=model, clip=clip, lora_name=name,
                strength_model=p.get(f'lora_strength_{i}_model', 1.0),
                strength_clip=p.get(f'lora_strength_{i}_clip', 1.0))[:2]
    return model, clip


class ArchForgeNode:
    CATEGORY = "arch_forge"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)

    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        for name, typ, opts in SPECS:
            opts = dict(opts)
            if isinstance(typ, str) and typ.startswith('@'):
                key = typ[1:]
                if key == 'samplers': choices = comfy.samplers.KSampler.SAMPLERS
                elif key == 'schedulers': choices = comfy.samplers.KSampler.SCHEDULERS
                else: choices = folder_paths.get_filename_list(key)
                if key == 'loras': choices = ['None'] + choices
                required[name] = (choices, opts) if opts else (choices,)
            elif isinstance(typ, list): required[name] = (typ, opts) if opts else (typ,)
            else:
                if typ == 'STRING': opts['multiline'] = True
                required[name] = (typ, opts) if opts else (typ,)
        # Append the optional selectors without shifting existing widget values.
        vae_choices = ['Checkpoint VAE'] + folder_paths.get_filename_list('vae')
        lora_choices = ['None'] + folder_paths.get_filename_list('loras')
        strength = {'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.01}
        return {
            'required': required,
            'optional': {
                'vae_name': (vae_choices, {
                    'default': 'Checkpoint VAE',
                    'tooltip': 'Override the checkpoint VAE for image encoding, decoding and ControlNet. Does not change model-family compatibility.',
                }),
                'workflow': (WORKFLOWS, {
                    'default': 'SDXL',
                    'tooltip': 'SDXL: checkpoint + dual ControlNet. MOSS: two-pass MOSS-Image flow with depth/canny control LoRAs. Post-production is shared.',
                }),
                'lora_name_4': (lora_choices, {'default': 'None', 'tooltip': 'SDXL: next LoRA in the chain. MOSS: applied for pass 2 only (after the depth LoRA branch).'}),
                'lora_strength_4_model': ('FLOAT', dict(strength)),
                'lora_strength_4_clip': ('FLOAT', dict(strength)),
                'moss_unet_name': (_file_list('diffusion_models', 'unet'), {'tooltip': 'MOSS diffusion model (UNETLoader).'}),
                'moss_clip_name': (_file_list('text_encoders', 'clip'), {'tooltip': 'MOSS text encoder (CLIPLoader, type flux2).'}),
                'moss_vae_name': (_file_list('vae'), {'tooltip': 'MOSS VAE, e.g. flux2-vae_small.safetensors.'}),
                'moss_depth_lora': (lora_choices, {'default': 'None', 'tooltip': 'MOSS depth control LoRA. Strength = depth_strength.'}),
                'moss_canny_lora': (lora_choices, {'default': 'None', 'tooltip': 'MOSS canny control LoRA used in pass 2 when keep_artifacts is on. Strength = canny_strength.'}),
                'keep_artifacts': ('BOOLEAN', {'default': False, 'tooltip': 'MOSS pass 2. On: canny LoRA + canny map + VAE in the text encoders. Off: no canny LoRA, original image, no VAE in the encoders.'}),
                'moss_canny_low': ('INT', {'default': 120, 'min': 0, 'max': 255}),
                'moss_canny_high': ('INT', {'default': 220, 'min': 0, 'max': 255}),
                'moss_canny_resolution': ('INT', {'default': 1024, 'min': 256, 'max': 2048, 'step': 64}),
                'moss_leftover_noise_handoff': ('BOOLEAN', {'default': False, 'tooltip': 'MOSS: pass 1 returns leftover noise and pass 2 adds no noise.'}),
            },
        }

    def _run_sdxl(self, p, image):
        canny = call_node('Image Canny Filter', images=image,
            enable_threshold='true' if p['canny_enable_threshold'] else 'false',
            threshold_low=p['canny_threshold_low'], threshold_high=p['canny_threshold_high'])[0]
        depth = call_node('MiDaS-DepthMapPreprocessor', image=image,
            a=p['depth_midas_a'], bg_threshold=p['depth_midas_bg_threshold'], resolution=p['depth_resolution'])[0]
        model, clip, vae = call_node('CheckpointLoaderSimple', ckpt_name=p['checkpoint_name'])[:3]
        vae_name = p.get('vae_name', 'Checkpoint VAE')
        if vae_name != 'Checkpoint VAE':
            vae = call_node('VAELoader', vae_name=vae_name)[0]
        model, clip = apply_loras(model, clip, p, range(1, 5))
        positive = call_node('CLIPTextEncode', clip=clip, text=p['positive_prompt'])[0]
        negative = call_node('CLIPTextEncode', clip=clip, text=p['negative_prompt'])[0]
        for key, hint in [('canny',canny), ('depth',depth)]:
            cn = call_node('ControlNetLoader', control_net_name=p[key+'_controlnet'])[0]
            positive, negative = call_node('ControlNetApplyAdvanced', positive=positive, negative=negative,
                control_net=cn, image=hint, strength=p[key+'_strength'],
                start_percent=p[key+'_start'], end_percent=p[key+'_end'], vae=vae)[:2]
        latent = call_node('VAEEncode', vae=vae, pixels=image)[0]
        samples = call_node('KSamplerAdvanced', model=model, positive=positive, negative=negative,
            latent_image=latent, add_noise='enable', noise_seed=p['seed'], steps=p['steps'], cfg=p['cfg'],
            sampler_name=p['sampler_name'], scheduler=p['scheduler'], start_at_step=p['start_at_step'],
            end_at_step=p['end_at_step'], return_with_leftover_noise='disable')[0]
        return call_node('VAEDecode', vae=vae, samples=samples)[0]

    def _run_moss(self, p, image):
        for key, label in (('moss_unet_name', 'MOSS diffusion model'), ('moss_clip_name', 'MOSS text encoder'),
                           ('moss_vae_name', 'MOSS VAE'), ('moss_depth_lora', 'MOSS depth LoRA')):
            if not p.get(key) or p[key] == 'None':
                raise ValueError(f'MOSS workflow: select a {label} ({key}).')
        keep = bool(p['keep_artifacts'])
        if keep and (not p.get('moss_canny_lora') or p['moss_canny_lora'] == 'None'):
            raise ValueError('MOSS workflow: keep_artifacts is on but no canny LoRA (moss_canny_lora) is selected.')
        handoff = bool(p['moss_leftover_noise_handoff'])
        start, end, steps = p['start_at_step'], p['end_at_step'], p['steps']

        depth = call_node('MiDaS-DepthMapPreprocessor', image=image,
            a=p['depth_midas_a'], bg_threshold=p['depth_midas_bg_threshold'], resolution=p['depth_resolution'])[0]
        model = call_node('UNETLoader', unet_name=p['moss_unet_name'], weight_dtype='default')[0]
        clip = call_node('CLIPLoader', clip_name=p['moss_clip_name'], type='flux2', device='default')[0]
        vae = call_node('VAELoader', vae_name=p['moss_vae_name'])[0]

        # LoRAs 1-3 feed both passes; the depth LoRA branches off here for pass 1.
        model, clip = apply_loras(model, clip, p, range(1, 4))
        model_pass1 = call_node('LoraLoaderModelOnly', model=model, lora_name=p['moss_depth_lora'],
            strength_model=p['depth_strength'])[0]
        # LoRA 4 is pass-2 only; its CLIP output is used by every text encode.
        model_pass2, clip = apply_loras(model, clip, p, [4])

        positive1 = moss_encode(clip, depth, p['positive_prompt'], vae)
        negative1 = moss_encode(clip, depth, p['negative_prompt'], vae)
        latent = call_node('VAEEncode', vae=vae, pixels=image)[0]
        samples = call_node('KSamplerAdvanced', model=model_pass1, positive=positive1, negative=negative1,
            latent_image=latent, add_noise='enable', noise_seed=p['seed'], steps=steps, cfg=p['cfg'],
            sampler_name=p['sampler_name'], scheduler=p['scheduler'], start_at_step=start, end_at_step=end,
            return_with_leftover_noise='enable' if handoff else 'disable')[0]

        if keep:
            canny = call_node('CannyEdgePreprocessor', image=image, low_threshold=p['moss_canny_low'],
                high_threshold=p['moss_canny_high'], resolution=p['moss_canny_resolution'])[0]
            model_pass2 = call_node('LoraLoaderModelOnly', model=model_pass2, lora_name=p['moss_canny_lora'],
                strength_model=p['canny_strength'])[0]
            positive2 = moss_encode(clip, canny, p['positive_prompt'], vae)
            negative2 = moss_encode(clip, canny, p['negative_prompt'], vae)
        else:
            positive2 = moss_encode(clip, image, p['positive_prompt'], None)
            negative2 = moss_encode(clip, image, p['negative_prompt'], None)

        samples = call_node('KSamplerAdvanced', model=model_pass2, positive=positive2, negative=negative2,
            latent_image=samples, add_noise='disable' if handoff else 'enable', noise_seed=p['seed'], steps=steps,
            cfg=p['cfg'], sampler_name=p['sampler_name'], scheduler=p['scheduler'], start_at_step=end,
            end_at_step=steps, return_with_leftover_noise='disable')[0]
        return call_node('VAEDecode', vae=vae, samples=samples)[0]

    def execute(self, **p):
        for name, typ, opts in SPECS:
            if name not in p and 'default' in opts: p[name] = opts['default']
        for name, value in OPTIONAL_DEFAULTS.items():
            p.setdefault(name, value)
        moss = p['workflow'] == 'MOSS'
        if not 0 <= p['start_at_step'] < p['end_at_step'] <= p['steps']:
            raise ValueError('Require 0 <= start_at_step < end_at_step <= steps.')
        if not moss:
            for key in ('canny','depth'):
                if p[key+'_start'] > p[key+'_end']: raise ValueError(f'{key}: start exceeds end.')
            if p['canny_threshold_low'] > p['canny_threshold_high']:
                raise ValueError('Canny low threshold must not exceed high threshold.')
        elif p['moss_canny_low'] > p['moss_canny_high']:
            raise ValueError('MOSS canny low threshold must not exceed high threshold.')
        if p['enable_lucy'] and p['lucy_kernel_size'] % 2 == 0:
            raise ValueError('Lucy kernel_size must be odd.')
        image = p['input_image']
        decoded = self._run_moss(p, image) if moss else self._run_sdxl(p, image)
        processed = decoded
        if p['enable_ssao']:
            ao_depth = call_node('MiDaS-DepthMapPreprocessor', image=decoded,
                a=p['depth_midas_a'], bg_threshold=p['depth_midas_bg_threshold'], resolution=p['ssao_resolution'])[0]
            if ao_depth.shape[1:3] != decoded.shape[1:3]:
                ao_depth = comfy.utils.common_upscale(ao_depth.movedim(-1,1), decoded.shape[2], decoded.shape[1], 'bilinear', 'disabled').movedim(1,-1)
            composite = call_node('Image SSAO (Ambient Occlusion)', images=decoded, depth_images=ao_depth,
                strength=p['ssao_strength'], radius=p['ssao_radius'], ao_blur=p['ssao_blur'],
                specular_threshold=p['ssao_specular_threshold'],
                enable_specular_masking='True' if p['ssao_enable_specular'] else 'False', tile_size=p['ssao_tile_size'])[0]
            processed = call_node('ImageBlend', image1=decoded, image2=composite,
                blend_factor=p['ssao_blend_factor'], blend_mode=p['ssao_blend_mode'])[0]
        upscale = call_node('UpscaleModelLoader', model_name=p['upscale_model'])[0]
        final = call_node('ImageUpscaleWithModel', upscale_model=upscale, image=processed)[0]
        if p['enable_lucy']:
            final = call_node('Image Lucy Sharpen', images=final, iterations=p['lucy_iterations'], kernel_size=p['lucy_kernel_size'])[0]
        if p['enable_final_sharpen'] and p['sharpen_radius'] > 0:
            final = call_node('ImageSharpen', image=final, sharpen_radius=p['sharpen_radius'], sigma=p['sharpen_sigma'], alpha=p['sharpen_alpha'])[0]
        if p['enable_perspective_correction']:
            try:
                helper = importlib.import_module('.perspective', package=__package__)
            except (ImportError, TypeError) as exc:
                raise RuntimeError('Auto-straighten requires your existing perspective.py in this package. It was not supplied; disable auto-straighten or restore that file.') from exc
            if final.shape[0] != 1:
                raise ValueError('Auto-crop perspective correction currently supports batch size 1 only.')
            array = final[0].detach().mul(255).clamp(0,255).byte().cpu().numpy()
            corrected = helper.compute_perspective_correction(array, strength=p['perspective_strength'], auto_crop=p['perspective_auto_crop'])
            final = torch.from_numpy(np.ascontiguousarray(corrected)).float().unsqueeze(0).div(255).to(final.device)
        return (final,)

NODE_CLASS_MAPPINGS = {'ArchForgeNode': ArchForgeNode}
NODE_DISPLAY_NAME_MAPPINGS = {'ArchForgeNode': 'Arch Forge'}
