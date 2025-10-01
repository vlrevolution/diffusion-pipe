from pathlib import Path
import re
import tarfile

import peft
import torch
from torch import nn
import torch.nn.functional as F
import safetensors.torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms
import imageio
from lycoris import create_lycoris, LycorisNetwork

from utils.common import (
    is_main_process,
    VIDEO_EXTENSIONS,
    round_to_nearest_multiple,
    round_down_to_multiple,
)


def make_contiguous(*tensors):
    return tuple(x.contiguous() for x in tensors)


def extract_clips(video, target_frames, video_clip_mode):
    # video is (channels, num_frames, height, width)
    frames = video.shape[1]
    if frames < target_frames:
        # TODO: think about how to handle this case. Maybe the video should have already been thrown out?
        print(
            f"video with shape {video.shape} is being skipped because it has less than the target_frames"
        )
        return []

    if video_clip_mode == "single_beginning":
        return [video[:, :target_frames, ...]]
    elif video_clip_mode == "single_middle":
        start = int((frames - target_frames) / 2)
        assert frames - start >= target_frames
        return [video[:, start : start + target_frames, ...]]
    elif video_clip_mode == "multiple_overlapping":
        # Extract multiple clips so we use the whole video for training.
        # The clips might overlap a little bit. We never cut anything off the end of the video.
        num_clips = ((frames - 1) // target_frames) + 1
        start_indices = torch.linspace(0, frames - target_frames, num_clips).int()
        return [video[:, i : i + target_frames, ...] for i in start_indices]
    else:
        raise NotImplementedError(
            f"video_clip_mode={video_clip_mode} is not recognized"
        )


def convert_crop_and_resize(pil_img, width_and_height):
    if pil_img.mode not in ["RGB", "RGBA"] and "transparency" in pil_img.info:
        pil_img = pil_img.convert("RGBA")

    # add white background for transparent images
    if pil_img.mode == "RGBA":
        canvas = Image.new("RGBA", pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    return ImageOps.fit(pil_img, width_and_height)


class PreprocessMediaFile:
    def __init__(
        self,
        config,
        support_video=False,
        framerate=None,
        round_height=16,
        round_width=16,
        round_frames=4,
    ):
        self.config = config
        self.video_clip_mode = config.get("video_clip_mode", "single_beginning")
        print(f"using video_clip_mode={self.video_clip_mode}")
        self.pil_to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.support_video = support_video
        self.framerate = framerate
        print(f"using framerate={self.framerate}")
        self.round_height = round_height
        self.round_width = round_width
        self.round_frames = round_frames
        if self.support_video:
            assert self.framerate
        self.tarfile_map = {}

    def __del__(self):
        for tar_f in self.tarfile_map.values():
            tar_f.close()

    def __call__(self, spec, mask_filepath, size_bucket=None):
        is_video = Path(spec[1]).suffix in VIDEO_EXTENSIONS

        if spec[0] is None:
            tar_f = None
            filepath_or_file = str(spec[1])
        else:
            tar_filename = spec[0]
            tar_f = self.tarfile_map.setdefault(
                tar_filename, tarfile.TarFile(tar_filename)
            )
            filepath_or_file = tar_f.extractfile(str(spec[1]))

        if is_video:
            assert self.support_video
            num_frames = 0
            for frame in imageio.v3.imiter(filepath_or_file, fps=self.framerate):
                num_frames += 1
                height, width = frame.shape[:2]
            video = imageio.v3.imiter(filepath_or_file, fps=self.framerate)
        else:
            num_frames = 1
            pil_img = Image.open(filepath_or_file)
            height, width = pil_img.height, pil_img.width
            video = [pil_img]

        if size_bucket is not None:
            size_bucket_width, size_bucket_height, size_bucket_frames = size_bucket
        else:
            size_bucket_width, size_bucket_height, size_bucket_frames = (
                width,
                height,
                num_frames,
            )

        height_rounded = round_to_nearest_multiple(
            size_bucket_height, self.round_height
        )
        width_rounded = round_to_nearest_multiple(size_bucket_width, self.round_width)
        frames_rounded = (
            round_down_to_multiple(size_bucket_frames - 1, self.round_frames) + 1
        )
        resize_wh = (width_rounded, height_rounded)

        if mask_filepath:
            mask_img = Image.open(mask_filepath).convert("RGB")
            img_hw = (height, width)
            mask_hw = (mask_img.height, mask_img.width)
            if mask_hw != img_hw:
                raise ValueError(
                    f"Mask shape {mask_hw} was not the same as image shape {img_hw}.\n"
                    f"Image path: {spec[1]}\n"
                    f"Mask path: {mask_filepath}"
                )
            mask_img = ImageOps.fit(mask_img, resize_wh)
            mask = torchvision.transforms.functional.to_tensor(mask_img)[0].to(
                torch.float16
            )  # use first channel
        else:
            mask = None

        resized_video = torch.empty((num_frames, 3, height_rounded, width_rounded))
        for i, frame in enumerate(video):
            if not isinstance(frame, Image.Image):
                frame = torchvision.transforms.functional.to_pil_image(frame)
            cropped_image = convert_crop_and_resize(frame, resize_wh)
            resized_video[i, ...] = self.pil_to_tensor(cropped_image)

        if hasattr(filepath_or_file, "close"):
            filepath_or_file.close()

        if not self.support_video:
            return [(resized_video.squeeze(0), mask)]

        # (num_frames, channels, height, width) -> (channels, num_frames, height, width)
        resized_video = torch.permute(resized_video, (1, 0, 2, 3))
        if not is_video:
            return [(resized_video, mask)]
        else:
            videos = extract_clips(resized_video, frames_rounded, self.video_clip_mode)
            return [(video, mask) for video in videos]


class BasePipeline:
    framerate = None

    def load_diffusion_model(self):
        pass

    def get_vae(self):
        raise NotImplementedError()

    def get_text_encoders(self):
        raise NotImplementedError()

    def configure_adapter(self, adapter_config):
        """
        Configures and applies an adapter to the model's transformer.
        This method acts as a dispatcher based on the adapter configuration.
        """
        adapter_library = adapter_config.get("library", "peft").lower()

        if adapter_library == "peft":
            self._create_peft_adapter(adapter_config)
        elif adapter_library == "lycoris":
            self._create_lycoris_adapter(adapter_config)
        else:
            raise NotImplementedError(
                f"Adapter library '{adapter_library}' is not supported."
            )

        # Set dtypes and original_name for all parameters, regardless of library.
        for name, p in self.transformer.named_parameters():
            if not hasattr(p, "original_name"):
                p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config["dtype"])

    def _create_peft_adapter(self, adapter_config):
        """Creates and applies a PEFT adapter (e.g., LoRA)."""
        target_linear_modules = set()
        for name, module in self.transformer.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        adapter_type = adapter_config.get("type", "lora").lower()
        if adapter_type == "lora":
            peft_config = peft.LoraConfig(
                r=adapter_config["rank"],
                lora_alpha=adapter_config["alpha"],
                lora_dropout=adapter_config.get("dropout", 0.0),
                bias="none",
                target_modules=target_linear_modules,
            )
        else:
            raise NotImplementedError(
                f"PEFT adapter type '{adapter_type}' is not implemented."
            )

        self.peft_config = peft_config
        self.adapter_model = peft.get_peft_model(self.transformer, peft_config)
        if is_main_process():
            self.adapter_model.print_trainable_parameters()

    def _create_lycoris_adapter(self, adapter_config):
        """Creates and applies a LyCORIS adapter (e.g., LoKR)."""
        # LyCORIS uses presets to define which modules to target.
        # We will set a preset that targets the modules defined in `adapter_target_modules`.
        LycorisNetwork.apply_preset(
            {
                "target_module": self.adapter_target_modules,
            }
        )

        self.adapter_model = create_lycoris(
            self.transformer,
            multiplier=1.0,
            linear_dim=adapter_config.get("rank", 4),
            linear_alpha=adapter_config.get("alpha", 1.0),
            algo=adapter_config.get("type", "lora"),
            **adapter_config.get("network_args", {}),
        )
        self.adapter_model.apply_to()

    def save_adapter(self, save_dir, state_dict):
        adapter_library = self.config["adapter"].get("library", "peft").lower()

        if adapter_library == "peft":
            # Default PEFT saving logic
            self.peft_config.save_pretrained(save_dir)
            safetensors.torch.save_file(
                state_dict,
                save_dir / "adapter_model.safetensors",
                metadata={"format": "pt"},
            )
        elif adapter_library == "lycoris":
            # LyCORIS saving logic
            comfyui_state_dict = {
                "diffusion_model." + k: v for k, v in state_dict.items()
            }
            safetensors.torch.save_file(
                comfyui_state_dict,
                save_dir / "adapter_model.safetensors",
                metadata={"format": "pt"},
            )
        else:
            raise NotImplementedError(
                f"Saving for adapter library '{adapter_library}' is not supported."
            )

    def load_adapter_weights(self, adapter_path):
        if is_main_process():
            print(f"Loading adapter weights from path {adapter_path}")
        safetensors_files = list(Path(adapter_path).glob("*.safetensors"))
        if len(safetensors_files) == 0:
            raise RuntimeError(f"No safetensors file found in {adapter_path}")
        if len(safetensors_files) > 1:
            raise RuntimeError(f"Multiple safetensors files found in {adapter_path}")
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        modified_state_dict = {}
        model_parameters = set(name for name, p in self.transformer.named_parameters())
        for k, v in adapter_state_dict.items():
            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r"^(transformer|diffusion_model)\.", "", k)
            # Replace weight at end for LoRA format
            k = re.sub(r"\.weight$", ".default.weight", k)
            if k not in model_parameters:
                raise RuntimeError(
                    f"modified_state_dict key {k} is not in the model parameters"
                )
            modified_state_dict[k] = v
        self.transformer.load_state_dict(modified_state_dict, strict=False)

    def load_and_fuse_adapter(self, path):
        peft_config = peft.LoraConfig.from_pretrained(path)
        lora_model = peft.get_peft_model(self.transformer, peft_config)
        self.load_adapter_weights(path)
        lora_model.merge_and_unload()

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def get_call_vae_fn(self, vae):
        raise NotImplementedError()

    def get_call_text_encoder_fn(self, text_encoder):
        raise NotImplementedError()

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    # Get param groups that will be passed into the optimizer. Models can override this, e.g. SDXL
    # supports separate learning rates for unet and text encoders.
    def get_param_groups(self, parameters):
        return [{"params": parameters}]

    # Default loss_fn. MSE between output and target, with mask support.
    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast("cuda", enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                if "pseudo_huber_c" in self.config:
                    c = self.config["pseudo_huber_c"]
                    loss = torch.sqrt((output - target) ** 2 + c**2) - c
                else:
                    loss = F.mse_loss(output, target, reduction="none")
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss

        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError("Block swapping is not implemented for this model")

    def prepare_block_swap_training(self):
        pass

    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass
