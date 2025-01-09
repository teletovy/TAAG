'''
앞으로의 pipeline base code - 0918
'''
import inspect
# import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import logging
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
from torch.optim.adam import Adam
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin, FromSingleFileMixin, IPAdapterMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from utils.ptp_utils_plus import AttendExciteAttnProcessor, AttentionStore
from utils.attn_utils import fn_smoothing_func, fn_get_topk, fn_clean_mask, fn_get_otsu_mask
from tqdm import tqdm
from sklearn.cluster import KMeans
import nltk
import cv2
import os

logging.basicConfig(format='%(asctime)s: %(message)s',level=logging.INFO)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionAttendAndExcitePipeline

        >>> pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        ... ).to("cuda")


        >>> prompt = "a cat and a frog"

        >>> # use get_indices function to find out indices of the tokens you want to alter
        >>> pipe.get_indices(prompt)
        {0: '<|startoftext|>', 1: 'a</w>', 2: 'cat</w>', 3: 'and</w>', 4: 'a</w>', 5: 'frog</w>', 6: '<|endoftext|>'}

        >>> token_indices = [2, 5]
        >>> seed = 6141
        >>> generator = torch.Generator("cuda").manual_seed(seed)

        >>> images = pipe(
        ...     prompt=prompt,
        ...     token_indices=token_indices,
        ...     guidance_scale=7.5,
        ...     generator=generator,
        ...     num_inference_steps=50,
        ...     max_iter_to_alter=25,
        ... ).images

        >>> image = images[0]
        >>> image.save(f"../images/{prompt}_{seed}.png")
        ```
"""


class TAMMOPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion and Attend-and-Excite and Latent Consistency Models.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.K = 1
        self.cross_attention_maps_cache = None

        if safety_checker is None and requires_safety_checker:
            logging.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logging.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    #NOTE 1107 nsfw on/off차이가 크지 않아서 다시 off... -> 회의 이후에 다시 결정할 것
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept
    #NOTE 1105 앞으로는 이걸 on!
    # def run_safety_checker(self, image, device, dtype):
    #     return image, None

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        indices,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        indices_is_list_ints = isinstance(indices, list) and isinstance(indices[0], int)
        indices_is_list_list_ints = (
            isinstance(indices, list) and isinstance(indices[0], list) and isinstance(indices[0][0], int)
        )

        if not indices_is_list_ints and not indices_is_list_list_ints:
            raise TypeError("`indices` must be a list of ints or a list of a list of ints")

        if indices_is_list_ints:
            indices_batch_size = 1
        elif indices_is_list_list_ints:
            indices_batch_size = len(indices)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if indices_batch_size != prompt_batch_size:
            raise ValueError(
                f"indices batch size must be same as prompt batch size. indices batch size: {indices_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents    
    
    def fn_calc_kld_loss_func(self, log_var, mu):
        return torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp()), dim=0)
    
    # def fn_compute_loss(
    #     self, 
    #     indices: List[int], 
    #     smooth_attentions: bool = True,
    #     K: int = 1,
    #     attention_res: int = 16,) -> torch.Tensor:
        
    #     # -----------------------------
    #     # cross-attention response loss
    #     # -----------------------------
    #     aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
    #         from_where=("up", "down", "mid"), is_cross=True)
        
    #     # cross attention map preprocessing
    #     cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:-1]
    #     cross_attention_maps = cross_attention_maps * 100
    #     cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

    #     # Shift indices since we removed the first token
    #     indices = [index - 1 for index in indices]

    #     # clean_cross_attention_loss
    #     clean_cross_attention_loss = 0.

    #     # Extract the maximum values
    #     topk_value_list, topk_coord_list_list = [], []
    #     for i in indices:
    #         cross_attention_map_cur_token = cross_attention_maps[:, :, i]
    #         if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            
    #         topk_coord_list, _ = fn_get_topk(cross_attention_map_cur_token, K=K)

    #         topk_value = 0
    #         for coord_x, coord_y in topk_coord_list: topk_value = topk_value + cross_attention_map_cur_token[coord_x, coord_y]
    #         topk_value = topk_value / K

    #         topk_value_list.append(topk_value)
    #         topk_coord_list_list.append(topk_coord_list)

    #         # -----------------------------------
    #         # clean cross_attention_map_cur_token
    #         # -----------------------------------
    #         clean_cross_attention_map_cur_token                     = cross_attention_map_cur_token
    #         clean_cross_attention_map_cur_token_mask                = fn_get_otsu_mask(clean_cross_attention_map_cur_token)
    #         clean_cross_attention_map_cur_token_mask                = fn_clean_mask(clean_cross_attention_map_cur_token_mask, topk_coord_list[0][0], topk_coord_list[0][1])
            
    #         clean_cross_attention_map_cur_token_foreground          = clean_cross_attention_map_cur_token * clean_cross_attention_map_cur_token_mask + (1 - clean_cross_attention_map_cur_token_mask)
    #         clean_cross_attention_map_cur_token_background          = clean_cross_attention_map_cur_token * (1 - clean_cross_attention_map_cur_token_mask)

    #         if clean_cross_attention_map_cur_token_background.max() > clean_cross_attention_map_cur_token_foreground.min():
    #             clean_cross_attention_loss = clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max()
    #         else: clean_cross_attention_loss = clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max() * 0

    #     cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in topk_value_list]
    #     cross_attn_loss = max(cross_attn_loss_list)

    #     # ----------------------------
    #     # self-attention conflict loss
    #     # ----------------------------
    #     self_attention_maps = self.attention_store.aggregate_attention(
    #         from_where=("up", "down", "mid"), is_cross=False)
        
    #     self_attention_map_list = []
    #     for topk_coord_list in topk_coord_list_list:
    #         self_attention_map_cur_token_list = []
    #         for coord_x, coord_y in topk_coord_list:

    #             self_attention_map_cur_token = self_attention_maps[coord_x, coord_y]
    #             self_attention_map_cur_token = self_attention_map_cur_token.view(attention_res, attention_res).contiguous()
    #             self_attention_map_cur_token_list.append(self_attention_map_cur_token)

    #         if len(self_attention_map_cur_token_list) > 0:
    #             self_attention_map_cur_token = sum(self_attention_map_cur_token_list) / len(self_attention_map_cur_token_list)
    #             if smooth_attentions: self_attention_map_cur_token = fn_smoothing_func(self_attention_map_cur_token)
    #         else:
    #             self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
    #             self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

    #         self_attention_map_list.append(self_attention_map_cur_token)

    #     self_attn_loss, number_self_attn_loss_pair = 0, 0
    #     number_token = len(self_attention_map_list)
    #     for i in range(number_token):
    #         for j in range(i + 1, number_token): 
    #             number_self_attn_loss_pair = number_self_attn_loss_pair + 1
    #             self_attention_map_1 = self_attention_map_list[i]
    #             self_attention_map_2 = self_attention_map_list[j]

    #             self_attention_map_min = torch.min(self_attention_map_1, self_attention_map_2) 
    #             self_attention_map_sum = (self_attention_map_1 + self_attention_map_2) 
    #             cur_self_attn_loss = (self_attention_map_min.sum() / (self_attention_map_sum.sum() + 1e-6))
    #             self_attn_loss = self_attn_loss + cur_self_attn_loss

    #     if number_self_attn_loss_pair > 0: self_attn_loss = self_attn_loss / number_self_attn_loss_pair

    #     cross_attn_loss = cross_attn_loss * torch.ones(1).to(self._execution_device)
    #     self_attn_loss  = self_attn_loss * torch.ones(1).to(self._execution_device)

    #     if cross_attn_loss > 0.5:    self_attn_loss = self_attn_loss * 0
    #     joint_loss = cross_attn_loss * 1. +  self_attn_loss * 1. + clean_cross_attention_loss * 1.

    #     return joint_loss, cross_attn_loss, self_attn_loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(
        self,
        sampling_step: int,        
        latents: torch.Tensor,
        indices: List[int],
        opt_key: List[str],        
        text_embeddings: torch.Tensor,
        step_size: float,
        t: int,
        K: int,
        max_refinement_steps: int = 20,        
        logger: Any = None,
        mask_loss_type: str = 'lg'        
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        # target_loss = max(0, 1.0 - threshold)
        # target_self_loss = 0.3
        opt_succeed = False # NOTE should check if the loop is fine
        while opt_succeed:
            iteration += 1
            latents = latents.clone().detach().requires_grad_(True)
            self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            self.unet.zero_grad()

            opt_loss, succeeds = self.fn_adaptive_optimization(indices=indices, opt_key=opt_key, mask=self.mask, smooth_attentions=True,
                                                               K=K, mask_loss_type=mask_loss_type, logger=logger)
            
            if len(succeeds) == 1:
                opt_succeed = succeeds[0]
            else:
                opt_succeed = True
                for succeed in succeeds:
                    opt_succeed *= succeed
                # opt_succeed = succeeds[0] * succeeds[1]
            
            if opt_loss != 0: latents = self._update_latent(latents, opt_loss, step_size)

            if logger is not None:
                logger.info(f"\t Sampling {sampling_step}th Try {iteration} loss {opt_loss.item():0.4f}.")
            else:
                logging.info(f"\t Sampling {sampling_step}th Try {iteration} loss {opt_loss.item():0.4f}.")

            if iteration >= max_refinement_steps:
                if logger is not None:
                    logger.info(f"\t Sampling {sampling_step}th Try {iteration} loss {opt_loss.item():0.4f}.")
                else:
                    logging.info(f"\t Exceeded max number of iterations ({max_refinement_steps})! ")
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        _ = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        self.unet.zero_grad()
        
        opt_loss, succeeds = self.fn_adaptive_optimization(indices=indices, opt_key=opt_key, mask=self.mask, smooth_attentions=True,
                                                               K=K, mask_loss_type=mask_loss_type, logger=logger)
        if logger is not None:
            logger.info(f"\t Finished with loss of: {opt_loss:0.4f}.")
        else:
            logging.info(f"\t Finished with loss of: {opt_loss:0.4f}.")
        
        return opt_loss, succeeds
        # return joint_loss, cross_attn_loss, self_attn_loss, clean_loss, latents, None

    # InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization
    def fn_initno(
        self,
        latents: torch.Tensor,
        indices: List[int],
        text_embeddings: torch.Tensor, # NOTE from now text_embeddings is only text condition embeddings
        initno_opt_keys: List[str],
        K: int,
        use_grad_checkpoint: bool = False,
        initno_lr: float = 1e-2,
        max_step: int = 50,
        round: int = 0,        
        num_inference_steps: int = 50,
        device: str = "",
        denoising_step_for_loss: int = 1,
        guidance_scale: int = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        do_classifier_free_guidance: bool = False,   
        attention_resolution: Tuple = (16, 16), # TODO multiple resolution ver으로 바꿔야 함
        logger: Any = None
         
    ):
        '''InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization'''
        latents = latents.clone().detach()
        log_var, mu = torch.zeros_like(latents), torch.zeros_like(latents)
        log_var, mu = log_var.clone().detach().requires_grad_(True), mu.clone().detach().requires_grad_(True)
        optimizer = Adam([log_var, mu], lr=initno_lr, eps=1e-3)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        optimization_succeed = False
        for iteration in tqdm(range(max_step)):
            fail_cnt = 0
            optimized_latents = latents * (torch.exp(0.5 * log_var)) + mu
            
            # prepare scheduler
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            # loss records
            # joint_loss_list, cross_attn_loss_list, self_attn_loss_list = [], [], []
            
            # denoising loop
            for i, t in enumerate(timesteps):
                if i >= denoising_step_for_loss: break
                # Forward pass of denoising with text conditioning
                # NOTE 0920 이제부터 text condition embedding만 input으로 바뀌기 때문에 그에 따라 바꿔줌
                if use_grad_checkpoint:
                    noise_pred_text = checkpoint.checkpoint(self.unet, optimized_latents, t, text_embeddings, use_reentrant=False).sample
                else: noise_pred_text = self.unet(optimized_latents, t, encoder_hidden_states=text_embeddings).sample

                # joint_loss, cross_attn_loss, self_attn_loss = self.fn_compute_loss(
                #     indices=indices, K=1)
                opt_loss, succeeds  = self.fn_adaptive_optimization(indices=indices, opt_key=initno_opt_keys, mask=None, smooth_attentions=True,
                                                           attention_res=attention_resolution, K=K, logger=logger)
                # joint_loss_list.append(joint_loss), cross_attn_loss_list.append(cross_attn_loss), self_attn_loss_list.append(self_attn_loss)
                if denoising_step_for_loss > 1:
                    with torch.no_grad():
                        if use_grad_checkpoint:
                            noise_pred_uncond = checkpoint.checkpoint(self.unet, optimized_latents, t, text_embeddings[0].unsqueeze(0), use_reentrant=False).sample
                        else: noise_pred_uncond = self.unet(optimized_latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample

                    if do_classifier_free_guidance: noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    optimized_latents = self.scheduler.step(noise_pred, t, optimized_latents, **extra_step_kwargs).prev_sample
                
            # joint_loss      = sum(joint_loss_list) / denoising_step_for_loss
            # cross_attn_loss = max(cross_attn_loss_list)
            # self_attn_loss  = max(self_attn_loss_list)            
            # print loss records
            # joint_loss_list         = [_.item() for _ in joint_loss_list]
            # cross_attn_loss_list    = [_.item() for _ in cross_attn_loss_list]
            # self_attn_loss_list     = [_.item() for _ in self_attn_loss_list]            
            if len(succeeds) == 1:
                optimization_succeed = succeeds[0]
            else:
                for i, success in enumerate(succeeds):
                    if not success: fail_cnt += 1
                    if logger is not None:
                        logger.info(f'\t {initno_opt_keys[i]} loss success: {success}')                    
                # for succeed in succeeds:
                #     optimization_succeed *= succeed
            if fail_cnt < 1:
                optimization_succeed = True
                break
            else:
                optimization_succeed = False
  
            self.unet.zero_grad()
            optimizer.zero_grad()
            opt_loss = opt_loss.mean()
            opt_loss.backward()
            optimizer.step()

            # update kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            while kld_loss > 0.001:
                optimizer.zero_grad()
                kld_loss = kld_loss.mean()
                kld_loss.backward()
                optimizer.step()
                kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            
            torch.cuda.empty_cache()

        optimized_latents = (latents * (torch.exp(0.5 * log_var)) + mu).clone().detach()
        # if self_attn_loss <= 1e-6: self_attn_loss = self_attn_loss + 1. # really need?

        return optimized_latents, bool(optimization_succeed), opt_loss
        # return optimized_latents, optimization_succeed, cross_attn_loss + self_attn_loss 

    def register_attention_control(self, token_indices, prompts, tokenizer):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet,
                                                         token_indices=token_indices,
                                                         tokenizer=tokenizer,
                                                         prompts=prompts) #NOTE 09/23 token_indices added ; BA re impl

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alte"""
        ids = self.tokenizer(prompt).input_ids
        indices = {i: tok for tok, i in zip(self.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
        return indices

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        num_of_subjects: Union[List[int], List[List[int]]],
        prompt: Union[str, List[str]],
        token_indices: Union[List[int], List[List[int]]],
        cfg: Dict,                
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attn_res: Optional[Union[Tuple[int], List[Tuple[int]]]] = (16, 16),
        clip_skip: Optional[int] = None,                        
        logger: Optional[Any] = None,
        eos_token: Optional[List[str]] = None,
        save_path: Optional[Any] = None,
        seed: Optional[int] = None
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            token_indices (`List[int]`):
                The token indices to alter with attend-and-excite.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            max_iter_to_alter (`int`, *optional*, defaults to `25`):
                Number of denoising steps to apply attend-and-excite. The `max_iter_to_alter` denoising steps are when
                attend-and-excite is applied. For example, if `max_iter_to_alter` is `25` and there are a total of `30`
                denoising steps, the first `25` denoising steps applies attend-and-excite and the last `5` will not.
            thresholds (`dict`, *optional*, defaults to `{0: 0.05, 10: 0.5, 20: 0.8}`):
                Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in.
            scale_factor (`int`, *optional*, default to 20):
                Scale factor to control the step size of each attend-and-excite update.
            attn_res (`tuple`, *optional*, default computed from width and height):
                The 2D resolution of the semantic attention map.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        self.cross_attention_maps_cache = None

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            token_indices,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = cfg.guidance_scale > 1.0        

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )                

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control(token_indices, [prompt], self.tokenizer) #NOTE 09/23 new BA re impl

        # default config for step size from original repo
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = cfg.scale_factor * np.sqrt(scale_range)

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt :] if do_classifier_free_guidance else prompt_embeds
        )

        if isinstance(token_indices[0], int):
            token_indices = [token_indices]

        indices = []

        for ind in token_indices:
            indices = indices + [ind] * num_images_per_prompt

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - cfg.num_inference_steps * self.scheduler.order

        # 8. initno
        # run_initno = True # NOTE now initno flagged
        if cfg.run_initno:
            max_round = 5
            with torch.enable_grad():
                optimized_latents_pool = []
                for round in range(max_round):
                    optimized_latents, optimization_succeed, opt_loss = self.fn_initno(
                        latents=latents,
                        indices=token_indices[0],
                        text_embeddings=text_embeddings,
                        max_step=10,
                        num_inference_steps=cfg.num_inference_steps,
                        device=device,
                        guidance_scale=cfg.guidance_scale,
                        generator=generator,
                        eta=eta,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        round=round,
                        initno_opt_keys=cfg.initno.opt_keys,
                        K=cfg.initno.K,
                        # logger=logger
                    )
                    optimized_latents_pool.append((opt_loss, round, optimized_latents.clone(), latents.clone(), optimization_succeed))
                    if optimization_succeed: break
            
                    latents = self.prepare_latents(
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents=None,
                    )
                optimized_latents_pool.sort()
                
                if optimized_latents_pool[0][4] is True:
                    latents = optimized_latents_pool[0][2]                        
                else:                
                    optimized_latents, optimization_succeed, opt_loss = self.fn_initno(
                        latents=optimized_latents_pool[0][3],
                        indices=token_indices[0],
                        text_embeddings=text_embeddings,
                        max_step=50,
                        num_inference_steps=cfg.num_inference_steps,
                        device=device,
                        guidance_scale=cfg.guidance_scale,
                        generator=generator,
                        eta=eta,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        round=round,
                        initno_opt_keys=cfg.initno.opt_keys,
                        K=cfg.initno.K,
                        # logger=logger
                    ) 
                    latents = optimized_latents                
                
                torch.cuda.empty_cache()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(cfg.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # eos token for BA loss
        self.eos_token_index = eos_token
        self._determine_eos_token([prompt])
        # store attention map
        cross_attention_maps_inter_list, self_attention_maps_inter_list = [], []
        intermediate_image_list = [] #NOTE 11/21 중간 과정 결과 취득을 위해서
        mask_construct = {
            'cross': None,
            'self': None,
            'cluster': None
        }

        self.masks_per_token = None
        with self.progress_bar(total=cfg.num_inference_steps) as progress_bar:

            mask_anormaly = False
            for i, t in enumerate(timesteps):                
                # self.attention_store.mask = None #NOTE 09/23 guidance 할때 안할때 모두 fg from BA
                # adaptive optimization process
                # choose adaptive optimization key
                if not i > cfg.max_iter_to_alter:
                    if i < cfg.max_layout_iter:
                        opt_key = cfg.denoising.layout_opt_key
                    elif i < cfg.max_shape_iter:
                        opt_key = cfg.denoising.shape_opt_key
                    elif i < cfg.max_texture_iter:
                        opt_key = cfg.denoising.texture_opt_key
                    
                    with torch.enable_grad():
                        latents = latents.clone().detach().requires_grad_(True)
                        updated_latents = []
                        for latent, index, text_embedding in zip(latents, indices, text_embeddings):
                            # Forward pass of denoising with text conditioning
                            latent = latent.unsqueeze(0)
                            text_embedding = text_embedding.unsqueeze(0)

                            self.unet(
                                latent,
                                t,
                                encoder_hidden_states=text_embedding,
                                cross_attention_kwargs=cross_attention_kwargs,
                            ).sample
                            self.unet.zero_grad()

                            opt_loss, succeeds = self.fn_adaptive_optimization(indices=index, opt_key=opt_key, mask=self.masks_per_token,
                                                                     K=cfg.denoising.K, mask_loss_type=cfg.denoising.mask_loss_type,
                                                                     logger=logger)

                            # if cfg.visu_attn and i % cfg.attn_sampling_rate == 0:
                            #     # TODO should specify visualized attention resolutions                               
                            #     cross_attention_maps = self.attention_store.aggregate_attention(
                            #         from_where=("up", "down", "mid"), is_cross=True, res=(16, 16))
                            #     # self_attention_maps = self.attention_store.aggregate_attention(
                            #     #     from_where=("up", "down", "mid"), is_cross=False, res=(16, 16))
                            #     cross_attention_maps_inter_list.append(cross_attention_maps.clone().detach())     
                            #     # self_attention_maps_inter_list.append(self_attention_maps.clone().detach())

                            # If this is an iterative refinement step, verify we have reached the desired threshold for all
                            if i < cfg.max_iter_to_alter and cfg.denoising.iterative_refine and(i == 10 or i == 20):

                                opt_succeed = True
                                for succeed in succeeds:
                                    opt_succeed *= succeed
                                
                                if not succeed:
                                    opt_loss, succeeds = self._perform_iterative_refinement_step(sampling_step=i, latents=latent, indices=index,
                                                            opt_key=opt_key, text_embeddings=text_embedding, step_size=step_size[i], t=t, K=cfg.denoising.K)                            

                            # Perform gradient update
                            if i < cfg.max_iter_to_alter and not cfg.run_sd:
                                if opt_loss != 0 and True:
                                    latent = self._update_latent(
                                        latents=latent,
                                        loss=opt_loss,
                                        step_size=step_size[i],
                                    )
                            updated_latents.append(latent)

                    latents = torch.cat(updated_latents, dim=0)
                
                # get mask from latents
                # if not i < cfg.max_shape_iter - 1 and i < cfg.max_iter_to_alter:
                #NOTE 09/24 23:23 매 step마다 mask generate 하는 것이 좋은지 알 수 없기 때문에 일단 최초의 한번으로 하고 이후 확인할 것
                if i == cfg.max_shape_iter - 1:
                    #NOTE 1105 mask building시점 확인전에는 off
                    # self.unet(latents, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs,).sample                    
                    prompts = [prompt]

                    cross_attn_maps = self.attention_store.aggregate_attention(from_where=("up", "down"), is_cross=True, res=(16, 16)).detach()                                        
                    self_attn_maps = self.attention_store.aggregate_attention(from_where=("up", "down"), is_cross=False, res=(16, 16)).detach()

                    #NOTE 1127 paper visualization
                    if cfg.visu_mask_construct:
                        mask_construct['cross'] = cross_attn_maps
                        mask_construct['self'] = self_attn_maps

                    tokenized_prompt = nltk.word_tokenize(prompts[-1])
                    nouns =[(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt)) if pos[:2] == 'NN']
                    if logger is not None:
                        logger.info(f"{i}th sampling... nouns of {prompt}")
                        logger.info(f"\t {nouns}...")

                    self.masks_per_token, cluster_maps = self.get_mask(
                            obj_token_index=token_indices[0],
                            num_segments=cfg.mask.num_segments,                            
                            self_attn_maps=self_attn_maps, 
                            nouns=nouns, 
                            cross_attn_maps=cross_attn_maps
                        )
                    if cfg.visu_mask_construct:
                        mask_construct['cluster'] = cluster_maps
                    
                    
                    #NOTE 09/24 23:24 mask 평가 추가
                    cca_labels = []
                    for k, v in self.masks_per_token.items():
                        # evaluation mask
                        mask_np = v.astype(np.uint8)
                        cnt, _, _, _ = cv2.connectedComponentsWithStats(mask_np)
                        cca_labels.append(cnt) #NOTE as you know normal part should be 2 for each subjects mask
                        # os.makedirs(f'{save_path}/{seed}', exist_ok=True)
                        # cv2.imwrite(f'{save_path}/{seed}/{k}_{50 - i}th_steps_mask.jpg', v * 255)
                
                #NOTE 09/24 23:24 mask 기반 seed select 추가 anormal -> mask 내 군집이 2개이상, mask 내 군집이 0개 TODO nan case가 군집0개에 해당할 수 있는지 확인할것
                if self.masks_per_token is not None:
                    #NOTE multi numbers
                    for j, cca_label in enumerate(cca_labels):
                        if cca_label > num_of_subjects[j] + 1 or cca_label == 1:
                            mask_anormaly = True
                        # if anormal_mask > 2 or anormal_mask == 1: 
                        #     mask_anormaly = True
                
                if cfg.denoising.filtering and mask_anormaly: #NOTE 10/30 for ablation filtering추가
                    if logger is not None:
                        logger.info(f'anormaly {mask_anormaly}! ')
                    break

                # if cfg.denoising.fg and not i < cfg.max_shape_iter: 
                if cfg.denoising.fg and self.masks_per_token is not None: 
                    self.attention_store.mask = self.masks_per_token                    
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                if cfg.visu_denoising_process and i % cfg.attn_sampling_rate == 0:
                    alphas_comprod = self.scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
                    sqrt_alpha_prod = alphas_comprod[t] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(latents.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1) 

                    sqrt_one_minus_alpha_prod = (1 - alphas_comprod[t]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(latents.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                    # latents = sqrt_alpha_prod * denoised_at_t + sqrt_one_minus_alpha_prod * noise_pred
                    denoised_at_t = (latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
                    
                    denoised_image = self.vae.decode(denoised_at_t / self.vae.config.scaling_factor, return_dict=False)[0]
                    # denoised_image, has_nsfw_concept = self.run_safety_checker(denoised_image, device, prompt_embeds.dtype)
                    # if has_nsfw_concept is None:
                    do_denormalize = [True] * denoised_image.shape[0]                    
                    denoised_image = self.image_processor.postprocess(denoised_image, output_type=output_type, do_denormalize=do_denormalize)
                    intermediate_image_list.append(denoised_image)
                    del alphas_comprod, sqrt_alpha_prod, sqrt_one_minus_alpha_prod, denoised_at_t, denoised_image

                if cfg.visu_attn and i % cfg.attn_sampling_rate == 0:                    
                    cross_attention_maps = self.attention_store.aggregate_attention(
                        from_where=("up", "down", "mid"), is_cross=True, res=(16, 16))                    
                    cross_attention_maps_inter_list.append(cross_attention_maps.clone().detach())     
                    

            # show attention map at each timestep # NOTE 0916 deleted            

        # 8. Post-processing
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), {
            'masks': self.masks_per_token, 'cross': cross_attention_maps_inter_list,  'intermediate': intermediate_image_list,
            # 'nsfw': has_nsfw_concept[0],
            'mask_construct': mask_construct
        }, mask_anormaly
    
    def fn_adaptive_optimization(
        self, 
        indices: List[int], 
        opt_key: List[str],
        mask: Dict = None,
        smooth_attentions: bool = True,
        K: int = 1,
        attention_res: Tuple = (16, 16),
        mask_loss_type: str = 'lg',
        logger: Any = None,) -> torch.Tensor:

        success = []
        
        # get aggregated cross attention maps
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"), is_cross=True, res=attention_res)
        
        # cross attention map preprocessing
        cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:-1]
        cross_attention_maps = cross_attention_maps * 100
        cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]        
        # NOTE 0917 TODO loss dict가 아닌 다른형태로 받아서 opt_key에 있는 모든 loss의 가중합을 구현하자
        # NOTE 0917 그동안 mask ver에서 cross leak loss를 cross leak pair, 내경우 2로 나누지 않았음 기억할 것 !
        ane_loss, shape_loss, mask_loss = 0., 0., 0.

        if 'ane' in opt_key:
            loss_dict = self.compute_initno_loss(indices=indices, cross_attention_maps=cross_attention_maps, smooth_attentions=smooth_attentions,
                                 K=K, opt_key=opt_key, attention_res=attention_res)
            ane_loss = loss_dict['ane'] * 1.0 + loss_dict['cross_clean'] * 0.1 + loss_dict['cross_align'] * 0.1 + loss_dict['self_attn'] * 1.0
            ane_success = (loss_dict['ane'] < 0.2)
            self_attn_success = (loss_dict['self_attn'] < 0.3)
            if torch.is_tensor(ane_success): ane_success = ane_success.item()            
            if torch.is_tensor(self_attn_success): self_attn_success = self_attn_success.item()
            success.append(ane_success)
            success.append(self_attn_success)
            # succeed.append(ane_success)
            # succeed.append(self_attn_success)            
            if logger is not None:
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        logger.info(f'\t {k} loss: {v.item():0.4f}')
                    else:
                        logger.info(f'\t {k} loss: {v:0.4f}')
        
        if 'cross_shape' in opt_key or 'ab_cross_shape':
            if 'cross_shape' in opt_key:
                shape_loss = self.compute_cross_shape_loss(indices=indices, cross_attention_maps=cross_attention_maps, smooth_attentions=smooth_attentions,)
                shape_success = (shape_loss < 0.2)
                if torch.is_tensor(shape_success): shape_success = shape_success.item()
                success.append(shape_success)
                if logger is not None:
                    if torch.is_tensor(shape_loss):
                        logger.info(f'\t cross_shape loss {shape_loss.item():0.4f}')
                    else:
                        logger.info(f'\t cross_shape loss {shape_loss:0.4f}')
            if 'ab_cross_shape' in opt_key:
                loss_dict = self.compute_initno_loss(indices=indices, cross_attention_maps=cross_attention_maps, smooth_attentions=smooth_attentions,
                                 K=K, opt_key='self_attn', attention_res=attention_res)
                shape_loss = loss_dict['self_attn']
                shape_success = (loss_dict['self_attn'] < 0.3)
                if torch.is_tensor(shape_success): shape_success = shape_success.item()
                success.append(shape_success)
                if logger is not None:
                    if torch.is_tensor(shape_loss):
                        logger.info(f'\t ablation shape loss {shape_loss.item():0.4f}')
                    else:
                        logger.info(f'\t ablation shape loss {shape_loss:0.4f}')
            else:
                ValueError('Invalid opt_key?')
        
        if 'cross_mask' in opt_key:            
            #NOTE 09/23 ba re impl
            if mask_loss_type == 'ba':
                ba_cross_loss = self.compute_ba_mask_loss(attention_maps=cross_attention_maps, mask=mask, token_indices=indices, is_cross=True)
                b, s = ba_cross_loss.shape
                ba_cross_loss = ba_cross_loss.sum()
                normalized_ba_cross_loss = ba_cross_loss / b / s
                ba_cross_success = (normalized_ba_cross_loss < 0.2)

                self_attention_maps = self.attention_store.aggregate_attention(from_where=("up", "down", "mid"), is_cross=False, res=attention_res)
                ba_self_loss = self.compute_ba_mask_loss(attention_maps=self_attention_maps, mask=mask, token_indices=None, is_cross=False)
                b, s  = ba_self_loss.shape                
                ba_self_loss = ba_self_loss.sum()
                normalized_ba_self_loss = ba_self_loss / b / s
                ba_self_success = (normalized_ba_self_loss < 0.2)

                mask_loss = 1.0 * ba_cross_loss + 1.0 * ba_self_loss
                mask_success = (ba_cross_success.item() and ba_self_success.item())            
            
            else:
                #TODO should re-impl here
                mask_loss = self.compute_cross_mask_loss(indices=indices, cross_attention_maps=cross_attention_maps, smooth_attentions=smooth_attentions,
                                                     loss_type=mask_loss_type, mask=mask)
            
                mask_success = ((mask_loss < 0.2))
            
            if torch.is_tensor(mask_success): mask_success.item()
            success.append(mask_success) # TODO check layout guidance's threhold
            
            if logger is not None:
                if torch.is_tensor(mask_loss):
                    logger.info(f'mask loss {mask_loss.item():0.4f}')                    
                else:
                    logger.info(f'mask loss {mask_loss:0.4f}')
                if mask_loss_type == 'ba':
                    if torch.is_tensor(ba_cross_loss) and torch.is_tensor(ba_self_loss):
                        logger.info(f'cross mask {ba_cross_loss.item():0.4f} and self mask {ba_self_loss.item():0.4f}')
                    else:
                        logger.info(f'cross mask {ba_cross_loss:0.4f} and self mask {ba_self_loss:0.4f}')

        opt_loss = ane_loss * 1.0 + shape_loss * 1.0 + mask_loss * 1.0
        return opt_loss, success
    
    def compute_initno_loss(
        self,
        indices,
        cross_attention_maps,
        smooth_attentions,
        K,        
        opt_key,
        attention_res
    ):
        loss_dict = {}
        # Extract the maximum values
        topk_value_list, topk_coord_list_list = [], []
        # smoothed_target_attn_maps = []
        for i in indices:
            cross_attention_map_cur_token = cross_attention_maps[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            # smoothed_target_attn_maps.append(cross_attention_map_cur_token)
            topk_coord_list, _ = fn_get_topk(cross_attention_map_cur_token, K=K)

            topk_value = 0
            for coord_x, coord_y in topk_coord_list: topk_value = topk_value + cross_attention_map_cur_token[coord_x, coord_y]
            topk_value = topk_value / K

            topk_value_list.append(topk_value)
            topk_coord_list_list.append(topk_coord_list)

            clean_cross_attn_loss = 0
            if 'cross_clean' in opt_key:
                # -----------------------------------
                # clean cross_attention_map_cur_token
                # -----------------------------------
                clean_cross_attention_map_cur_token                     = cross_attention_map_cur_token
                clean_cross_attention_map_cur_token_mask                = fn_get_otsu_mask(clean_cross_attention_map_cur_token)
                clean_cross_attention_map_cur_token_mask                = fn_clean_mask(clean_cross_attention_map_cur_token_mask, topk_coord_list[0][0], topk_coord_list[0][1])
                clean_cross_attention_map_cur_token_foreground          = clean_cross_attention_map_cur_token * clean_cross_attention_map_cur_token_mask + (1 - clean_cross_attention_map_cur_token_mask)
                clean_cross_attention_map_cur_token_background          = clean_cross_attention_map_cur_token * (1 - clean_cross_attention_map_cur_token_mask)

                if clean_cross_attention_map_cur_token_background.max() > clean_cross_attention_map_cur_token_foreground.min():
                    clean_cross_attn_loss = clean_cross_attn_loss + clean_cross_attention_map_cur_token_background.max()
                else:
                    clean_cross_attn_loss = clean_cross_attn_loss + clean_cross_attention_map_cur_token_background.max() * 0
            loss_dict['cross_clean'] = clean_cross_attn_loss

        cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in topk_value_list]
        cross_attn_loss = max(cross_attn_loss_list)
        loss_dict['ane'] = cross_attn_loss

        cross_attn_alignment_loss = 0
        if 'cross_align' in opt_key:            
            alpha = 0.9
            if self.cross_attention_maps_cache is None: self.cross_attention_maps_cache = cross_attention_maps.detach().clone()
            else: self.cross_attention_maps_cache = self.cross_attention_maps_cache * alpha + cross_attention_maps.detach().clone() * (1 - alpha)
            
            for i in indices:
                cross_attention_map_cur_token = cross_attention_maps[:, :, i]
                if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
                cross_attention_map_cur_token_cache = self.cross_attention_maps_cache[:, :, i]
                if smooth_attentions: cross_attention_map_cur_token_cache = fn_smoothing_func(cross_attention_map_cur_token_cache)
                cross_attn_alignment_loss = cross_attn_alignment_loss + torch.nn.L1Loss()(cross_attention_map_cur_token, cross_attention_map_cur_token_cache)          
            
        loss_dict['cross_align'] = cross_attn_alignment_loss

        self_attn_loss = 0
        if 'self_attn' in opt_key:            
            self_attention_maps = self.attention_store.aggregate_attention(
                from_where=("up", "down", "mid"), is_cross=False, res=attention_res)
            
            attention_res = self_attention_maps.shape[0]

            self_attention_map_list = []
            for topk_coord_list in topk_coord_list_list:
                self_attention_map_cur_token_list = []
                for coord_x, coord_y in topk_coord_list:

                    self_attention_map_cur_token = self_attention_maps[coord_x, coord_y]
                    self_attention_map_cur_token = self_attention_map_cur_token.view(attention_res, attention_res).contiguous()
                    self_attention_map_cur_token_list.append(self_attention_map_cur_token)

                if len(self_attention_map_cur_token_list) > 0:
                    self_attention_map_cur_token = sum(self_attention_map_cur_token_list) / len(self_attention_map_cur_token_list)
                    if smooth_attentions: self_attention_map_cur_token = fn_smoothing_func(self_attention_map_cur_token)
                else:
                    self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
                    self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

                self_attention_map_list.append(self_attention_map_cur_token)

            self_attn_loss, number_self_attn_loss_pair = 0, 0
            number_token = len(self_attention_map_list)
            for i in range(number_token):
                for j in range(i + 1, number_token): 
                    number_self_attn_loss_pair = number_self_attn_loss_pair + 1
                    self_attention_map_1 = self_attention_map_list[i]
                    self_attention_map_2 = self_attention_map_list[j]

                    self_attention_map_min = torch.min(self_attention_map_1, self_attention_map_2) 
                    self_attention_map_sum = (self_attention_map_1 + self_attention_map_2)
                    cur_self_attn_loss = (self_attention_map_min.sum() / (self_attention_map_sum.sum() + 1e-6))
                    self_attn_loss = self_attn_loss + cur_self_attn_loss

            if number_self_attn_loss_pair > 0: self_attn_loss = self_attn_loss / number_self_attn_loss_pair

        loss_dict['self_attn'] = self_attn_loss
        
        return loss_dict
    
    def compute_cross_shape_loss(
        self,
        indices,
        cross_attention_maps,
        smooth_attentions,
        ):

        cross_leak_loss = 0.

        subject_cross_maps = []
        for i in indices:
            cross_attention_map_cur_token = cross_attention_maps[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            subject_cross_maps.append(cross_attention_map_cur_token)

        number_cross_attn_loss_pair = 0        
        number_token = len(subject_cross_maps)
        
        for i in range(number_token):
            for j in range(i+1, number_token):
                number_cross_attn_loss_pair = number_cross_attn_loss_pair + 1
                cross_attention_map_1 = subject_cross_maps[i]
                cross_attention_map_2 = subject_cross_maps[j]
                cross_attention_map_min = torch.min(cross_attention_map_1, cross_attention_map_2)
                cross_attnetion_map_1_sum = cross_attention_map_1.sum()
                cross_attnetion_map_2_sum = cross_attention_map_2.sum()
                cur_cross_leak_loss = (cross_attention_map_min.sum() / torch.min(cross_attnetion_map_1_sum, cross_attnetion_map_2_sum))
                cross_leak_loss = cross_leak_loss + cur_cross_leak_loss

        if number_cross_attn_loss_pair > 0 :
            cross_leak_loss = cross_leak_loss / number_cross_attn_loss_pair
        cross_leak_loss = cross_leak_loss * torch.ones(1).to(self._execution_device)

        return cross_leak_loss

    def compute_cross_mask_loss(
        self,
        indices,
        cross_attention_maps,
        smooth_attentions,        
        # loss_type, #NOTE 09/23 ba function 따로 impl, ba 부분 삭제
        mask):

        subject_cross_maps = []
        for i in indices:
            cross_attention_map_cur_token = cross_attention_maps[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            subject_cross_maps.append(cross_attention_map_cur_token)

        subj_loss = 0
        subject_number = len(indices)
        
        for subj_num in range(subject_number):
            subj_cross_maps = subject_cross_maps[subj_num] # 16 x 16            
            
            subj_mask = mask[indices[subj_num]]
            subj_mask = torch.from_numpy(subj_mask).float().to(subj_cross_maps.device) # maybe float doesn't needed
            
            
            activation_value = (subj_mask * subj_cross_maps).sum(dim=-1) / subj_cross_maps.sum(dim=-1) # 이름은 lg에서 따옴, TODO shoud check this part
            subj_loss += torch.mean((1 - activation_value) ** 2)        
            
        # subj_loss = subj_loss / subject_number # NOTE 0920 이거 원래 나누지 않고 합으로 했었음 # NOTE 0920 21:07 한번 안해보고 써보자        
        return subj_loss
    
    # ref: lpm official code
    def cluster_mine(self, self_attn_maps, num_segments):
        ''' clustering self attention map with 32x32 / 16x16 resolution '''
        np.random.seed(1)
        resolution = self_attn_maps.shape[0]
        attn = self_attn_maps.cpu().numpy().reshape(resolution ** 2, resolution ** 2)
        kmeans = KMeans(n_clusters=num_segments, n_init=10).fit(attn)
        clusters = kmeans.labels_
        clusters = clusters.reshape(resolution, resolution)
        return clusters

    def cluster2noun_mine(self,
        clusters, nouns, cross_attn_maps, num_segments, background_segment_threshold=0.3):        
        result = {}
        nouns_indices = [index for (index, word) in nouns] # 입력 prompt에서 명사의 list
        nouns_maps = cross_attn_maps.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]] # 찾은 명사 list 를 이용해 cross attention map에 접근
        # normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(2, axis=0).repeat(2, axis=1) # 32x32 zero maps
        normalized_nouns_maps = np.zeros_like(nouns_maps) # for 16x16 res
        for i in range(nouns_maps.shape[-1]):
            # curr_noun_map = nouns_maps[:, :, i].repeat(2, axis=0).repeat(2, axis=1)
            curr_noun_map = nouns_maps[:, :, i]
            normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
        for c in range(num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))] # 명사별 cross maps을 클러스터 레이블별 마스크와 곱해서 score maps를 구한다
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps] #  score map이 cluster map에 부합되는 정도를 나타냄
            result[c] = nouns[np.argmax(np.array(scores))] if max(scores) > background_segment_threshold else "BG"
            # 클러스터 레이블별 가장 scores가 높은 noun을 구함
        return result

    def get_mask(self, obj_token_index, num_segments, self_attn_maps, nouns, cross_attn_maps):
        
        masks_per_tokens = {}
        background_nouns = []
        obj_token_index = [i - 1 for i in obj_token_index]        

        for idx, noun in nouns:
            if idx in obj_token_index:
                background_nouns.append((idx, noun))

        clusters = self.cluster_mine(self_attn_maps=self_attn_maps, num_segments=num_segments)
        # cluster labeling된 32x32 map kinda seg-map
        cluster2noun = self.cluster2noun_mine(clusters=clusters, nouns=nouns, cross_attn_maps=cross_attn_maps,
                     num_segments=num_segments, background_segment_threshold=0.3) 
        # cluster2noun = {cluster label number : 'BG' or (token index, 'target noun')}
        
        # token 별 mask
        for single_obj_idx in obj_token_index:
            mask = clusters.copy()            
            obj_segments = [c for c in cluster2noun if cluster2noun[c][0] == single_obj_idx]
            background_segments = [c for c in cluster2noun if cluster2noun[c] != single_obj_idx]
            for c in range(num_segments):
                if c in background_segments and c not in obj_segments:
                    mask[clusters == c] = 0
                else:
                    mask[clusters == c] = 1
            
            # print(f'hello there {np.unique(mask)}')
            masks_per_tokens[single_obj_idx] = mask
        
        #NOTE 1127 visu mask construct를 위해 cluster return에 추가
        return masks_per_tokens, clusters
        
    
    #NOTE 09/23 BA loss re-implementation
    def compute_ba_mask_loss(
        self,
        attention_maps, #NOTE should be b x head dim x i x l
        mask,
        token_indices,
        is_cross,
        ):
        '''
        BA attention maps = b x head x i x l
        mine case = h x w x l ; because of aggregation
        lets reshaps ours h x w x l -> b, i=hw, l
        '''
        #TODO should check batch size cfg of not cfg form while guidance !!
        # re-shape attention maps as BA's form
        a_h, a_w, l = attention_maps.shape
        attention_maps = attention_maps.reshape(a_h * a_w, l).unsqueeze(0) #NOTE should be b x i x l

        mean_foreground_values, mean_background_values = [], []
        # reshape mask as BA's form
        subject_masks = []
        for key, mask in mask.items():
            mask = torch.from_numpy(mask).bool().reshape(-1).to(attention_maps.device)
            subject_masks.append(mask)
        subject_masks = torch.stack(subject_masks, dim=0) #NOTE should be #S x i
        subject_numbers = subject_masks.shape[0]
        bg_mask = subject_masks.sum(dim=0) == 0
        bg_mask = bg_mask.to(attention_maps.device) #NOTE should be i

        if is_cross:
            #NOTE 09/24 eos term추가, official code에 있어서, 하지만 지워야 할 수도 있으니 기억해둘 것 ! TODO eos term 제거 버전도 사용해볼것     
            token_indices = [[token_idx] + [self.eos_token_index] for token_idx in token_indices]       
            contexts = token_indices
        else:
            contexts = subject_masks

        for i, (masks, context) in enumerate(zip(subject_masks, contexts)):
            context_attn = attention_maps[ :, :, context] #NOTE should be b x i x #S
            # context_attn[:, masks] b x #M x #S
            mean_foreground_values.append(context_attn[:, masks].sum(dim=(1, 2)))
            mean_background_values.append(context_attn[:, bg_mask].sum(dim=(1, 2)))
        
        mean_foreground_values = torch.stack(mean_foreground_values, dim=1) #NOTE should be b x #S
        mean_background_values = torch.stack(mean_background_values, dim=1) #NOTE should be b x #S

        iou = mean_foreground_values / (mean_foreground_values + mean_background_values * subject_numbers)
        
        ba_mask_loss = (1 - iou) ** 2 #NOTE should be b x #S

        return ba_mask_loss        
        
    #NOTE 09/24 BA reimpl
    def _tokenize(self, prompts):
        ids = self.tokenizer.encode(prompts[0])
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        return [token[:-4] for token in tokens]  # remove ending </w>
    
    def _determine_eos_token(self, prompts):
        tokens = self._tokenize(prompts)
        eos_token_index = len(tokens) + 1
        if self.eos_token_index is None:
            self.eos_token_index = eos_token_index
        elif eos_token_index != self.eos_token_index:
            raise ValueError(f'Wrong EOS token index. Tokens are: {tokens}.')

