import os
import torch
import datetime
import time
import logging
from omegaconf import OmegaConf
from torchvision.utils import save_image
from utils.ane_dataset import AnEPrompt
from pipelines.pipeline_initno_plus_mask_reprod import SDInitNoPlusMaskV001
from visu_utils.attention_visualization_utils import show_cross_attention, show_self_attention

def get_indices(prompt, logger):
    indices = []
    if ' and ' in prompt:
        prompt_parts = prompt.split(' and ')
    elif ' with ' in prompt:
        prompt_parts = prompt.split(' with ')
    else:        
        logger.info(f"Unable to split prompt: {prompt}. "
                f"Looking for 'and' or 'with' for splitting! Skipping!")        
    
    for idx, p in enumerate(prompt.split(' ')):
        if p == prompt_parts[0].split(' ')[-1]:
            indices.append(idx+1)
        elif p == prompt_parts[1].split(' ')[-1]:
            indices.append(idx+1)
    
    return indices

def get_logger(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(path, f"{__name__}.log"))
    file_handler.setLevel(logging.INFO)
    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger



def main(cfg):
    # get config
    cfg = OmegaConf.load(cfg)
    # get exp name
    now = datetime.datetime.now()
    now = now.strftime("%Y%m%dT%H%M%S")    

    # load diffusers pipeline
    pipe = SDInitNoPlusMaskV001.from_pretrained(cfg.model_id).to("cuda")

    # output dir
    result_root = os.path.join(cfg.result_root, now)    
    os.makedirs('{:s}'.format(result_root), exist_ok=True)

    # get logger
    logger = get_logger(result_root)

    exp_start = time.time()

    # prompt & seed number
    PROMPTS = AnEPrompt()
    SEEDS = [x for x in range(64)]

    # save current config
    OmegaConf.save(cfg, f'{result_root}/config.yaml')    
    
    for prompt in PROMPTS(cfg.subset_key):
        prompt_dir = os.path.join(result_root, prompt.replace(" ", "_"))
        os.makedirs(prompt_dir, exist_ok=True)        
        logger.info(f"Prompt: {prompt}")        
        token_indices = get_indices(prompt, logger)
        logger.info(f"token indices:{token_indices}")            
        
        for SEED in SEEDS:
            if SEED > 10:
                break
            logger.info('Seed ({}) Processing the ({}) prompt'.format(SEED, prompt))

            gen_start = time.time()
            generator = torch.Generator("cuda").manual_seed(SEED)         
        
            images, attention_dict = pipe(
                        prompt=prompt, # masking을 위한 generation
                        token_indices=token_indices,                
                        generator=generator,                                    
                        seed=SEED,                
                        logger=logger,
                        cfg=cfg,                        
                        segmentation_self_attn=True
                    )        
        
            images = images.images
            image = images[0]

            image.save(f"{prompt_dir}/initno_{token_indices[0]}n{token_indices[1]}_{SEED}.jpg")
            logger.info(f"\t saved at {prompt_dir}/initno_{token_indices[0]}n{token_indices[1]}_{SEED}.jpg")
            gen_end = time.time()
            logger.info(f"\t Elapsed {gen_end - gen_start:0.4f} sec")

            visu_prompt=["<|startoftext|>",] + prompt.split(' ') + ["<|endoftext|>",] 
            os.makedirs(f'{prompt_dir}/{SEED}/crossattn', exist_ok=True)

            self_visu_prompt = []
            for idx in token_indices:
                self_visu_prompt.append(prompt.split(' ')[idx-1])
            os.makedirs(f'{prompt_dir}/{SEED}/selfattn', exist_ok=True)

            # get self attn corresponding max value's coordination

            if cfg.tmp_results:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                os.makedirs(f'{prompt_dir}/{SEED}/intermediates', exist_ok=True)
                for idx, inter_image in enumerate(attention_dict['inter']):
                    inter_image[0].save(f'{prompt_dir}/{SEED}/intermediates/{idx}th_inter.jpg')

            for sampling_steps, cross_attn_map in enumerate(attention_dict['cross']):                                
                cross_attn_img, _ = show_cross_attention(prompts=visu_prompt, attention_maps=cross_attn_map)
                cross_attn_img.save(f"{prompt_dir}/{SEED}/crossattn/{sampling_steps}th_attn.png")    
            
            # NOTE 0909 OOM issue로 잠시 주석처리
            # for sampling_steps, self_attn_map in enumerate(attention_dict['self']):                                
            #     self_attn_img = show_self_attention(self_attention_maps=self_attn_map, prompts=self_visu_prompt)
            #     self_attn_img.save(f"{prompt_dir}/{SEED}/selfattn/{sampling_steps}th_attn.png")

            for k, v in attention_dict['masks'].items():
                # logger.info(f'before {np.unique(v)}')
                # logger.info(f'after {torch.unique(torch.from_numpy(v).float())}')
                save_image(torch.from_numpy(v).float(), f'{prompt_dir}/{SEED}/{k}_mask.jpg')

    exp_end = time.time()
    elapse = exp_end - exp_start
    hours = elapse // 3600
    min = (elapse % 3600) // 60
    sec = (elapse % 3600) % 60
    logger.info(f'\t exp finished {hours}h {min}min {sec}sec')

if __name__ == '__main__':
    # cfg_path = '/home/initno1/GSN/initno/conf/conf_initno_plus_v001_mask.yaml'
    cfg_path = '/home/work/sh22/plus/conf/conf_initno_plus_mask_reprod.yaml'
    main(cfg_path)