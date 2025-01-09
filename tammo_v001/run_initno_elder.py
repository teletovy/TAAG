import os
import torch
import time
from omegaconf import OmegaConf

from utils.ane_dataset import AnEPrompt
from pipelines.pipeline_initno import StableDiffusionInitNOPipeline
from visu_utils.attention_visualization_utils import show_cross_attention
import cv2
from utils.code_utils import copy_src_files, get_logger, get_indices, get_time_str, get_iou_of_mask
# from visu_utils.attention_visualization_utils import show_cross_attention, show_self_attention

def main(cfg):
    # get config
    cfg = OmegaConf.load(cfg)
    # get exp name
    save_path = os.path.join(cfg.result_root, get_time_str())
    copy_src_files(
        os.path.dirname(os.path.abspath(__file__)), # current dir
        os.path.join(save_path, 'src')
        )

    # load diffusers pipeline
    pipe = StableDiffusionInitNOPipeline.from_pretrained(cfg.model_id).to("cuda")    

    # get logger
    logger = get_logger(save_path)
    exp_start = time.time()

    # prompt & seed number
    PROMPTS = AnEPrompt()
    SEEDS = [x for x in range(cfg.init_seed_num, cfg.init_seed_num + cfg.max_seed_numbers)]

    # save current config
    OmegaConf.save(cfg, os.path.join(save_path, 'config.yaml'))    
    
    for prompt in PROMPTS(cfg.subset_key):
        prompt_dir = os.path.join(save_path, prompt.replace(" ", "_"))
        os.makedirs(prompt_dir, exist_ok=True)
        logger.info(f"Prompt: {prompt}")
        token_indices = get_indices(prompt, logger)
        logger.info(f"token indices:{token_indices}")
        image_cnt = 0

        for SEED in SEEDS:
            if image_cnt == cfg.image_nums:
                break            
            
            logger.info('Seed ({}) Processing the ({}) prompt'.format(SEED, prompt))            
            
            generator = torch.Generator("cuda").manual_seed(SEED)
            gen_start = time.time()
            
            images, results_dict = pipe(
                prompt=prompt,
                token_indices=token_indices,                
                generator=generator,
                # result_root=result_root,
                seed=SEED,                
                logger=logger,
                cfg=cfg
            )

            images = images.images
            image = images[0]
            image_cnt += 1

            image.save(f"{prompt_dir}/{token_indices[0]}n{token_indices[1]}_seed{SEED}.jpg")
            logger.info(f"\t saved at {prompt_dir}/{token_indices[0]}n{token_indices[1]}_seed{SEED}.jpg")
            gen_end = time.time()
            logger.info(f"\t Generation one images elapsed {gen_end - gen_start:0.4f} sec")

            visu_prompt=["<|startoftext|>",] + prompt.split(' ') + ["<|endoftext|>",] 
            os.makedirs(f'{prompt_dir}/{SEED}/crossattn', exist_ok=True)            

            if cfg.visu_attn:
                for sampling_steps, cross_attn_map in enumerate(results_dict['cross']):                                
                    cross_attn_img, _ = show_cross_attention(prompts=visu_prompt, attention_maps=cross_attn_map)
                    cross_attn_img.save(f"{prompt_dir}/{SEED}/crossattn/{sampling_steps}th_attn.png")                
            
            for k, v in results_dict['masks'].items():
                # evaluation mask                
                cv2.imwrite(f'{prompt_dir}/{SEED}/{k}_mask.jpg', v * 255)

    exp_end = time.time()
    logger.info(f'\t exp finished {exp_end - exp_start:0.4f} sec')
    

if __name__ == '__main__':
    cfg_path = '/home/initno1/GSN/initno_plus_v002/conf/conf_initno.yaml'
    main(cfg_path)