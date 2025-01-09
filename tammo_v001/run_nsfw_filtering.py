import os
import torch
import time
from omegaconf import OmegaConf

from visu_utils.attention_visualization_utils import show_cross_attention, view_images
import cv2
from utils.code_utils import copy_src_files, get_logger, get_indices, get_time_str, get_iou_of_mask
import numpy as np


def main(cfg):
    # get config
    cfg = OmegaConf.load(cfg)    
    copy_src_files(
        os.path.dirname(os.path.abspath(__file__)), # current dir
        os.path.join(cfg.sample_dir, 'src_nsfw')
        )

    # load diffusers pipeline
    #NOTE 1123 pipeline 선택 부분 추가
    if cfg.pipe_name == 'ours':
        from pipelines.pipeline_mino_v002 import SDMINOv002
        pipe = SDMINOv002.from_pretrained(cfg.model_id).to("cuda")
    
    elif cfg.pipe_name == 'initno':
        from pipelines.pipeline_initno import StableDiffusionInitNOPipeline
        pipe = StableDiffusionInitNOPipeline.from_pretrained(cfg.model_id).to("cuda")

    # get logger
    logger = get_logger(os.path.join(cfg.sample_dir, 'src_nsfw'))
    logger.info(f'Run {__file__} file')    
    
    # save current config
    OmegaConf.save(cfg, os.path.join(os.path.join(cfg.sample_dir, 'src_nsfw'), 'config.yaml'))    
    
    for prompt in os.listdir(cfg.sample_dir):
        max_seed_number = 0
        nsfw_list = []
        if os.path.isdir(os.path.join(cfg.sample_dir,prompt)):
            for img_name in os.listdir(os.path.join(cfg.sample_dir,prompt)):
                if img_name.endswith('.jpg') and not img_name.split('.')[0].endswith('attn'):
                    seed_number = int(img_name.split('_seed')[-1].split('.')[0])
                    if seed_number > max_seed_number:
                        max_seed_number = seed_number
                    image = cv2.imread(os.path.join(cfg.sample_dir,prompt, img_name))
                    if np.mean(image) == 0.0:
                        # print(prompt, img_name)
                        nsfw_list.append(img_name)            
            # print(prompt, max_seed_number)
            # print(nsfw_list)
            logger.info(f'{prompt}, max seed {max_seed_number}')
            for nsfw in nsfw_list:
                logger.info(nsfw)
            if len(nsfw_list) != 0:
                # start_seed_number 부터
                prompt_dir = os.path.join(os.path.join(cfg.sample_dir, prompt)) 
                prompt = prompt.replace('_', ' ')
                logger.info(f'Prompt {prompt}')
                # filtering nsfw list
                token_indices = get_indices(prompt, logger)
                logger.info(f"token indices:{token_indices}")          

                if cfg.pipe_name == 'ours':
                    seeds = [x + max_seed_number + 1 for x in range(cfg.max_seed_numbers)]   
                elif cfg.pipe_name == 'initno':
                    seeds = [x + cfg.start_seed for x in range(cfg.max_seed_numbers)]

                gen_cnt = 0             
                for index, seed in enumerate(seeds):
                    if gen_cnt == len(nsfw_list):
                        break                    
                    logger.info(f"starts with {seed}")
                    generator = torch.Generator("cuda").manual_seed(seed)
                    gen_start = time.time()                   

                    if cfg.pipe_name == 'ours':
                        images, results_dict, mask_anormaly = pipe(
                            prompt=prompt,
                            token_indices=token_indices,                    
                            logger=logger,
                            cfg=cfg,
                            generator=generator,
                            save_path=prompt_dir,
                            seed=seed)
                        
                    elif cfg.pipe_name == 'initno':
                        images, results_dict = pipe(
                            prompt=prompt,
                            token_indices=token_indices,                    
                            logger=logger,
                            cfg=cfg,
                            generator=generator,                        
                            seed=seed)
                        mask_anormaly = False
                        
                    
                    if (cfg.denoising.filtering and mask_anormaly) or results_dict["nsfw"]:              
                        nsfw_result = results_dict["nsfw"]
                        logger.info(f"nsfw {nsfw_result}")  
                        logger.info(f"mask anormaly {mask_anormaly}")
                        pass

                    else:
                        mask_dict = results_dict['masks']
                        images = images.images
                        image = images[0]
                        

                        image.save(f"{prompt_dir}/{cfg.pipe_name}_{token_indices[0]}n{token_indices[1]}_nsfw_seed{seed}.jpg")
                        gen_cnt += 1
                        logger.info(f"\t saved at {prompt_dir}/{cfg.pipe_name}_{token_indices[0]}n{token_indices[1]}_nsfw_seed{seed}.jpg")
                        gen_end = time.time()
                        logger.info(f"\t Generation one images elapsed {gen_end - gen_start:0.4f} sec")

                        visu_prompt=["<|startoftext|>",] + prompt.split(' ') + ["<|endoftext|>",] 
                        os.makedirs(f'{prompt_dir}/{seed}/crossattn', exist_ok=True)       

                        # NOTE 10/30 deprecated
                        if cfg.visu_denoising_process:
                            os.makedirs(f'{prompt_dir}/{seed}/intermediates', exist_ok=True)
                            for idx, inter_image in enumerate(results_dict['intermediate']):
                                inter_image[0].save(f'{prompt_dir}/{seed}/intermediates/{idx * cfg.attn_sampling_rate}th_inter.jpg')

                        if cfg.visu_attn:
                            for sampling_steps, cross_attn_map in enumerate(results_dict['cross']):                                
                                cross_attn_img, splited_image = show_cross_attention(prompts=visu_prompt, attention_maps=cross_attn_map)
                                cross_attn_img.save(f"{prompt_dir}/{seed}/crossattn/{sampling_steps * cfg.attn_sampling_rate}th_attn.png")
                                for subject_index in token_indices:
                                    splited_pil_img = view_images(splited_image[subject_index], display_image=False)
                                    splited_pil_img.save(f"{prompt_dir}/{seed}/crossattn/{sampling_steps * cfg.attn_sampling_rate}th_idx{subject_index}_attn.png")
                  
                    logger.info(f"\t saved at {prompt_dir}/{token_indices[0]}n{token_indices[1]}_seed{seed}.jpg")
                    gen_end = time.time()
                    logger.info(f"\t Generation one images elapsed {gen_end - gen_start:0.4f} sec")

                for remove in nsfw_list:
                    if os.path.exists(os.path.join(prompt_dir, remove)):
                        logger.info(f'remove {os.path.join(prompt_dir, remove)}')
                        os.remove(os.path.join(prompt_dir, remove))
                


    


if __name__ == '__main__':
    cfg_path = '/home/initno1/GSN/mino_v002/conf/conf_initno_nsfw.yaml'
    main(cfg_path)
