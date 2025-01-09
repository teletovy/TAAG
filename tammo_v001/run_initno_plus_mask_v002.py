'''
main frame work가 될 code - 0918
name 변경 plus -> seed_selection - 0920
'''
import os
import torch
# import datetime
import time
# import logging
from omegaconf import OmegaConf
# from torchvision.utils import save_image

from utils.ane_dataset import AnEPrompt
from pipelines.pipeline_initno_plus_mask_v002 import SDPlusV002
from visu_utils.attention_visualization_utils import show_cross_attention
import cv2
from utils.code_utils import copy_src_files, get_logger, get_indices, get_time_str, get_iou_of_mask

def main(cfg):
    # get config
    cfg = OmegaConf.load(cfg)

    save_path = os.path.join(cfg.result_root, get_time_str())
    copy_src_files(
        os.path.dirname(os.path.abspath(__file__)), # current dir
        os.path.join(save_path, 'src')
        )

    # load diffusers pipeline
    pipe = SDPlusV002.from_pretrained(cfg.model_id).to("cuda")    

    # get logger
    logger = get_logger(save_path)
    logger.info(f'Run {__file__} file')
    exp_start = time.time()
    # prompt & seed number
    PROMPTS = AnEPrompt()
    # save current config
    OmegaConf.save(cfg, os.path.join(save_path, 'config.yaml'))
    SEEDS = [x for x in range(cfg.max_seed_numbers)]

    for prompt in PROMPTS(cfg.subset_key):        
        prompt_dir = os.path.join(save_path, prompt.replace(" ", "_"))
        os.makedirs(prompt_dir, exist_ok=True)        
        logger.info(f"Prompt: {prompt}")        
        token_indices = get_indices(prompt, logger)
        logger.info(f"token indices:{token_indices}")            
        
        image_cnt = 0 #NOTE 09/24 23:28 mask기반 seed selection?
        
        for SEED in SEEDS:
            if image_cnt == cfg.image_nums:
                break

            gen_start = time.time()
            generator = torch.Generator("cuda").manual_seed(SEED)       
            
            images, results_dict, mask_anormaly = pipe(
                        prompt=prompt,
                        token_indices=token_indices,                                        
                        logger=logger,
                        cfg=cfg,
                        generator=generator,
                        save_path=prompt_dir,
                        seed=SEED
                        # start_seed_num=start_seed_number
                    )            
            if mask_anormaly:                
                pass

            else:
                mask_dict = results_dict['masks']

                images = images.images
                image = images[0]
                image_cnt += 1

                image.save(f"{prompt_dir}/{token_indices[0]}n{token_indices[1]}_seed{SEED}.jpg")
                logger.info(f"\t saved at {prompt_dir}/{token_indices[0]}n{token_indices[1]}_seed{SEED}.jpg")
                gen_end = time.time()
                logger.info(f"\t Generation one images elapsed {gen_end - gen_start:0.4f} sec")

                visu_prompt=["<|startoftext|>",] + prompt.split(' ') + ["<|endoftext|>",] 
                os.makedirs(f'{prompt_dir}/{SEED}/crossattn', exist_ok=True)

                # self_visu_prompt = []
                # for idx in token_indices:
                #     self_visu_prompt.append(prompt.split(' ')[idx-1])
                # os.makedirs(f'{prompt_dir}/{seed_number}/selfattn', exist_ok=True)
                # get self attn corresponding max value's coordination

                #NOTE 10/30 deprecated
                # if cfg.tmp_results:
                #     os.makedirs(f'{prompt_dir}/{SEED}/intermediates', exist_ok=True)
                #     for idx, inter_image in enumerate(results_dict['inter']):
                #         inter_image[0].save(f'{prompt_dir}/{SEED}/intermediates/{idx}th_inter.jpg')

                if cfg.visu_attn:
                    for sampling_steps, cross_attn_map in enumerate(results_dict['cross']):                                
                        cross_attn_img, _ = show_cross_attention(prompts=visu_prompt, attention_maps=cross_attn_map)
                        cross_attn_img.save(f"{prompt_dir}/{SEED}/crossattn/{sampling_steps}th_attn.png")                
                
                for i in range(len(token_indices)):
                    for j in range(i+1, len(token_indices)):
                        mask_iou = get_iou_of_mask(mask_dict[token_indices[i] - 1], mask_dict[token_indices[j] - 1])
                
                for k, v in results_dict['masks'].items():
                    # evaluation mask                
                    cv2.imwrite(f'{prompt_dir}/{SEED}/{k}_{mask_iou:0.3f}_mask.jpg', v * 255)

    exp_end = time.time()
    elapse = exp_end - exp_start
    hours = elapse // 3600
    min = (elapse % 3600) // 60
    sec = (elapse % 3600) % 60
    logger.info(f'\t exp finished {hours}h {min}min {sec}sec')


if __name__ == '__main__':
    cfg_path = '/home/initno1/GSN/mino_v002/conf/conf_mino_v002.yaml'
    main(cfg_path)
