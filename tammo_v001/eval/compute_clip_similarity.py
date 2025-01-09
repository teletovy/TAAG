import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import numpy as np
# import pyrallis
import torch
from PIL import Image
from tqdm import tqdm
import os
# sys.path.append(".")
# sys.path.append("..")

from imagenet_utils import get_embedding_for_prompt, imagenet_templates
import logging
import time

@dataclass
class EvalConfig:
    # def __init__(self, output_path=None, metrics_save_path=None):
    #     self.output_path = output_path
    #     self.metrics_save_path = metrics_save_path

    output_path: Path = Path("./outputs/")
    metrics_save_path: Path = Path("./metrics/")

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)


# @pyrallis.wrap()
# def run(config: EvalConfig):
def run(config, logger):
    # print("Loading CLIP model...")
    logger.info("Loading CLIP model...")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    # print("Done.")
    logger.info("Done.")

    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    logger.info(f"Running on {len(prompts)} prompts...")
    

    results_per_prompt = {}
    for prompt in tqdm(prompts):

        logger.info(f'Running on: "{prompt}"')

        # get all images for the given prompt
        # image_paths = [p for p in (config.output_path / prompt).rglob('*') if p.suffix in ['.png', '.jpg']]
        image_paths = []
        for p in (config.output_path / prompt).rglob('*'):
            if p.suffix in ['.png', '.jpg']:
                if not os.path.splitext(p)[0].endswith('attn') and not os.path.splitext(p)[0].endswith('mask'):
                    image_paths.append(p)       
        
        
        images = [Image.open(p) for p in image_paths]        
        image_names = [p.name for p in image_paths]
        queries = [preprocess(image).unsqueeze(0).to(device) for image in images]

        logger.info(f'prompt {prompt} img file length {len(image_names)}')

        if len(image_names) != 0:
            with torch.no_grad():

                prompt = prompt.replace('_', ' ')

                # split prompt into first and second halves
                if ' and ' in prompt:
                    prompt_parts = prompt.split(' and ')
                elif ' with ' in prompt:
                    prompt_parts = prompt.split(' with ')
                else:
                    logger.info(f"Unable to split prompt: {prompt}. "
                        f"Looking for 'and' or 'with' for splitting! Skipping!")
                    continue

                # extract texture features
                full_text_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)
                first_half_features = get_embedding_for_prompt(model, prompt_parts[0], templates=imagenet_templates)
                second_half_features = get_embedding_for_prompt(model, prompt_parts[1], templates=imagenet_templates)

                # extract image features
                images_features = [model.encode_image(image) for image in queries]
                images_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in images_features]

                # compute similarities
                full_text_similarities = [(feat.float() @ full_text_features.T).item() for feat in images_features]
                first_half_similarities = [(feat.float() @ first_half_features.T).item() for feat in images_features]
                second_half_similarities = [(feat.float() @ second_half_features.T).item() for feat in images_features]

                results_per_prompt[prompt] = {
                    'full_text': full_text_similarities,
                    'first_half': first_half_similarities,
                    'second_half': second_half_similarities,
                    'image_names': image_names,
                }    
        
    for prompt, prompt_dict in results_per_prompt.items():
        if len(prompt_dict['full_text']) != 10:
            print('prompt: {} full text len {} '.format(prompt, len(prompt_dict['full_text'])))
        if len(prompt_dict['first_half']) != 10:
            print('prompt: {} first half len {} '.format(prompt, len(prompt_dict['first_half'])))
        if len(prompt_dict['second_half']) != 10:
            print('prompt: {} second half len {} '.format(prompt, len(prompt_dict['second_half'])))
        if len(prompt_dict['image_names']) != 10:
            print('prompt: {} image names len {} '.format(prompt, len(prompt_dict['image_names'])))
        

    
    # aggregate results
    aggregated_results = {
        'full_text_aggregation': aggregate_by_full_text(results_per_prompt),
        'min_first_second_aggregation': aggregate_by_min_half(results_per_prompt),
    }

    with open(config.metrics_save_path / "clip_raw_metrics.json", 'w') as f:
        json.dump(results_per_prompt, f, sort_keys=True, indent=4)
    with open(config.metrics_save_path / "clip_aggregated_metrics.json", 'w') as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)


def aggregate_by_min_half(d):
    """ Aggregate results for the minimum similarity score for each prompt. """
    min_per_half_res = [[min(a, b) for a, b in zip(d[prompt]["first_half"], d[prompt]["second_half"])] for prompt in d]
    min_per_half_res = np.array(min_per_half_res).flatten()
    return np.average(min_per_half_res)


def aggregate_by_full_text(d):
    """ Aggregate results for the full text similarity for each prompt. """
    full_text_res = [v['full_text'] for v in d.values()]
    full_text_res = np.array(full_text_res).flatten()
    return np.average(full_text_res)


if __name__ == '__main__':

    start = time.time()
    output_path = '/home/initno1/exp/dnb/20241221_022049'
    metrics_save_path = f'{output_path}_eval_ti-sim/'
    
    config = EvalConfig(
        output_path=Path(output_path),
        metrics_save_path=Path(metrics_save_path)
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(metrics_save_path, f"{__name__}.log"))
    file_handler.setLevel(logging.INFO)
    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.info(f'compute from {output_path} ..')
    logger.info(f'saved {metrics_save_path} ..')

    run(config=config, logger=logger)

    end = time.time()
    print(f'elapsed {end-start:0.4f} sec')
