import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import numpy as np
import pyrallis
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

# sys.path.append(".")
# sys.path.append("..")

from imagenet_utils import get_embedding_for_prompt, imagenet_templates
import os

@dataclass
class EvalConfig:
    output_path: Path = Path("./outputs/")
    metrics_save_path: Path = Path("./metrics/")

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)


# @pyrallis.wrap()
def run(config, logger):
    
    logger.info("Loading CLIP model...")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    logger.info("Done.")

    logger.info("Loading BLIP model...")
    blip_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco",
                                                              is_eval=True, device=device)
    logger.info("Done.")

    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    logger.info(f"Running on {len(prompts)} prompts...")

    results_per_prompt = {}
    for prompt in tqdm(prompts):
        if prompt.split('_')[0] == 'src':
            pass
        else:
            logger.info(f'Running on: "{prompt}"')

            # get all images for the given prompt
            image_paths = []
            for p in (config.output_path / prompt).rglob('*'):
                if p.suffix in ['.png', '.jpg']:
                    if not os.path.splitext(p)[0].endswith('attn') and not os.path.splitext(p)[0].endswith('mask'):
                        image_paths.append(p)
                    
            # if len(image_paths) > 10 and len(image_paths) != 64:
            #     max_seed_num = 0
            #     max_path_idx = 0
            #     for idx_path, path in enumerate(image_paths):
            #         current_seed_num = int(path.name.split('.')[0].split('d')[-1])
            #         if current_seed_num > max_seed_num:
            #             max_seed_num = current_seed_num
            #             max_path_idx = idx_path
            #     image_paths.pop(max_path_idx)                    

            # image_paths = [p for p in (config.output_path / prompt).rglob('*') if p.suffix in ['.png', '.jpg']]
            images = [Image.open(p) for p in image_paths]
            image_names = [p.name for p in image_paths]

            if len(image_names) != 0:
                with torch.no_grad():
                    # extract prompt embeddings
                    prompt = prompt.replace("_", " ")
                    prompt_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)

                    # extract blip captions and embeddings
                    blip_input_images = [vis_processors["eval"](image).unsqueeze(0).to(device) for image in images]
                    blip_captions = [blip_model.generate({"image": image})[0] for image in blip_input_images]
                    texts = [clip.tokenize([text]).cuda() for text in blip_captions]
                    caption_embeddings = [model.encode_text(t) for t in texts]
                    caption_embeddings = [embedding / embedding.norm(dim=-1, keepdim=True) for embedding in caption_embeddings]

                    text_similarities = [(caption_embedding.float() @ prompt_features.T).item()
                                        for caption_embedding in caption_embeddings]

                    results_per_prompt[prompt] = {
                        'text_similarities': text_similarities,
                        'captions': blip_captions,
                        'image_names': image_names,
                    }

    # aggregate results
    total_average, total_std = aggregate_text_similarities(results_per_prompt)
    aggregated_results = {
        'average_similarity': total_average,
        'std_similarity': total_std,
    }

    with open(config.metrics_save_path / "blip_raw_metrics.json", 'w') as f:
        json.dump(results_per_prompt, f, sort_keys=True, indent=4)
    with open(config.metrics_save_path / "blip_aggregated_metrics.json", 'w') as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)


def aggregate_text_similarities(result_dict):
    all_averages = [result_dict[prompt]['text_similarities'] for prompt in result_dict]
    all_averages = np.array(all_averages).flatten()
    total_average = np.average(all_averages)
    total_std = np.std(all_averages)
    return total_average, total_std

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


if __name__ == '__main__':
    import time
    import logging

    start = time.time()
    output_path = '/home/initno1/exp/dnb/20241221_022049'
    metrics_save_path = f'{output_path}_eval_blip_tt_sim/'

    config = EvalConfig(
        output_path=Path(output_path),
        metrics_save_path=Path(metrics_save_path)
    )

    logger = get_logger(metrics_save_path)

    logger.info(f'run on {output_path}')
    run(config=config, logger=logger)

    end = time.time()
    logger.info(f'elapsed {end-start:0.4f} sec')
