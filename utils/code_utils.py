import time
from os import path as osp
import shutil
import time
import os
import logging
import numpy as np

def make_dir(path, is_archive=False):
    """mkdirs. If path exists, rename it with timestamp and create a new one.
    Args:
        path (str): Folder path.
    """
    if osp.exists(path) and is_archive:
        path = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {path}', flush=True)        
        os.makedirs(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
    return path

def copytree(src, dst, symlinks=False):        
    ignore = shutil.ignore_patterns('*.pyc', 'tmp*', '.caches', '__pycache__', '*.pth', '*.gz', '*.npy', '*.pb', '*.pt', '*.yaml') #NOTE 09/24 yaml 저장 항목에서 제외
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore=ignore)
        else:
            shutil.copy2(s, d)


def copy_src_files(sorce_src_path, save_src_path):
    # save_src_path = os.path.join(save_dir, 'src')
    if os.path.isdir(save_src_path):
        save_src_path = make_dir(save_src_path, is_archive=True)
    else:
        save_src_path = make_dir(save_src_path, is_archive=False)
    copytree(sorce_src_path, save_src_path)
    print("Copy script files at {}!!".format(save_src_path))


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

#NOTE 1223 혼동을 위해서, MO inidicing을 위해 추가됨
#NOTE prompt_name은 ours를 위해 추가된 부분
def get_indices(prompt, logger, prompt_name=None):
    indices = []    
    number_of_subject = []

    if ' and ' in prompt:
        prompt_parts = prompt.split(' and ')

    elif ' with ' in prompt:
        prompt_parts = prompt.split(' with ')

    elif ' next to ' in prompt and not ' standing ' in prompt:
        prompt_parts = prompt.split(' next to ')

    elif ' on ' in prompt:
        prompt_parts = prompt.split(' on ')

    elif ' standing ' in prompt and not ' next to ' in prompt:
        prompt_parts = [prompt.split(' standing ')[0]]

    elif ' standing next to ' in prompt:
        prompt_parts = [prompt.split(' standing next to ')[0]]        

    elif ' floating ' in prompt:
        prompt_parts = [prompt.split(' floating ')[0]]

    else:        
        logger.info(f"Unable to split prompt: {prompt}. "
                f"Looking for 'and' or 'with' for splitting! Skipping!")  

    if len(prompt_parts) == 2:
        #NOTE 1223
        if prompt_name == 'multi_objects':
            for prompt_part in prompt_parts:
                if prompt_part.split(' ')[0] == 'one':
                    number_of_subject.append(1)

                elif prompt_part.split(' ')[0] == 'two':
                    number_of_subject.append(2)

                elif prompt_part.split(' ')[0] == 'three':
                    number_of_subject.append(3)

                elif prompt_part.split(' ')[0] == 'a' or prompt_part.split(' ')[0] == 'the':
                    number_of_subject.append(1)
        else:
            number_of_subject = [1, 1]
    
    elif len(prompt_parts) == 1:
        #NOTE 1223
            if prompt_name == 'multi_objects':
                if prompt_parts[0].split(' ')[0] == 'one':
                    number_of_subject.append(1)

                elif prompt_parts[0].split(' ')[0] == 'two':
                    number_of_subject.append(2)

                elif prompt_parts[0].split(' ')[0] == 'three':
                    number_of_subject.append(3)
            else:
                number_of_subject = [1]
    else:
        ValueError('Invalid prompt type. You should check type of prompt datatset !.')
    
    for idx, p in enumerate(prompt.split(' ')):        
        if len(prompt_parts) == 2:
            if p == prompt_parts[0].split(' ')[-1]:
                indices.append(idx+1)
            elif p == prompt_parts[1].split(' ')[-1]:
                indices.append(idx+1)

        elif len(prompt_parts) == 1:
            if p == prompt_parts[0].split(' ')[-1]:
                indices.append(idx + 1)
        else:
            ValueError('Invalid prompt type')
    
    
    assert len(indices) == len(number_of_subject)    
    return indices, number_of_subject


#NOTE 1109 att mask binding에 포함시키는게 논리에 맞아서
def get_texture_indices(prompt, logger, subset):
    indices = []    
    with_flag = False
    texture_indices = []

    if ' and ' in prompt:
        prompt_parts = prompt.split(' and ')
        
    elif ' with ' in prompt:
        prompt_parts = prompt.split(' with ')
        with_flag = True
    else:        
        logger.info(f"Unable to split prompt: {prompt}. "
                f"Looking for 'and' or 'with' for splitting! Skipping!")        
    
    for idx, p in enumerate(prompt.split(' ')):
        if p == prompt_parts[0].split(' ')[-1]: # 앞단이다.
            indices.append(idx+1)
            if subset == 'objects':
                texture_indices.append([idx - 1, idx]) #NOTE 이하도 같은 이유, sot제거후 인덱스가 필요해서 미리 처리함                

            elif subset == 'animals_objects':                
                texture_indices.append([idx])

            else:
                ValueError(f'invalid subset {subset}')

        elif p == prompt_parts[1].split(' ')[-1]:
            indices.append(idx+1)
            if subset == 'objects':
                texture_indices.append([idx - 1, idx])                

            elif subset == 'animals_objects':
                if with_flag:
                    texture_indices.append([idx])
                else:
                    texture_indices.append([idx - 1, idx])  
            else:
                ValueError(f'invalid subset {subset}') 
    
    return indices, texture_indices

def get_logger(path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(path, "main_log.log")) #NOTE 09/24 log filename 변경
    file_handler.setLevel(logging.INFO)
    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger

def hard_mask_to_bboxes(mask, logger):    
    ''''''        
    X, Y = mask.shape    
    x_coord, y_coord = [], []
    for x in range(X):
        mask_colummn = mask[x]
        for y in range(Y):
            check_value = mask_colummn[y]
            if check_value > 0.5:
                x_coord.append(x)
                y_coord.append(y)

    x_min, y_min, x_max, y_max = min(x_coord), min(y_coord), max(x_coord), max(y_coord)
    # W:H -> W/H    
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    return [x_min, y_min, x_max, y_max], aspect_ratio

def get_iou_of_mask(mask_A, mask_B):
    intersection = np.logical_and(mask_A, mask_B)
    union = np.logical_or(mask_A, mask_B)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score

def untitle(mask, logger):
    logger.info(f'binary mask? {np.unique(mask)}')
    mask = mask.astype(np.uint8)
    bboxes, aspect_ratio = hard_mask_to_bboxes(mask)
    # mask_iou = get_iou_of_mask()



if __name__ == '__main__':
    import numpy as np
    test_score_maps = np.ones_like((16, 16, 5))
    for test_score_map in test_score_maps:
        print(test_score_map.shape)
    from dnb_prompt import DnBPrompt
    PROMPTS = DnBPrompt()
    subset_key= 'multi_objects'
    for prompt in PROMPTS(subset_key):      
        token_indices, num_of_subject = get_indices(prompt=prompt, logger=None, prompt_name=subset_key)
        print(prompt)
        print(token_indices)
        print(num_of_subject)
