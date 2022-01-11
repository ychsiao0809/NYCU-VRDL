import pandas as pd
import os
import pydicom
from tqdm import tqdm
from PIL import Image
import argparse

def resize(image_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    dirs = os.listdir(image_path)
    for file in dirs:
        img = Image.open(os.path.join(image_path, file))
        (w, h) = img.size
        
        # print("resize (%d, %d) to (%d, %d)" % (w, h, w/3, h/3))
        # continue
        
        # new_img = img.resize((int(w/3), int(h/3)))
        new_img = img.resize((48, 48))
        new_img.save(os.path.join(output_path, file))
    # for image_name in tqdm(images):
    #     dcm_file = os.path.join(dcm_path, f'{image_name}.dcm')
    #     dcm_data = pydicom.read_file(dcm_file)
    #     im = Image.fromarray(dcm_data.pixel_array)
    #     im.save(os.path.join(output_path, f'{image_name}.png'))

def main(args):
    image_path =args.input_dir
    output_path = args.output_dir

    resize(image_path, output_path)
    

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='datasets/training_hr_images/', help="input images path")
    parser.add_argument('--output-dir', type=str, default='datasets/training_lr_images/', help='output images path')
    args = parser.parse_args()
    main(args)