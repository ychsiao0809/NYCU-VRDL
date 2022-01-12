import os
from PIL import Image
import argparse

def resize(image_path):
    dirs = os.listdir(image_path)
    for file in dirs:
        img = Image.open(os.path.join(image_path, file))
        (w, h) = img.size
        lower_edge = min(w, h)
        if lower_edge < 96:
            print("Remove image %s (Size: %dx%d)" % (file, w, h))
            os.remove(os.path.join(image_path, file))

def main(args):
    image_path =args.input_dir
    resize(image_path)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='data/training_hr_images', help="input images path")
    args = parser.parse_args()
    main(args)
