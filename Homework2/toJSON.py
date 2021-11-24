import os
import cv2
import json
import argparse

def main(args):
    
    filepath = 'yolov5/runs/detect/%s/' % args.exp
    ans = []

    filenames = os.listdir(filepath+'labels/')

    for filename in sorted(filenames, key=lambda x: int(x[:-4])):
        image_id = filename.replace('.txt', '')

        f = open(filepath+'labels/'+filename, 'r')

        image_file = filepath+image_id+'.png' 
        if not os.path.exists(image_file):
            print("FileNotFound:", image_file)
            exit(1)
        im = cv2.imread(image_file)
        h, w, c = im.shape
        print("Image size: %dx%d" % (h,w))

        for line in f.readlines():
            s = line.strip().split(' ') # class, x_center, y_center, width, height, score
            cls = int(s[0])
            x_c = w * float(s[1])
            y_c = h * float(s[2])
            scr = float(s[5])

            # calculate bbox
            width  = w * float(s[3])
            height = h * float(s[4])
            top    = y_c - height/2
            left   = x_c - width/2

            dict = {
                "image_id": int(image_id),
                "category_id": cls,
                "bbox": [left, top, width, height],
                "score": scr,
            }
            ans.append(dict)
        f.close()
    ret = json.dump(ans, open('answer.json', 'w'), indent=4)


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='exp', help='name of experiment directory')
    return parser


if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()

    main(args)