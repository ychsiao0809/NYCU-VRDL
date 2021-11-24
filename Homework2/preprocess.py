import h5py
import cv2
import os
from shutil import copyfile
from random import sample
import argparse


def get_name(index, hdf5_data):
    name_ref = hdf5_data['/digitStruct/name'][index].item()
    return ''.join([chr(v[0]) for v in hdf5_data[name_ref]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item_ref = hdf5_data['/digitStruct/bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item_ref][key]
        values = [hdf5_data[attr[i].item()][0][0].astype(int)
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values
    return attrs

def main(args):
    os.chdir('./datasets')

    with h5py.File('train/digitStruct.mat') as hdf5_data:
        max = hdf5_data['digitStruct/name'].shape[0]
        valid_idx = sample(range(max), args.valid)
        for i in range(max):
            img_name = get_name(i, hdf5_data)
            img_file = 'train/' + img_name
            
            if not os.path.exists(img_file):
                print("FileNotFound:", img_file)
                exit(1)
            
            im = cv2.imread(img_file)
            h, w, _ = im.shape
            bbox = get_bbox(i, hdf5_data)            

            if i in valid_idx:
                data_path = 'shvn/valid/'
            else:
                data_path += 'shvn/train/'

            # Store label
            fp = open(data_path+'labels/'+img_name.replace('.png','.txt'), 'w')
            arr_l = len(bbox['label'])
            for idx in range(arr_l):
                label = bbox['label'][idx]
                if label==10:
                    label = 0
                _l = bbox['left'][idx]
                _t = bbox['top'][idx]
                _w = bbox['width'][idx]
                if (_l+_w)>w:
                    _w = w-_l-1
                _h = bbox['height'][idx]
                if (_t+_h)>h:
                    _h = h-_t-1
                x_center = (_l + _w/2)/w
                y_center = (_t + _h/2)/h
                bbox_width = _w/w
                bbox_height = _h/h
                # print(label, x_center, y_center, bbox_width, bbox_height)
                s = str(label)+ ' '+str(x_center)+' '+str(y_center)+' '+ \
                    str(bbox_width)+' '+str(bbox_height)+'\n'
                # if idx!=(arr_l-1):
                #     s += '\n'
                fp.write(s)
            # Store image
            copyfile('train/'+img_name, data_path+'images/'+img_name)
            fp.close()

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--valid', type=int, help='number of validation data', default=3000)
    return parser
       
if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()

    main(args)