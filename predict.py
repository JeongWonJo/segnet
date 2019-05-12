import numpy as np
import keras
from PIL import Image

from model import SegNet

import dataset
import cv2
import numpy as np
import os
import argparse

height = 360
width = 480
classes = 12
epochs = 100
batch_size = 1
log_filepath='./logs_100/'
data_shape = 360*480


# directory option
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, help='File path to the style image')
    parser.add_argument('--style_dir', type=str, help='Directory path to multiple images')
    parser.add_argument('--output', type=str, default='output', help='Directory to save the output image(s)')
    args = parser.parse_args()
    return args

def writeImage(image, filename):
    """ label data to colored image """
    Sky = [128,128,128]
    Building = [128,64,128]
    Pole = [128,64,128]
    Road_marking = [128,64,128]
    Road = [128,64,128]
    Pavement = [128,64,128]
    Tree = [128,64,128]
    SignSymbol = [128,64,128]
    Fence = [128,64,128]
    Car = [128,64,128]
    Pedestrian = [128,64,128]
    Bicyclist = [128,64,128]
    Unlabelled = [128,64,128]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)



if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if "/" in args.output:
        output = args.output
    else:
        output = args.output + "/"

    print("loading data...")
    ds = dataset.Dataset(test_file='val.txt', classes=classes)
    test_X = ds.load_data_test('test') # need to implement, y shape is (None, 360, 480, classes)
    test_X = ds.preprocess_inputs(test_X)

    model = keras.models.load_model('seg.h5')
    probs = model.predict(test_X, batch_size=10)
    
    # save segmented image
    index=len(probs)
    for prob in probs:
        prob = prob.reshape((height, width, classes)).argmax(axis=2)
        i = len(probs) - index
        writeImage(prob, output+'val' + str(i) + '.png')
        index-=1

    # change sky
    content_paths = [f for f in os.listdir(output)]
    content_paths.sort()
    index = 0
    for content in content_paths:
        img = cv2.imread(output+content)
        upstate_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # these are the gray color mask using the RGB channels
        hsv_color1 = np.asarray([0, 0, 0])
        hsv_color2 = np.asarray([ 128,   128, 128])

        # get mask of pixels that are in gray range
        mask = cv2.inRange(upstate_hls, hsv_color1, hsv_color2)

        # detect gray part in the picture
        # Normalize the mask to keep intensity between 0 and 1
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_rgb= mask_rgb.astype(float)/255
        
        #content_paths = [f for f in os.listdir(output)]
        original_path = open('val.txt')
        original_path = original_path.readlines()
        original_path = original_path[index].strip()
        print(original_path)
        if args.style_dir:
            style_dir = [args.style_dir+f for f in os.listdir(args.style_dir)]
        if args.style:
            style_dir = [args.style]
        
        for style in style_dir:
            background = cv2.imread(style)
            original = cv2.imread(original_path)
            background = background.astype(float)
            original = original.astype(float)
            
            original = cv2.multiply(1.0-mask_rgb,original)
            background = cv2.multiply(mask_rgb, background)
            dst= cv2.add(original,background)
            
            if '/' in style:
                output_style = style.split('/')[-1]
            else:
                output_style = style
            cv2.imwrite(output+output_style.split(".")[0]+'_'+str(index)+'.png', dst)
        index+=1

