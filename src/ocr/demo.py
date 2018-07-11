#-*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import ocr
import time
import numpy as np
from PIL import Image
from glob import glob
import os
import codecs
import sys

if __name__ == '__main__':

    # TODO: you should change the path
    if sys.argv[1] == 'test':
        name = 'test'
    else:
        name = 'train'
    image_files = glob('../../data/' + name + '_images/*.*')
    with codecs.open('../../data/result_' + name + '_ocr.txt', 'w', encoding='utf-8') as f:
        t = time.time()
        for image_file in sorted(image_files):
            filename = os.path.split(image_file)[1]
            image = np.array(Image.open(image_file).convert('RGB'))
            result, image_framed = ocr.model(image)
            text = filename
            for key in result:
                text += '\t'
                text += result[key][1]
            text += '\n'
            print(text)
            f.write(text)
    print('------> total cost', time.time() - t, '------->images', len(image_files))
    print('done')



