# encoding:utf-8
'''
对需要进行预测的图片进行预处理
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np


def prepare_image(img_path, model):
    # 加载图像
    img = load_img(img_path, target_size=(512, 384))        # x = np.array(img, dtype='float32')test
    # 图像预处理
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    results = model.predict(x)
    print(results)
    return results
