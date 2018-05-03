# encoding:utf-8
'''
调用各个模块实现对图片的测试和显示结果
'''
from plot import show
import predict
from prepare_image import prepare_image


def detect(img_path):
    model = predict.model()
    results = prepare_image(img_path=img_path, model=model)
    show(img_path=img_path, results=results)

if __name__ == '__main__':

    detect("test/15.jpg")
