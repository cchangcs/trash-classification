# encoding:utf-8
'''
用于对预测结果图片进行显示
'''
import cv2
import numpy as np

# 根据预测结果显示对应的文字label
classes_types = ['cardboard', 'glass', 'trash']


def generate_result(result):
    for i in range(3):
        if(result[0][i] == 1):
            print(i)
            return classes_types[i]


def show(img_path, results):
    # 对结果进行显示
    frame = cv2.imread(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, generate_result(results), (10, 140), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('img', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()