
import numpy as np

import cv2

from PIL import Image
# img1 = cv2.imread('pred_a10_40.png')
# img2 = cv2.imread('gt_a10_40.png')

img1 = Image.open('pred_a10_40.png')
img2 = Image.open('gt_a10_40.png')







import cv2

def get_contours(img):

    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]
def main(gt, pred):
    # 1.导入图片
    img1 = cv2.imread(gt)#'gt_a10_38.png'
    img2 = cv2.imread(pred)#'pred_a10_38.png'
    # img3 = cv2.imread("a10_1.png")
    # 2.获取图片连通域
    cnt1 = get_contours(img1)
    cnt2 = get_contours(img2)
    # cnt3 = get_contours(img3)
    # 3.创建计算距离对象
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    # 4.计算轮廓之间的距离
    d1 = hausdorff_sd.computeDistance(cnt1, cnt1)
    print("与自身的距离hausdorff\t d1=", d1)
    d2 = hausdorff_sd.computeDistance(cnt1, cnt2)
    print("与5图片的距离hausdorff\t d2=", d2)
    return d2
    # d3 = hausdorff_sd.computeDistance(cnt1, cnt3)
    # print("与6图片的距离hausdorff\t d3=", d3)
# if __name__ == '__main__':
#     main()


from os.path import join

def hsd_main(gt_dir, pred_dir, png_name_list):
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    hsd = 0
    for ind in range(len(gt_imgs)):
        print(gt_imgs[ind], pred_imgs[ind])
        hsd += main(gt_imgs[ind], pred_imgs[ind])

    mhsd = hsd / len(gt_imgs)
    return mhsd


gt_dir = 'D:\paper_plus\paper_conclusion\\gt_10'
pred_dir = r'D:\paper_plus\paper_conclusion\DeepLabV3+\2013'
# png_name_list = 'D:\paper_plus\paper_conclusion\OUR\\test.txt'
import os
image_ids = open(os.path.join('D:\paper_plus\paper_conclusion\\name_list_13', "test_10_val_v2.txt"),'r').read().splitlines()

print(hsd_main(gt_dir, pred_dir, image_ids))