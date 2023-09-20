import torch
import cv2
from utils import *


if __name__ == "__main__":
    model = torch.load('mymodel.pt')

    # img = np.array(Image.open('img_train_330.jpg'))
    # img = img.transpose((2, 0, 1))
    # img = img[None, :]
    # img = torch.from_numpy(img)
    # img = img.float() / 255

    img2 = cv2.imread('img_train_156.jpg')
    img2 = img2.transpose((2, 0, 1))
    img2 = img2[None, :]
    img2 = torch.from_numpy(img2)
    img2 = img2.float() / 255


    with torch.no_grad():
        # out = model(img)
        out2 = model(img2)

    # out = non_max_suppression(out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
    out2 = non_max_suppression(out2, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
    # print(out)
    print(out2)
    