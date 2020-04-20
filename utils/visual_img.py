import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_src_img_label(img_path, label_path):
    x = cv2.imread(img_path)
    y = np.loadtxt(label_path)
    print("y: ", y)
    if len(y.shape) == 1:
        cv2.rectangle(x, (int(y[1]), int(y[2])), (int(y[3]) + int(y[1]), int(y[4]) + int(y[2])), (0, 0, 255), 3)
    else:
        for i in range(len(y)):
            cv2.rectangle(x, (int(y[i, 1]), int(y[i, 2])), (int(y[i, 3]) + int(y[i, 1]), int(y[i, 4]) + int(y[i, 2])), (0, 0, 255), 3)
            # print(int(y[i, 1]), int(y[i, 2]), int(y[i, 3]) + int(y[i, 1]), int(y[i, 4]) + int(y[i, 2]))
    print("x.shape: ", x.shape)
    print("y: ", y)
    plt.imshow(x)
    plt.show()