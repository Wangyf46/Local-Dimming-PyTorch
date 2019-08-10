import cv2

from lib.localdimming.transform import *


## TODO, only YUV, RGB or Iin is not


def getIcp_linear(Iin, LD):
    B, G, R = cv2.split(Iin)
    Y, U, V = rgbToyuv(R, G, B)
    # LD_max = np.max(LD)
    LD_max = 255.0


    K = np.where(LD==0.0, 0.0, LD_max / LD)
    Y1 = K * Y
    U1 = K * (U - 128) + 128
    V1 = K * (V - 128) + 128
    R1, G1, B1 = yuvTorgb(Y1, U1, V1)
    Icp = cv2.merge([B1, G1, R1])
    return Icp


def getIcp_unlinear(Iin, LD):
    B, G, R = cv2.split(Iin)
    Y, U, V = rgbToyuv(R, G, B)
    # LD_max = np.max(LD)
    LD_max = 255.0
    r = 2.2
    K = np.where(LD==0.0, 0.0, (LD_max / LD) ** (1 / r))
    Y1 = K * Y
    U1 = K * (U - 128) + 128
    V1 = K * (V - 128) + 128
    R1, G1, B1 = yuvTorgb(Y1, U1, V1)
    Icp = cv2.merge([B1, G1, R1])
    return Icp


def getIcp_2steps(Iin, LD):
    Iin_max = np.max(Iin)
    B, G, R = cv2.split(Iin)
    Y, U, V = rgbToyuv(R, G, B)
    a = 0.005
    LD_max = 255.0
    gamma = 2.2

    ## improve cr
    Y1 = Iin_max / (1 + np.exp(a * (LD - Y)))
    Y1 = np.where(Y1 > 255.0, 255.0, Y1)

    ## enhance diaplay quality
    K2 = (LD / LD_max) ** (1.0/gamma)
    Y2 = Y1 * np.log10(1 + Y * K2)

    K = np.where(Y1!=0.0, Y2/Y1, Y2/(Y1+0.00001))

    U2 = K * (U - 128) + 128
    V2 = K * (V - 128) + 128
    R1, G1, B1 = yuvTorgb(Y2, U2, V2)
    Icp = cv2.merge([B1, G1, R1])
    return Icp



def getIcp_log(Iin, LD):
    B, G, R = cv2.split(Iin)
    Y, U, V = rgbToyuv(R, G, B)
    M, N = LD.shape
    # LD_max = np.max(LD)
    LD_max = 255.0
    gamma = 2.2

    K = np.where(LD!=0, (LD_max / LD) ** (1/gamma), 0)
    K1 = np.log2(1 + K)

    Y1 = K1 * Y
    U1 = K1 * (U - 128) + 128
    V1 = K1 * (V - 128) + 128
    R1, G1, B1 = yuvTorgb(Y1, U1, V1)
    Icp = cv2.merge([B1, G1, R1])
    return Icp



'''
    # zeros = np.zeros(img_float.shape[:2], dtype="uint8")  # 创建与image相同大小的零矩阵
    # cv2.imshow("BLUE", cv2.merge([np.uint8(B), zeros, zeros]))  # 显示 （B，0，0）图像
    # cv2.imshow("GREEN",  cv2.merge([zeros, np.uint8(G), zeros]))  # 显示（0，G，0）图像
    # cv2.imshow("RED",  cv2.merge([zeros, zeros, np.uint8(R)]))  # 显示（0，0，R）图像
'''