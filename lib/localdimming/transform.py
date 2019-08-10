import numpy as np


def getLD_transform(LD):
    H, W = LD.shape
    LD_transform = np.zeros([H, W], dtype='float32')
    for i in range(H):
        for j in range(W):
            if (LD[i][j] >= 0.0) and (LD[i][j] < 8.0):
                LD_transform[i][j] = 0.4625 * LD[i][j] + 0.3
            else:
                LD_transform[i][j] = 1.1984 * LD[i][j] - 5.592
    return LD_transform


def rgbToyuv(R, G, B):
    R = np.where(R>255.0, 255.0, R)
    R = np.where(R<.0, 0.0, R)
    G = np.where(G>255.0, 255.0, G)
    G = np.where(G<0.0, 0.0, G)
    B = np.where(B>255.0, 255.0, B)
    B = np.where(B<0.0, 0.0, B)

    Y = 0.2989 * R + 0.5866 * G + 0.1145 * B
    U = -0.1688 * R - 0.3312 * G + 0.5 * B + 128
    V = 0.5 * R - 0.4184 * G - 0.0816 * B + 128

    Y = np.where(Y>255.0, 255.0, Y)
    Y = np.where(Y<0.0, 0.0, Y)
    U = np.where(U>255.0, 255.0, U)
    U = np.where(U<0.0, 0.0, U)
    V = np.where(V>255.0, 255.0, V)
    V = np.where(V<0.0, 0.0, V)

    return Y, U, V


def yuvTorgb(Y, U, V):
    Y = np.where(Y>255.0, 255.0, Y)
    Y = np.where(Y<0.0, 0.0, Y)
    U = np.where(U>255.0, 255.0, U)
    U = np.where(U<0.0, 0.0, U)
    V = np.where(V>255.0, 255.0, V)
    V = np.where(V<0.0, 0.0, V)

    R = Y + 1.4021 * (V - 128)
    G = Y - 0.3456 * (U - 128)- 0.7145 * (V - 128)
    B = Y + 1.771 * (U - 128)

    R = np.where(R > 255.0, 255.0, R)
    R = np.where(R < .0, 0.0, R)
    G = np.where(G > 255.0, 255.0, G)
    G = np.where(G < 0.0, 0.0, G)
    B = np.where(B > 255.0, 255.0, B)
    B = np.where(B < 0.0, 0.0, B)

    return R, G, B


def rgbToxyz(R, G, B):
    X = 0.4124 * R + 0.3576 * G + 0.1805 * B
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Z = 0.0193 * R + 0.1192 * G + 0.9505 * B
    return X, Y, Z


def xyzTolab(X, Y, Z):
    v1 = 1.0/3
    v2 = (6.0/29) ** 3
    v3 = (29.0/6) ** 2
    v4 = 4.0/29
    Xn = 95.047
    Yn = 100.0
    Zn = 108.883

    X1 = Y / Yn
    Y1 = X / Xn
    Z1 = Z / Zn

    Xf = np.where(X1 > v2, X1 ** v1, v1 * v3 * X1 + v4)
    Yf = np.where(Y1 > v2, Y1 ** v1, v1 * v3 * Y1 + v4)
    Zf = np.where(Z1 > v2, Z1 ** v1, v1 * v3 * Z1 + v4)

    L = 116 * Yf - 16
    a = 500 * (Xf - Yf)
    b = 200 * (Yf - Zf)

    return L, a, b


def Gammas(R, G, B):
    ## normalization [0.0, 1.0]
    R1 = np.where(R > 0.04045, ((R + 0.055) / 1.055) ** 2.4, R / 12.92)
    G1 = np.where(G > 0.04045, ((G + 0.055) / 1.055) ** 2.4, G / 12.92)
    B1 = np.where(B > 0.04045, ((B + 0.055) / 1.055) ** 2.4, B / 12.92)
    return R1, G1, B1