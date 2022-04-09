import numpy as np
from scipy.integrate import solve_ivp
from skimage.measure import shannon_entropy

def convert_to_binary(img):
    lst = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lst.append(np.binary_repr(img[i][j] ,width=8))
    lst = np.array(lst)
    return lst.reshape((img.shape[0], img.shape[1]))

def logistic_map(height, width, key_value, r):
    size = height * width
    r = r
    logistic_map = []
    x0 = key_value
    for i in range(size):
        temp = r * x0 * (1 - x0)
        x0 = temp
        logistic_map.append(temp)
    logistic_map = np.array(logistic_map)*(10**14)
    logistic_image = (np.floor(logistic_map)%256).astype("uint8")
    bin_img = [np.binary_repr(i, width = 8) for i in logistic_image]
    return np.array(bin_img)

def linear_shift_register(seed, height, width):
    size = height * width
    seed = seed
    lfsr = []
    lfsr.append(seed)
    for i in range(size-1):
        d0, d4, d5, d6 = seed[0], seed[4], seed[5], seed[6]
        xor = str(int(d0) ^ int(d4) ^ int(d5) ^ int(d6))
        seed = seed[1] + seed[2] + seed[3] + seed[4] + seed[5] + seed[6] + seed[7] + xor
        lfsr.append(seed)
    lfsr = np.array(lfsr).reshape((width, height))
    return lfsr

def rosslerMap(height, width, a, b, c, x0):
    f = lambda t, x: [-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)]
    SIZE = height*width
    tspan_ = np.linspace(0, 255, SIZE)
    sol = solve_ivp(f, [0, tspan_[-1]], x0, t_eval=tspan_)
    rossler_map, t = sol.y, sol.t
    rossler_map = abs(rossler_map * (10**14))
    rossler_map = (np.floor(rossler_map)%256).astype("uint8")
    x = []
    y = []
    z = []
    for i in range(rossler_map.shape[1]):
        x.append(np.binary_repr(rossler_map[0, i], width = 8))
        y.append(np.binary_repr(rossler_map[1, i], width = 8))
        z.append(np.binary_repr(rossler_map[2, i], width = 8))
    return np.array(x).reshape((height, width)), np.array(y).reshape((height, width)), np.array(z).reshape((height, width))

def henonMap(height, width, x, y, a, b):
    size = height*width
    x0, y0 = x, y
    xs, ys = np.zeros((size)), np.zeros((size))
    xs[0], ys[0] = x0, y0
    for i in range(1, size):
        y0 = b*x0
        x0 = 1 - (a*(x0**2)) +y0
        xs[i], ys[i] = x0, y0
    return np.array([np.binary_repr(i, width=8) for i in (np.floor(abs(xs * (10**14))%256)).astype("uint8")]).reshape((height, width))

def tentMap(height, width, x, r):
    size = height*width
    x0 = x
    xs = np.zeros((size))
    xs[0] = x
    for i in range(1, size):
        if x0<0.5:
            x0 = r*x0
        else:
            x0 = r*(1-x0)
        xs[i] = x0
    return np.array([np.binary_repr(i, width=8) for i in (np.floor(abs(xs * (10 ** 14))) % 256).astype("uint8")]).reshape((height, width))