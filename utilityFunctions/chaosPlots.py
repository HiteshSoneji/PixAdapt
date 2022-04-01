import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import solve_ivp

def tentMap(r, x):
    return np.where(x<0.5, r*x, r*(1-x))

def tentMapPlot():
    n = 10000
    r = np.linspace(1.0121, 2, n)
    iterations = 1000
    last = 100
    x = 1e-5 * np.ones(n)

    for i in range(iterations):
        x = tentMap(r, x)
        if i >= (iterations - last):
            plt.plot(r, x, ',k', alpha=.25, c='skyblue')
    plt.title('Tent map')
    plt.ylabel('$\it{x}$')
    plt.xlabel('$\it{r}$')
    plt.show()

def rosslerMapPlot():
    a = 0.1
    b = 0.1

    crange = np.arange(0.51, 30, 0.1)

    k = -1
    tspan_ = np.arange(0, 400, 0.05)
    tspan = [0, 400]
    xmax = []

    for c in crange:
        j = 0
        k = k+1
        f = lambda t, x: [-x[1]-x[2], x[0]+a*x[1], b+x[2]*(x[0]-c)]
        x0 = [1, 1, 0]
        sol = solve_ivp(f, tspan, x0, t_eval = tspan_)
        x, t = sol.y, sol.t
        count = np.argwhere(t>100)
        count = count.reshape((count.shape[0]))
        x = x[:, count]
        n = x.shape[1]
        temp = []
        for i in range(1, n-1):
            if (x[1, i-1]+ np.finfo(float).eps) < x[1, i] and x[1, i] > x[1, i+1]+np.finfo(float).eps:
                temp.append(x[1, i])
                j = j + 1
        if j > 1:
            plt.scatter([c for i in range(len(temp))], temp, alpha=.25, c ='lightblue', s=2)

    plt.xlabel("C")
    plt.ylabel("X")
    plt.title("Rossler Bifurcation diagram")
    plt.show()

def logistic(r, x):
    return r * x * (1 - x)

def logisticMapPlot():
    n = 10000
    r = np.linspace(2.5, 4.0, n)
    iterations = 1000
    last = 100
    x = 1e-5 * np.ones(n)

    for i in range(iterations):
        x = logistic(r, x)
        if i >= (iterations - last):
            plt.plot(r, x, ',k', alpha=.25, c='lightblue')
    plt.title('Logistic map')
    plt.ylabel('$\it{x}$')
    plt.xlabel('$\it{r}$')
    plt.show()


tentMapPlot()
