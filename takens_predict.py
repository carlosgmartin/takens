import numpy as np
import matplotlib.pyplot as plt
from takens import lorenz, integrate, mutual_information, rossler
from functools import lru_cache
from matplotlib.ticker import MaxNLocator

def get_optimal_lag(x):
  m = np.inf
  for lag in range(len(x)):
    m_next = mutual_information(x[lag:], np.roll(x, lag)[lag:])
    if m < m_next:
      return lag - 1
    else:
      m = m_next

#x = integrate(lorenz, [-8, 8, 27], np.linspace(0, 50, num=5000)).T[0]
x = integrate(rossler, [10, 0, 0], np.linspace(0, 200, num=20000)).T[0]

lag = get_optimal_lag(x)

@lru_cache(maxsize=None)
def y(i, d):
  return x[i + np.arange(d) * lag]

@lru_cache(maxsize=None)
def n(i, d):
  z = y(i, d)
  ys = np.array([np.max(np.abs(z - y(j, d))) for j in range(len(x) - d * lag)])
  return np.argmin(np.where(ys > 0, ys, np.inf))

@lru_cache(maxsize=None)
def a(i, d):
  j = n(i, d)
  return np.max(np.abs(y(i, d + 1) - y(j, d + 1))) / np.max(np.abs(y(i, d) - y(j, d)))

def E(d):
  return np.mean([a(i, d) for i in range(len(x) - d * lag)])

ds = np.arange(1, 20)
Es = np.empty(ds.shape)
for i in range(len(ds)):
  Es[i] = E(ds[i])
  print(ds[i], Es[i])

'''
lorenz
Es = [
  42679.278652,
  7.82363526585,
  1.56245262148,
  1.45424831907,
  1.38510207316,
  1.33595443488,
  1.28841275809,
  1.25461593162,
  1.24223100146,
  1.23719123147,
  1.23646630603,
  1.20381712184,
  1.1964324421,
  1.18753258517,
  1.17944374049,
  1.15406171579,
  1.10758286301,
  1.10570002705,
  1.10368518038
]

rossler
Es = [
  219307.21715,
  10.0257546413,
  1.83276678437,
  1.14288165908,
  1.11928596777,
  1.08419700319,
  1.07020830161,
  1.07298436778,
  1.0626421816,
  1.05667764773,
  1.04743086324,
  1.04019776786,
  1.03068167325,
  1.02895765979,
  1.02389756114,
  1.02183075383,
  1.02131517821,
  1.01878792737,
  1.01725007218
]

1 209977.822566
2 11.6281525351
3 1.98765837718
4 1.13200159558
5 1.10504101595
6 1.09295261829
7 1.07178097895
8 1.06236490862
9 1.06186140438
10 1.05535900149
11 1.04926870252
12 1.04188596937
13 1.04077487408
14 1.03651221496
15 1.02433264191
16 1.0223972937
17 1.02016020952
18 1.02001981225
19 1.01944094497
'''

plt.plot(ds[:-1], Es[1:] / Es[:-1], label='E1')
plt.plot(ds[1:-1], np.diff(Es[1:] / Es[:-1]), label='E1 derivative')
plt.xlabel('d')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Embedding dimension')
plt.show()




