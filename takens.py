# A demonstration of the Takens embedding theorem

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
import adaptivekde
from sklearn.metrics import mutual_info_score
from scipy.stats import iqr
import entropy_estimators as ee

def fd(x):
  # Freedman-Diaconis rule
  iqr = np.subtract(*np.percentile(x, [75, 25]))
  h = 2 * iqr * x.size ** (-1/3)
  if h == 0:
    return 1
  return np.floor((x.max() - x.min()) / h) + 1

def mutual_information(x, y):
  hist, xedges, yedges = np.histogram2d(x, y, bins=[fd(x), fd(y)])
  probs = hist / hist.sum()
  with np.errstate(divide='ignore', invalid='ignore'):
    joint = probs * np.log(probs / (probs.sum(axis=0) * probs.sum(axis=1)[:, np.newaxis]))
    return joint[np.isfinite(joint)].sum()

def mutual_information_kde(x, y):
  if np.allclose(x, y):
    return mutual_information(x, y)
  kde = gaussian_kde([x, y])
  raise NotImplementedError

def lorenz(state, time, rho=28, sigma=10, beta=8/3, noise=30 * 0):
  x, y, z = state
  return np.array([
    sigma * (y - x) + np.random.randn() * noise, 
    x * (rho - z) - y, 
    x * y - beta * z
  ])

def rossler(state, time, a=.15, b=.2, c=10):
  x, y, z = state
  return np.array([
    -y - z, 
    x + a * y, 
    b + z * (x - c)
  ])

def integrate(system, init, t):
  return odeint(system, init, t)

  def rk4(system, state, t, dt):
    k1 = dt * system(state, t)
    k2 = dt * system(state + k1 * .5, t)
    k3 = dt * system(state + k2 * .5, t)
    k4 = dt * system(state + k3, t)
    return state + (k1 + k2 + k2 + k3 + k3 + k4) / 6

  dt = np.diff(t)
  states = np.empty((len(t), len(init)))
  states[0] = init
  for i in range(len(t) - 1):
    states[i + 1] = rk4(system, states[i], t[i], dt[i])
  return states

if __name__ == '__main__':
  t, dt = np.linspace(0, 50, num=5000, retstep=True)
  x, y, z = integrate(lorenz, [-8, 8, 27], t).T

  #t, dt = np.linspace(0, 200, num=20000, retstep=True)
  #x, y, z = integrate(rossler, [10, 0, 0], t).T

  ax = plt.figure(0).gca(projection='3d')
  ax.set_title('Phase portrait')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.plot(x, y, z, linewidth=1)

  #x = np.loadtxt('audusd.csv', delimiter=';', usecols=1)
  #t = np.linspace(0, 1, num=len(x))

  ax = plt.figure(1).gca()
  ax.set_title('Time series')
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.plot(t, x, linewidth=1)

  ax = plt.figure(2).gca()
  ax.set_title('Lag selection')
  ax.set_xlabel('Lag')
  ax.set_ylabel('Mutual information')
  lags = np.arange(len(x) // 2)
  mis = np.array([
    mutual_information(x[lag:], np.roll(x, lag)[lag:])
    for lag in lags
  ])
  lag = np.argmax(np.diff(mis) > 0)
  print('Lag: {:.2f}'.format(lag * dt))
  ax.plot(lags[:lag*8] * dt, mis[:lag*8], linewidth=1)
  ax.axvline(x=lag * dt, linestyle='dotted')

  ax = plt.figure(3).gca(projection='3d')
  ax.set_title('Lag embedding')
  ax.plot(
    x.flatten(),
    np.roll(x, lag).flatten(),
    np.roll(x, lag * 2).flatten(),
    linewidth=1
  )

  plt.show()







