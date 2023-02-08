# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # evenly sampled time at 200ms intervals
# t = np.arange(-45, 45, 0.2)
# fct = (1/(45-abs(t+0.001)))**2
# fct2 = -np.log(45 - abs(t+0.001))+np.log(45)
#
# # red dashes, blue squares and green triangles
# # plt.plot(t, fct, 'r-')
# plt.plot(t, fct2, 'b--')
# plt.show()
import math

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

import time


v_limit = 0.2
time_interval = 10

# Define problem
U_min = -np.array([v_limit, np.deg2rad(90.0)])
U_max =  np.array([0, np.deg2rad(90.0)])
U_min_random = U_min
U_max_random = U_max
for i in range(time_interval-1):
    U_min_random = np.concatenate((U_min_random, U_min),axis=None)
    U_max_random = np.concatenate((U_max_random, U_max), axis=None)

# state_new.x = state.x + v * math.cos(state.yaw - state.yawt) * math.cos(state.yawt) * DT
# state_new.y = state.y + v * math.cos(state.yaw - state.yawt) * math.sin(state.yawt) * DT
# state_new.yaw = state.yaw + dyaw * DT
# state_new.yawt = state.yawt + v / ROD_LEN * math.sin(state.yaw - state.yawt) * DT \
#                  - CP_OFFSET * dyaw * math.cos(state.yaw - state.yawt) / ROD_LEN * DT
DT = 0.1

# Vehicle parameters
LENGTH = 0.72  # [m]
LENGTH_T = 0.36  # [m]
WIDTH = 0.48  # [m]
BACKTOWHEEL = 0.36  # [m]
WHEEL_LEN = 0.1  # [m]
WHEEL_WIDTH = 0.05  # [m]
TREAD = 0.2  # [m]
WB = 0.3  # [m]
ROD_LEN = 0.5  # [m]
CP_OFFSET = 0.1  # [m]

def plot_car(x, y, yaw, length, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover
    # using BACKTOWHEEL parameter
    # outline = np.array([[-BACKTOWHEEL, (length - BACKTOWHEEL), (length - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
    #                    [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    outline = np.array([[-length / 2, (length - length / 2), (length - length / 2), -length / 2, -length / 2],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    # fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
    #                      [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    # fl_wheel = np.copy(fr_wheel)
    # fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    # Rot2 = np.array([[math.cos(steer), math.sin(steer)],
    #                  [-math.sin(steer), math.cos(steer)]])

    # fr_wheel = (fr_wheel.T.dot(Rot2)).T
    # fl_wheel = (fl_wheel.T.dot(Rot2)).T
    # fr_wheel[0, :] += WB
    # fl_wheel[0, :] += WB

    # fr_wheel = (fr_wheel.T.dot(Rot1)).T
    # fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    # fr_wheel[0, :] += x
    # fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    # fl_wheel[0, :] += x
    # fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    # plt.plot(np.array(fr_wheel[0, :]).flatten(),
    #          np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    # plt.plot(np.array(fl_wheel[0, :]).flatten(),
    #          np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

def f(x, u):
    x1 = []
    x1.append(x[0] + u[0] * math.cos(x[2]-x[3]) * math.cos(x[3]) * DT)
    x1.append(x[1] + u[0] * math.cos(x[2]-x[3]) * math.sin(x[3]) * DT)
    x1.append(x[2] + u[1] * DT)
    x1.append(x[3] + u[0] / ROD_LEN * math.sin(x[2]-x[3]) * DT - CP_OFFSET * u[1] * math.cos(x[2]-x[3]) / ROD_LEN * DT)
    x = x1
    return x

start = time.time()

# RandUP
M    = 1000
x = [0, 0, 0, -0] # 0.5234
# with random control inputs but keep same during the time interval
ys = np.zeros((M, 4))
# with completely random control inputs
ys_random = np.zeros((M, 4))
# bang-bang strategy
ysu = np.zeros(4)
ysl = np.zeros(4)
ysz = np.zeros(4)
limit = 50
us = np.random.uniform(low=U_max, high=U_min, size=(M,2))
us_random = np.random.uniform(low=U_max_random, high=U_min_random, size=(M, 2*time_interval))
# print("us", us, "us_random", len(us_random[0]))
for t in range(time_interval):
    if t == 0:
        ysu = f(x, [-v_limit, np.deg2rad(limit)])
        ysl = f(x, [-v_limit, -np.deg2rad(limit)])
        ysz = f(x, [-v_limit, 0])
        for i in range(len(us)):
            ys[i, :] = f(x, us[i])
            # print("comparison", us[i], us_random[i,t*2:t*2+2])
            ys_random[i, :] = f(x, us_random[i,t*2:t*2+2])
    else:
        ysu = f(ysu, [-v_limit, np.deg2rad(limit)])
        ysl = f(ysl, [-v_limit, -np.deg2rad(limit)])
        ysz = f(ysz, [-v_limit, 0])
        for i in range(len(us)):
            ys[i, :] = f(ys[i], us[i])
            # compare the control inputs of the 100 st point
            if i == 100:
                print("comparison", us[i],us_random[i, t * 2:t * 2 + 2])
            ys_random[i, :] = f(ys_random[i], us_random[i, t*2:t*2+2])
print(ys[:, :2])
hull = scipy.spatial.ConvexHull(ys[:, :2])
# hull_random = scipy.spatial.ConvexHull(ys_random[:, :2])
# hull_bang = scipy.spatial.ConvexHull([[0,0],ysu, ysl, ysz])

end = time.time()
print(end - start)

# Plot
plt.scatter(ys[:,0],ys[:,1], color='b')
plt.scatter(ys_random[:,0],ys_random[:,1], color='g')
plt.scatter(ysu[0],ysu[1], color='r')
plt.scatter(ysl[0],ysl[1], color='r')
plt.scatter(ysz[0],ysz[1], color='r')
j=0
print(ys.shape)
test_x = np.array([1, 2, 3])
np.delete(test_x, 1)
print(test_x)
# while j < len(ys):
#     if abs(ys[j, 3] - ys[j, 2]) >= np.deg2rad(45):
#         print("delete", j)
#         np.delete(ys, j, 0)
#     j+=1
# print(ys.shape)
# for i in range(len(ys)):
#     # if i % 10 == 0:
#     if abs(ys[i, 3] - ys[i, 2]) <= np.deg2rad(45) and i % 10 == 0:
#         print(abs(ys[i, 3] - ys[i, 2]))
#         plot_car(ys[i, 0], ys[i, 1], ys[i, 3], LENGTH_T)
#
#         plot_car(ys[i, 0] + np.cos(ys[i, 3]) * ROD_LEN + np.cos(ys[i, 2]) * CP_OFFSET,
#                 ys[i, 1] + np.sin(ys[i, 3]) * ROD_LEN + np.sin(ys[i, 2]) * CP_OFFSET, ys[i, 2], LENGTH)

# for tractor
# plot_car(state.x + np.cos(state.yawt) * ROD_LEN + np.cos(state.yaw) * CP_OFFSET,
#          state.y + np.sin(state.yawt) * ROD_LEN + np.sin(state.yaw) * CP_OFFSET, state.yaw, LENGTH)
for s in hull.simplices:
  plt.plot(ys[s,0], ys[s,1], 'g')
plt.show()