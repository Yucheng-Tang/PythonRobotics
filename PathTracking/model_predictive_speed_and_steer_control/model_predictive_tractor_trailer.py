"""

Path tracking simulation with iterative linear model predictive control

Robot model: differential driven mobile robot with passive trailer

author: Yucheng Tang (@Yucheng-Tang)

Citation: Atsushi Sakai (@Atsushi_twi)
"""
import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import os
from sympy import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise


# NX = 4  # x = x, y, v, yaw
# NU = 2  # a = [accel, steer]
# T = 5  # horizon length

NX = 4 # x = x, y, yaw, yawt
NU = 2 # u = [v, w]
T = 10 # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([5.0, 5.0]) # ([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.01, 10.0])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 20 # 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 2.0  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]
ROD_LEN = 3.0 # [m]
CP_OFFSET = 0.0 # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]

MAX_OMEGA = np.deg2rad(60.0)  # maximum rotation speed [rad/s]

MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True

# TODO: add control horizon variable
# TODO: Notebook and Git

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, yawt=0.0, v = 0.0, dyaw = 0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        # self.v = v
        # self.dyaw = dyaw
        self.predelta = None


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(xref_t, xref_t1, x_t):
    # dyawt = (xref_t1[3]-xref_t[3])
    # dyawt is not the velocity theta2 at operation point!!!
    rp_t = np.array([xref_t[0], xref_t[1]])
    rp_t1 = np.array([xref_t1[0], xref_t1[1]])
    v_r = np.linalg.norm(rp_t1 - rp_t)
    w_r = xref_t1[2] - xref_t[2]
    # velocity = distance / DT

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    # A[3, 3] = 1.0

    # model from pythonrobotic
    A[0, 2] = v_r * math.sin(x_t[3] - x_t[2]) * math.cos(x_t[3])
    A[0, 3] = - v_r * math.sin(2 * x_t[3] - x_t[2])
    A[1, 2] = v_r * math.sin(x_t[3] - x_t[2]) * math.sin(x_t[3])
    A[1, 3] = v_r * math.cos(2 * x_t[3] - x_t[2])
    A[3, 2] = (v_r * math.cos(x_t[3] - x_t[2]) - CP_OFFSET * w_r * math.sin(x_t[3] - x_t[2])) / ROD_LEN
    A[3, 3] = 1.0 - (v_r * math.cos(x_t[3] - x_t[2]) - CP_OFFSET * w_r * math.sin(x_t[3] - x_t[2])) / ROD_LEN

    # my model
    # A[0, 1] = DT * dyawt
    # A[1, 0] = - DT * dyawt
    # A[1, 3] = DT * v_r * math.cos(xref_t[3] - xref_t[2])
    # mdoel from paper
    # A[0, 1] = dyawt
    # A[1, 0] = - dyawt
    # A[1, 3] = v_r * math.cos(xref_t[2] - xref_t[3])

    B = np.zeros((NX, NU))

    # model from pythonrobotics
    B[0, 0] = DT * math.cos(x_t[3] - x_t[2]) * math.cos(x_t[3])
    B[1, 0] = DT * math.cos(x_t[3] - x_t[2]) * math.sin(x_t[3])
    B[2, 1] = DT
    B[3, 0] = - DT * math.sin(x_t[3] - x_t[2]) / ROD_LEN
    B[3, 1] = - DT * CP_OFFSET * math.cos(x_t[3] - x_t[2]) / ROD_LEN

    # my model
    # B[0, 0] = DT * math.cos(x_t[3] - x_t[2])
    # B[2, 1] = - DT
    # B[3, 0] = DT * math.sin(x_t[3] - x_t[2]) / ROD_LEN

    # model from paper
    # B[0, 0] = DT * math.cos(x_t[2] - x_t[3])
    # B[2, 1] = DT
    # B[3, 0] = DT * math.sin(x_t[2] - x_t[3]) / ROD_LEN
    # print("B", B[0, 0], B[3, 0])

    C = np.zeros(NX)

    # model from pythonrobotics
    C[0] = - v_r * (x_t[2] * math.sin(x_t[3] - x_t[2]) * math.cos(x_t[3]) - x_t[3] * math.sin(2 * x_t[3] - x_t[2]))
    C[1] = - v_r * (x_t[2] * math.sin(x_t[3] - x_t[2]) * math.sin(x_t[3]) + x_t[3] * math.cos(2 * x_t[3] - x_t[2]))
    C[3] = (x_t[3] - x_t[2]) * (v_r * math.cos(x_t[3] - x_t[2]) - CP_OFFSET * w_r * math.sin(x_t[3] - x_t[2])) / ROD_LEN

    # my model
    # C[0] = DT * v_r * math.cos(xref_t[3] - xref_t[2])
    # C[2] = DT * (xref_t1[2] - xref_t[2])
    # C[3] = DT * (xref_t1[3] - xref_t[3])

    # model from paper
    # C[0] = - v_r * math.cos(xref_t[2] - xref_t[3])
    # C[2] = (xref_t1[2] - xref_t[2])
    # C[3] = (xref_t1[3] - xref_t[3])
    # print("C", C[2], C[3])

    return A, B, C

# TODO: modify the plot_car to use x2, y2, yaw and yawt, plot tractor-trailer system in one function?
def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    # fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
    #                      [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

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

def update_state(state, v, dyaw):

    # print("(x,y, yaw, yawt): ", state.x, state.y, state.yaw, state.yawt, " control input (v, dyaw):", v, dyaw)

    # input check
    if dyaw >= MAX_OMEGA:
        dyaw = MAX_OMEGA
    elif dyaw <= -MAX_OMEGA:
        dyaw = -MAX_OMEGA

    # my model
    # state.x = state.x + v * math.cos(state.yawt - state.yaw) * math.cos(state.yawt) * DT
    # state.y = state.y + v * math.cos(state.yawt - state.yaw) * math.sin(state.yawt) * DT
    # state.yaw = state.yaw + dyaw * DT
    # state.yawt = state.yawt - v / ROD_LEN * math.sin(state.yawt - state.yaw) * DT

    # model from paper
    state_new = State(x=state.x, y=state.y, yaw=state.yaw, yawt=state.yawt)

    state_new.x = state.x + v * math.cos(state.yaw - state.yawt) * math.cos(state.yawt) * DT
    state_new.y = state.y + v * math.cos(state.yaw - state.yawt) * math.sin(state.yawt) * DT
    state_new.yaw = state.yaw + dyaw * DT
    state_new.yawt = state.yawt + v / ROD_LEN * math.sin(state.yaw - state.yawt) * DT \
                    - CP_OFFSET * dyaw * math.cos(state.yaw - state.yawt) / ROD_LEN

    # state.x = state.x + v * math.cos(state.yaw - state.yawt) * math.cos(state.yawt) * DT
    # state.y = state.y + v * math.cos(state.yaw - state.yawt) * math.sin(state.yawt) * DT
    # state.yaw = state.yaw + dyaw * DT
    # state.yawt = state.yawt + v / ROD_LEN * math.sin(state.yaw - state.yawt) * DT

    return state_new


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, ov, odyaw, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[2], yawt=x0[3])
    for (vi, dyawi, i) in zip(ov, odyaw, range(1, T + 1)):
        state = update_state(state, vi, dyawi)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.yaw
        xbar[3, i] = state.yawt

    return xbar


def iterative_linear_mpc_control(xref, x0, ov, odyaw):
    """
    MPC contorl with updating operational point iteraitvely
    """

    if ov is None or odyaw is None:
        ov = [0.0] * T
        odyaw = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, ov, odyaw, xref)
        pov, podyaw = ov[:], odyaw[:]
        ov, odyaw, ox, oy, oyaw, oyawt = linear_mpc_control(xref, xbar, x0)
        du = sum(abs(ov - pov)) + sum(abs(odyaw - podyaw))  # calc u change value
        if du <= DU_TH: # iteration finish param
            break
    else:
        print("Iterative is max iter")

    return ov, odyaw, ox, oy, oyaw, oyawt

# TODO: check control model
def linear_mpc_control(xref, xbar, x0):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        # cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xref[:, t], xref[:, t+1], xbar[:, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        # anti-jackknife
        constraints += [x[3, t] - x[2, t] <= np.deg2rad(45.0)]
        # TODO: anti-jackknife activated?

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
        #     constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
        #                     MAX_DSTEER * DT]

    # cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)
    # TODO: target pose penalty and input difference cost?

    constraints += [x[:, 0] == x0]
    # constraints += [x[2, :] <= MAX_SPEED]
    # constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_SPEED]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_OMEGA]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        oyaw = get_nparray_from_matrix(x.value[2, :])
        oyawt = get_nparray_from_matrix(x.value[3, :])
        ov = get_nparray_from_matrix(u.value[0, :])
        odyaw = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        ov, odyaw, ox, oy, oyaw, oyawt = None, None, None, None, None, None

    return ov, odyaw, ox, oy, oyaw, oyawt


def calc_ref_trajectory(state, cx, cy, cyaw, cyawt, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyawt, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    # xref[2, 0] = sp[ind]
    #TODO: if the pose of tractor could be calculate in this case, when yes, define the function cal_tractor_pose
    xref[2, 0] = cyaw[ind] # cal_tractor_pose(cyawt[ind])
    xref[3, 0] = cyawt[ind]

    for i in range(T + 1):
        # TODO: what will happen when v is not state variable and should be negative?
        if (ind + i) < ncourse:
            xref[0, i] = cx[ind + i]
            xref[1, i] = cy[ind + i]
            xref[2, i] = cyaw[ind + i]  # cal_tractor_pose(cyawt[ind + dind])
            xref[3, i] = cyawt[ind + i]
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            # xref[2, i] = sp[ncourse - 1]
            # TODO: traget yaw position should be pre-defined!!!
            xref[2, i] = cyaw[ncourse - 1] # cal_tractor_pose(cyawt[ncourse - 1])
            xref[3, i] = cyawt[ncourse - 1]

    return xref, ind


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    if isgoal:
        return True

    return False


def do_simulation(cx, cy, cyaw, cyawt, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cyawt: course yaw position of the trailer list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yawt - cyawt[0] >= math.pi:
        state.yawt -= math.pi * 2.0
    elif state.yawt - cyawt[0] <= -math.pi:
        state.yawt += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    yawt = [state.yawt]
    t = [0.0]
    dyaw = [0.0]
    v = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyawt, 0)

    # txt_list = [state.x-np.cos(state.yaw)]
    # tyt_list = [state.y-np.sin(state.yaw)]
    txt_list = [state.x + math.cos(state.yawt) * ROD_LEN]
    tyt_list = [state.y + math.sin(state.yawt) * ROD_LEN]
    # txm_list = [state.x-np.cos(state.yaw)]
    # tym_list = [state.y-np.sin(state.yaw)]

    ov, odyaw = None, None

    #TODO: check the function smooth_yaw
    cyawt = smooth_yaw(cyawt)

    while MAX_TIME >= time:
        xref, target_ind = calc_ref_trajectory(
            state, cx, cy, cyaw, cyawt, ck, sp, dl, target_ind)
        print(target_ind)
        # target_ind += 1

        x0 = [state.x, state.y, state.yaw, state.yawt]  # current state

        ov, odyaw, ox, oy, oyaw, oyawt = iterative_linear_mpc_control(
            xref, x0, ov, odyaw)

        if odyaw is not None:
            dyawi, vi = odyaw[0], ov[0]

        state = update_state(state, vi, dyawi)
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        yawt.append(state.yawt)
        t.append(time)
        dyaw.append(dyawi)
        v.append(vi)

        if time == 0.2:
            phi_tractrix = 1
            phi_model = 0
        # dyaw = yaw[-1] - yaw[-2]
        # TODO: Tractrix curve for kinematic model?

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        # print("ov, odyaw", ov, odyaw)
        # print("ox, oy, oyaw, oyawt", ox, oy, oyaw, oyawt)
        # print("ref", xref[0, :], xref[1, :], xref[2, :])
        #
        # print("state trailer: ", state.x, state.y, state.yawt,
        #       "diff ref state trailer: ", (xref[0,0] - state.x),  xref[1,0] - state.y, xref[3,0] - state.yawt,
        #       "state tractor: ", state.x + np.cos(state.yawt) * ROD_LEN, state.y + np.sin(state.yawt) * ROD_LEN,
        #       state.yaw, "control input", dyawi, vi)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            # TODO: the whole tractor-trailer system will be plot in one function
            plot_car(state.x, state.y, state.yawt)

            # for tractor
            plot_car(state.x + np.cos(state.yawt) * ROD_LEN + np.cos(state.yaw) * CP_OFFSET, state.y + np.sin(state.yawt) * ROD_LEN + np.sin(state.yaw) * CP_OFFSET, state.yaw)
            txt_list.append(state.x + np.cos(state.yawt) * ROD_LEN)
            tyt_list.append(state.y + np.sin(state.yawt) * ROD_LEN)

            # txm, tym, tyawm, txt, tyt, tyawt = cal_trailer_pose(state, phi_tractrix, phi_model, dyaw)
            # txm_list.append(txm)
            # tym_list.append(tym)
            # txt_list.append(txt)
            # tyt_list.append(tyt)

            # plot_car(txt, tyt, tyawt)
            plt.axis("equal")
            plt.grid(True)
            # plt.title("Time[s]:" + str(round(time, 2))
            #           + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.05)
    return t, x, y, yaw, yawt, dyaw, v, txt_list, tyt_list

    # return t, x, y, yaw, v, dyaw, v, txm_list, tym_list, txt_list, tyt_list

# def cal_trailer_pose(tractor_state, phi_tractrix, phi_model, dyaw):
#     phi_model += - tractor_state.v * np.sin(phi_model) / 3 - dyaw
#     txm = tractor_state.x - np.cos(tractor_state.yaw + phi_model) * 3
#     tym = tractor_state.y - np.sin(tractor_state.yaw + phi_model) * 3
#     tyawm = tractor_state.yaw + phi_model
#
#     # equation from MA Clemens (Theory)
#     # phi_tractrix += - 1/3 + np.sin(phi_tractrix) - dyaw
#     # equation from MA Clemens (Code)
#     # phi_tractrix += 1 / 3 * np.sin(tractor_state.yaw - dyaw - phi_tractrix)
#     # txt = tractor_state.x - np.cos(tractor_state.yaw + phi_tractrix) * 3
#     # tyt = tractor_state.y - np.sin(tractor_state.yaw + phi_tractrix) * 3
#     # tyawt = tractor_state.yaw + phi_tractrix
#
#     rod_length = 3
#     t = np.tan(phi_tractrix+0.00001) #phi_tractrix
#
#     x = symbols('x')
#     # q1 = rod_length*ln((rod_length+sqrt(rod_length**2*t/(1+t)))/(sqrt(rod_length**2/(1+t)))) - sqrt(rod_length**2*t/(1+t))
#     # q2 = rod_length * ln((rod_length + sqrt(rod_length ** 2 * x / (1 + x))) / (sqrt(rod_length ** 2 / (1 + x)))) - sqrt(rod_length**2*x/(1+x))
#     # q3 = t*sqrt(rod_length**2/(1+t)) + tractor_state.v
#     # q4 = x*sqrt(rod_length**2/(1+x))
#
#     # print(q1, q3)
#
#     # eq = q1 - q2 + q3 -q4
#     eq = exp(x)-exp(-x) - 1/t
#     sol = solve(eq, x)
#     # print("sol", sol[0])
#
#     sol[0] += tractor_state.v
#     phi_tractrix = np.arctan(1/(math.exp(sol[0])-math.exp(-sol[0])))
#     print(phi_tractrix)
#     txt = tractor_state.x - np.cos(tractor_state.yaw + phi_tractrix) * 3
#     tyt = tractor_state.y - np.sin(tractor_state.yaw + phi_tractrix) * 3
#     tyawt = tractor_state.yaw + phi_tractrix
#
#     return txm, tym, tyawm, txt, tyt, tyawt




def calc_speed_profile(cx, cy, cyaw, target_speed):
    # Clculate the moving direction of each time step

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)
        # print(cyaw[i], move_direction, dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed
        # speed_profile[i] = -target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def get_straight_course(dl):
    ax = [0.0, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0]
    ay = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck


def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck


def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck

def get_circle_course(dl):
    t = np.linspace(0, 2*math.pi, num=200)
    ax = math.sin(t)
    ay = math.cos(t)

    return ax, ay, t



def main():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    # cx, cy, cyawt, ck = get_straight_course(dl)
    # cyawt = np.zeros(len(cyawt))
    # cx, cy, cyawt, ck = get_straight_course2(dl)
    # cx, cy, cyawt, ck = get_straight_course3(dl)
    # cx, cy, cyawt, ck = get_forward_course(dl)
    # print(cyawt)
    # cyawt = [abs(i) for i in cyawt]
    cx, cy, cyawt, ck = get_switch_back_course(dl)
    # cx, cy, cyawt = get_circle_course(dl)

    sp = calc_speed_profile(cx, cy, cyawt, TARGET_SPEED)
    # TODO: get cyaw from high-level planning part
    cyaw = np.copy(cyawt)
    print(sp)

    initial_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], yawt=cyawt[0])

    t, x, y, yaw, yawt, dyawt, v, txt, tyt = do_simulation(
        cx, cy, cyaw, cyawt, ck, sp, dl, initial_state)

    print("control input:", dyawt, v)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="trailer")
        plt.plot(txt, tyt, "-b", label="tractor")
        # plt.plot(txm, tym, "-y", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


def main2():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = get_straight_course3(dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=cy[0], yaw=0.0, yawt=0.0)
    print(initial_state.yaw, initial_state.yawt)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == '__main__':
    main()
    # main2()
