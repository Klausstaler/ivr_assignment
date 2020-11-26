from sympy import *

def calc_trans(theta, d=0.0, a=0.0, alpha=0.0):
    x_rot, x_trans, z_trans, z_rot = eye(4), eye(4), eye(4), eye(4)
    x_rot[:-1, :-1] = Matrix([[1, 0, 0],
                                                    [0, cos(alpha), -sin(alpha)],
                                                    [0, sin(alpha), -cos(alpha)]])
    x_trans[0, -1] = a
    z_rot[:-1, :-1] = Matrix([[cos(theta), -sin(theta), 0],
                                                    [sin(theta), cos(theta), 0],
                                                    [0, 0, 1]])
    z_trans[-2, -1] = d
    return z_rot@z_trans@x_trans@x_rot

def calculate_jacobian(robot):
    t1, t2, t3, t4 = symbols("theta_1"), symbols("theta_2"), symbols("theta_3"), symbols("theta_4")
    link1, link2, link3, link4 = robot.link1, robot.link2, robot.link3, robot.link4
    link1_mat = calc_trans(t1+link1.theta, d=link2.d, alpha=link1.alpha)
    link2_mat = calc_trans(t2+link2.theta, alpha=link2.alpha)
    link3_mat = calc_trans(t3, a=link3.a, alpha=-link3.alpha)
    link4_mat = calc_trans(t4, a=link4.a)
    fk = (link1_mat@link2_mat@link3_mat@link4_mat)[:3, -1]
    jac = fk.jacobian([t1, t2, t3, t4])
    return lambdify([t1,t2,t3,t4],jac,"numpy")


