import numpy as np
import scipy.linalg as la


def intersect(startpoint_1, endpoint_1, startpoint_2, endpoint_2):
    A = np.array([[endpoint_1[0] - startpoint_1[0], -endpoint_2[0] + startpoint_2[0]],
                  [endpoint_1[1] - startpoint_1[1], -endpoint_2[1] + startpoint_2[1]]])
    b = np.array([startpoint_2[0] - startpoint_1[0],
                 startpoint_2[1] - startpoint_1[1]])

    if np.isclose(0, la.det(A)):
        u = endpoint_1 - startpoint_1
        v = startpoint_2 - startpoint_1
        if np.allclose(0, u):
            proj = np.zeros(2)
        else:
            proj = np.dot(u, v) / np.dot(u, u) * u

        if np.allclose(v, proj):
            v_start = startpoint_2 - startpoint_1
            proj1 = np.dot(u, v_start) / np.dot(u, u)
            v_end = endpoint_2 - startpoint_1
            proj2 = np.dot(u, v_end) / np.dot(u, u)
            if max(proj1, proj2) < 0 or min(proj1, proj2) > 1:
                return False
            else:
                return True
        else:
            return False
    else:
        t = la.solve(A, b)
        if np.logical_and((t >= 0), (t <= 1)).all():
            return True
        else:
            return False


def collides(obstacles, startpoint, endpoint):
    for obstacle in obstacles:
        for segment in range(obstacle.shape[1] - 1):
            if intersect(obstacle[:, segment], obstacle[:, segment + 1], startpoint, endpoint):
                return True
    return False


def bounce(startpoint, endpoint, obstacle_start, obstacle_end):
    obstacle_vector = obstacle_end - obstacle_start
    obstacle_unit = obstacle_vector / np.sqrt(
        np.dot(obstacle_vector, obstacle_vector))
    normal_unit = np.array([-obstacle_unit[1], obstacle_unit[0]])

    incoming_vector = endpoint - startpoint
    incoming_magnitude = np.sqrt(np.dot(incoming_vector, incoming_vector))
    incoming_unit = incoming_vector / incoming_magnitude

    dot_product = np.dot(incoming_unit, normal_unit)
    reflected_unit = incoming_unit - 2 * dot_product * normal_unit

    reflected_vector = reflected_unit * incoming_magnitude
    new_endpoint = startpoint + reflected_vector

    return new_endpoint


def bilinear_interpolation(x, y, x_mesh, y_mesh, values):
    if len(x_mesh) != values.shape[0] or len(y_mesh) != values.shape[1]:
        raise ValueError("Dimensions of function do not match mesh values.")

    if x <= x_mesh[0]:
        x = x_mesh[0]
    elif x >= x_mesh[-1]:
        x = x_mesh[-1]

    if y <= y_mesh[0]:
        y = y_mesh[0]
    elif y >= y_mesh[-1]:
        y = y_mesh[-1]

    for x_segment in range(len(x_mesh)):
        if x_mesh[x_segment] <= x <= x_mesh[x_segment + 1]:
            break

    for y_segment in range(len(y_mesh)):
        if y_mesh[y_segment] <= y <= y_mesh[y_segment + 1]:
            break

    ll_corner = values[x_segment, y_segment]
    lh_corner = values[x_segment, y_segment + 1]
    hl_corner = values[x_segment + 1, y_segment]
    hh_corner = values[x_segment + 1, y_segment + 1]

    t_x = (x - x_mesh[x_segment]) / (x_mesh[x_segment + 1] - x_mesh[x_segment])
    t_y = (y - y_mesh[y_segment]) / (y_mesh[y_segment + 1] - y_mesh[y_segment])

    return ll_corner + (hl_corner - ll_corner) * t_x + (lh_corner - ll_corner) * t_y + (hh_corner - hl_corner - lh_corner + ll_corner) * t_x * t_y
