import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la


def intersect(startpoint_1, endpoint_1, startpoint_2, endpoint_2):
    """Determine if the 2D lines v1: `startpoint_1` to `endpoint_1` and v2: `startpoint_2` to `endpoint_2` intersect. Return a tuple with entry 0 as a Boolean stating if they intersect or touch, and if `True` is returned, next two entries show first intersection point with v2 along v1 and relative progression along v1."""
    A = np.array([[endpoint_1[0] - startpoint_1[0], -endpoint_2[0] + startpoint_2[0]],
                  [endpoint_1[1] - startpoint_1[1], -endpoint_2[1] + startpoint_2[1]]])
    b = np.array([startpoint_2[0] - startpoint_1[0],
                  startpoint_2[1] - startpoint_1[1]])

    if np.isclose(0, la.det(A)):
        u = endpoint_2 - startpoint_2
        v = startpoint_1 - startpoint_2
        if np.allclose(0, u):
            v_proj = np.zeros(2)
        else:
            v_proj = np.dot(u, v) / np.dot(u, u) * u

        if np.allclose(v, v_proj):
            v_start = startpoint_1 - startpoint_2
            proj1 = np.dot(u, v_start) / np.dot(u, u)
            v_end = endpoint_1 - startpoint_2
            proj2 = np.dot(u, v_end) / np.dot(u, u)
            if max(proj1, proj2) < 0 or min(proj1, proj2) > 1:
                return (False,)
            else:
                if 0 <= proj1 <= 1:
                    return (True, startpoint_1, 0)
                elif proj1 < 0:
                    progression = -proj1 / (proj2 - proj1)
                    return (True, startpoint_2, progression)
                elif proj1 > 1:
                    progression = (proj1 - 1) / (proj1 - proj2)
                    return (True, endpoint_2, progression)
        else:
            return (False,)
    else:
        t = la.solve(A, b)
        if np.logical_and((t >= 0), (t <= 1)).all():
            intersection_point = startpoint_1 + \
                t[0] * (endpoint_1 - startpoint_1)
            progression = t[0]
            return (True, intersection_point, progression)
        else:
            return (False,)


def collides(obstacles, startpoint, endpoint):
    """Determine if a line segment defined from `startpoint` to `endpoint` collides with a set of `obstacles`, a list of arrays of size `(2, number_of_points)` defining polylines."""
    for obstacle in obstacles:
        for segment in range(obstacle.shape[1] - 1):
            if intersect(obstacle[:, segment], obstacle[:, segment + 1], startpoint, endpoint)[0]:
                return True
    return False


def bounce(startpoint, endpoint, obstacle_start, obstacle_end):
    """Given a single line obstacle and a line segment, determine the endpoint including bouncing on the obstacle, then return as index `0` in tuple. Index `1` gives bounce point if it exists, `None` if not. Index `2` gives progress along path when colliding with the obstacle, if there is collision."""
    intersection = intersect(startpoint, endpoint,
                             obstacle_start, obstacle_end)
    if intersection[0]:
        bounce_point = intersection[1]
        progress = intersection[2]

        obstacle_vector = obstacle_end - obstacle_start
        endpoint_to_obstacle = endpoint - obstacle_start
        endpoint_projection = np.dot(endpoint_to_obstacle, obstacle_vector) / np.dot(
            obstacle_vector, obstacle_vector) * obstacle_vector

        new_endpoint = 2 * endpoint_projection - endpoint_to_obstacle + obstacle_start

        if progress != 0:
            return (new_endpoint, bounce_point, progress)
        else:
            return (endpoint, None, None)
    else:
        return (endpoint, None, None)


def bounce_obstacles(obstacles, startpoint, endpoint):
    bounce_points = []
    current_startpoint = startpoint
    current_endpoint = endpoint
    length = np.sqrt(np.dot(endpoint - startpoint,
                     endpoint - startpoint))
    while True:
        colliding_obstacles = []
        for obstacle in obstacles:
            for segment in range(obstacle.shape[1] - 1):
                intersection = intersect(current_startpoint, current_endpoint,
                                         obstacle[:, segment], obstacle[:, segment + 1])

                if intersection[0] and not np.isclose(0, intersection[2]):
                    colliding_obstacles.append(
                        ((obstacle[:, segment], obstacle[:, segment + 1]), intersection[1], intersection[2]))

        if len(colliding_obstacles) > 0:
            distances = [obstacle[2] for obstacle in colliding_obstacles]
            min_distance = np.min(distances)
            first_obstacle_index = np.argwhere(np.isclose(
                distances, np.ones_like(distances) * min_distance)).flatten()
            print(first_obstacle_index)
            if len(first_obstacle_index) == 1:
                first_obstacle = colliding_obstacles[first_obstacle_index[0]]
                bounce_point = first_obstacle[1]
                bounce_points.append(bounce_point)
                new_point = bounce(current_startpoint, current_endpoint,
                                   first_obstacle[0][0], first_obstacle[0][1])[0]
                current_startpoint = bounce_point
                current_endpoint = new_point
            else:
                current_endpoint = current_endpoint + \
                    np.random.normal(0, length * 10**-6, 2)
                continue
        else:
            return current_endpoint, np.array(bounce_points)


"""
obstacles = [np.array([[-4, -4, 5, 5, 1, 1, -1, -1, 5, 5, 12, 12, 10, 10, 7, 7, -6, -6],
                      [-8, 4, 4, -5, -5, 1, 1, -7, -7, -11, -11, 4, 4, -9, -9, 6, 6, -8]])]
startpoint = np.array([0, 0])
endpoint = np.random.normal(np.zeros(2), np.ones(2) * 100)
endpoint = np.array([20, 20])

bounce_demo = bounce_obstacles(obstacles, startpoint, endpoint)
print(bounce_demo[1])

fig, ax = plt.subplots()

for obstacle in obstacles:
    ax.plot(obstacle[0], obstacle[1], 'r', linewidth=3)
ax.plot([startpoint[0], bounce_demo[1][0, 0]],
        [startpoint[1], bounce_demo[1][0, 1]])
ax.plot(bounce_demo[1][:, 0], bounce_demo[1][:, 1])
ax.plot([bounce_demo[1][-1, 0], bounce_demo[0][0]],
        [bounce_demo[1][-1, 1], bounce_demo[0][1]])
plt.show()
"""


def bilinear_interpolation(x, y, x_mesh, y_mesh, values):
    """Given `x_mesh`, `y_mesh` for a rectangular grid, and `values` representing points at each value (`x_mesh`, `y_mesh`), give the bilinear interpolation at the point (`x`, `y`)."""
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
