import numpy as np
from geom_functions import *


def brownian_motion_obstacle_course(
        # Environmental parameters:
        obstacles, exits, x_mesh, y_mesh,
        # Time parameters:
        time_limit, timestep,
        # Motion parameters:
        start_point, drift, diffusion=1, speed_limit=np.inf, rolling_speed_period=1):
    """Simulate a 2D Brownian motion obstacle course in a maze given a set of `obstacles` from a `start_point`, to see if trajectory crosses `exits` within `time_limit`. Run the simulation by stepping times by `timestep`, according to a defined `drift` function that operates bilinearly on a grid defined by `x_mesh` and `y_mesh`, and a diffusion constant for the Brownian motion element. If `speed_limit` is set, only allow paths that lie within speed limit according to a `rolling_speed_period` indicating number of steps over which to average speed measurements."""

    success = False
    time_taken = None
    time = 0
    position = start_point
    times = np.zeros(0)
    path = np.zeros((2, 0))
    collisions = 0
    while time <= time_limit:
        times = np.append(times, time)
        path = np.append(path, np.transpose(np.array([position])), axis=1)
        if success:
            break

        time += timestep
        while True:
            force = bilinear_interpolation(
                position[0], position[1], x_mesh, y_mesh, drift)
            candidate_position = position + \
                np.random.standard_normal(
                    2) * diffusion * np.sqrt(timestep) + force * timestep
            if path.shape[1] < rolling_speed_period:
                old_position = path[:, 0]
                steps = path.shape[1]
            else:
                old_position = path[:, path.shape[1] - rolling_speed_period]
                steps = rolling_speed_period
            if np.sqrt(np.dot(candidate_position - old_position, candidate_position - old_position)) / (timestep * steps) > speed_limit:
                speed_limit_exceeded = True
            else:
                speed_limit_exceeded = False
            if collides(obstacles, position, candidate_position):
                collision = True
                collisions += 1
            else:
                collision = False
            if (not collision) and (not speed_limit_exceeded):
                break
        for exit in exits:
            for segment in range(exit.shape[1] - 1):
                if intersect(exit[:, segment], exit[:, segment + 1], position, candidate_position)[0]:
                    success = True
                    time_taken = time
        position = candidate_position
    return {'path': path, 'success': success, 'time_taken': time_taken, 'collisions': collisions}


"""
import matplotlib.pyplot as plt

obstacles = [np.array([[-4, -4, 5, 5, 1, 1, -1, -1, 5, 5, 12, 12, 10, 10, 7, 7, -6, -6],
                      [-8, 4, 4, -5, -5, 1, 1, -7, -7, -11, -11, 4, 4, -9, -9, 6, 6, -8]])]
exits = [np.array([[-4, -6], [-8, -8]])]
x_mesh = np.linspace(-8, 14)
y_mesh = np.linspace(-12, 8)
start_point = np.array([0, 0])
time_limit = 2000
drift_totals_zero = np.zeros((len(x_mesh), len(y_mesh), 2))
drift_weights_zero = np.zeros((len(x_mesh), len(y_mesh)))
drift_zero = np.zeros((len(x_mesh), len(y_mesh), 2))

sim = brownian_motion_obstacle_course(start_point=start_point,
                                      timestep=1,
                                      time_limit=time_limit,
                                      obstacles=obstacles,
                                      exits=exits,
                                      x_mesh=x_mesh, y_mesh=y_mesh,
                                      drift=drift_zero,
                                      diffusion=1,
                                      speed_limit=3,
                                      rolling_speed_period=10)
print(sim)
plt.plot(sim['path'][0], sim['path'][1])
plt.show()
"""
