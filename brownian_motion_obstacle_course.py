import numpy as np
from geom_functions import *


def brownian_motion_obstacle_course(start_point,
                                    timestep,
                                    time_limit,
                                    obstacles,
                                    exits,
                                    x_mesh,
                                    y_mesh,
                                    drift,
                                    diffusion,
                                    speed_limit,
                                    rolling_speed_period):

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
                if intersect(exit[:, segment], exit[:, segment + 1], position, candidate_position):
                    success = True
                    time_taken = time
        position = candidate_position
    return {'path': path, 'success': success, 'time_taken': time_taken, 'collisions': collisions}
