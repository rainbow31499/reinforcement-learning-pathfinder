import numpy as np
import datetime
from brownian_motion_obstacle_course import brownian_motion_obstacle_course


def train_next_generation(  # Simulation parameters:
        obstacles, exits, drift_function, x_mesh, y_mesh, start_point, time_limit, timestep, diffusion, speed_limit, rolling_speed_period, simulations, print_progress,
        # Training parameters:
        memory_retention, running_period, collision_penalty):
    # Step 1: Run simulations to collect training data

    generation = drift_function['generation']

    training_data = []

    for i in range(simulations):
        if print_progress:
            print('Running iteration {} of {}...'.format(i + 1, simulations))

        data_point = brownian_motion_obstacle_course(start_point=start_point,
                                                     timestep=timestep,
                                                     time_limit=time_limit,
                                                     obstacles=obstacles,
                                                     exits=exits,
                                                     x_mesh=x_mesh, y_mesh=y_mesh,
                                                     drift=drift_function['drift_value'],
                                                     diffusion=diffusion,
                                                     speed_limit=speed_limit,
                                                     rolling_speed_period=rolling_speed_period)

        data_point['generation'] = generation

        training_data.append(data_point)

    # Step 2: Train drift function based on training data

    drift_totals = drift_function['totals'] * memory_retention
    drift_weights = drift_function['weights'] * memory_retention

    for data_point in training_data:
        if data_point['success'] == False:
            weight = 0
        else:
            weight = 1 / data_point['time_taken'] / \
                (1 + collision_penalty * data_point['collisions'])

        for i in range(data_point['path'].shape[1] - 1):
            start = data_point['path'][:, i]
            if i + running_period < data_point['path'].shape[1]:
                end = data_point['path'][:, i + running_period]
                steps = running_period
            else:
                end = data_point['path'][:, -1]
                steps = data_point['path'].shape[1] - 1 - i
            direction = end - start
            x_index = np.argmin(np.absolute(x_mesh - start[0]))
            y_index = np.argmin(np.absolute(y_mesh - start[1]))

            drift_weights[x_index, y_index] += weight
            drift_totals[x_index, y_index] += direction / \
                (timestep * steps) * weight

    drift = np.zeros(shape=drift_totals.shape)
    for x_index in range(drift.shape[0]):
        for y_index in range(drift.shape[1]):
            if drift_weights[x_index, y_index] != 0:
                drift[x_index, y_index, :] = drift_totals[x_index,
                                                          y_index, :] / drift_weights[x_index, y_index]

    new_drift_function = {'totals': drift_totals,
                          'weights': drift_weights,
                          'drift_value': drift,
                          'x_mesh': x_mesh,
                          'y_mesh': y_mesh,
                          'generation': generation + 1,
                          'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                          'obstacles': obstacles,
                          'exits': exits
                          }

    return {"training_data": training_data, "new_drift_function": new_drift_function}
