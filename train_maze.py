import numpy as np
import datetime
from brownian_motion_obstacle_course import brownian_motion_obstacle_course


def train_next_generation(  # Simulation parameters:
        obstacles, exits, drift_function, x_mesh, y_mesh, start_point, time_limit, timestep, diffusion, speed_limit, rolling_speed_period, simulations, print_progress,
        # Training parameters:
        memory_retention, running_period, collision_penalty):
    """Train a Brownian motion by running simulations to collect training data from a given number of `simulations`, """
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


def train_maze_simulation(
        # Simulation parameters (initial):
        # Environmental parameters:
        obstacles, exits, x_mesh, y_mesh,
        # Time parameters:
        time_limit, steps=1000, min_step_size=1,
        # Motion parameters:
        start_point=np.array([0, 0]), diffusion=1, speed_limit=3, rolling_speed_period=20,
        # Training parameters:
        memory_retention=0.8, running_time=1, collision_penalty=0,
        # Target goal, stop when achieved:
        target_time=100, target_success_rate=0.95,
        # Adjustment parameters:
        new_time_fraction=0.7, level_up_success_rate=0.95, level_down_success_rate=0.3, level_down_time_multiplier=1.2,
        # Training limits:
        training_iterations=100, training_time_limit=3600, simulations=100,
        print_progress=True):
    """Implement the full training process for a single obstacle course with initial parameters."""

    # Initialize data
    drift_functions = []
    training_data = []

    # First drift function
    drift_totals_zero = np.zeros((len(x_mesh), len(y_mesh), 2))
    drift_weights_zero = np.zeros((len(x_mesh), len(y_mesh)))
    drift_zero = np.zeros((len(x_mesh), len(y_mesh), 2))

    drift_function = {'totals': drift_totals_zero,
                      'weights': drift_weights_zero,
                      'drift_value': drift_zero,
                      'x_mesh': x_mesh,
                      'y_mesh': y_mesh,
                      'generation': 0,
                      'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                      'obstacles': obstacles,
                      'exits': exits
                      }

    drift_functions.append(drift_function)

    # Training procedure
    start_time = datetime.datetime.now()
    for iteration in range(training_iterations):
        current_time = datetime.datetime.now()
        elapsed_time = current_time - start_time
        if elapsed_time.total_seconds() > training_time_limit:
            # Stop training after the specified time limit
            print('Time limit exceeded. Training stopped.')
            break

        generation = drift_function['generation']

        if print_progress:
            print('Training generation {}...'.format(generation))

        timestep = min(time_limit / steps, min_step_size)

        running_period = int(np.ceil(running_time / timestep))

        results = train_next_generation(drift_function=drift_function,
                                        obstacles=obstacles,
                                        exits=exits,
                                        x_mesh=x_mesh,
                                        y_mesh=y_mesh,
                                        start_point=start_point,
                                        time_limit=time_limit,
                                        timestep=timestep,
                                        simulations=simulations,
                                        print_progress=print_progress,
                                        diffusion=diffusion,
                                        speed_limit=speed_limit,
                                        rolling_speed_period=rolling_speed_period,
                                        memory_retention=memory_retention,
                                        running_period=running_period,
                                        collision_penalty=collision_penalty)

        current_training_data = results['training_data']
        new_drift_function = results['new_drift_function']

        under_target_time = 0
        for data_point in current_training_data:
            if data_point['success'] == True and data_point['time_taken'] <= target_time:
                under_target_time += 1

        if under_target_time / simulations >= target_success_rate:
            # Stop training when target is achieved
            if print_progress:
                print('Target achieved. Training stopped.')
            break

        # Save the data for records

        drift_functions.append(new_drift_function)
        training_data.extend(current_training_data)

        # Set new parameters for next generation based on analysis of training data

        drift_function = new_drift_function

        successes = 0
        for data_point in current_training_data:
            if data_point['success'] == True:
                successes += 1

        times = [(data_point['time_taken'] if data_point['time_taken']
                  != None else np.inf) for data_point in current_training_data]

        average_time = np.mean([time for time in times if time != np.inf])
        success_rate = successes / len(current_training_data)
        if print_progress:
            print('Generation {}: Success rate {}. Average time: {:.2f}'.format(
                generation, success_rate, average_time))

        if success_rate >= level_up_success_rate:
            time_limit = np.quantile(times, new_time_fraction)
            if print_progress:
                print('Level up. New time limit decreased to {} to reinforce better times'.format(
                    int(time_limit)))

        if success_rate <= level_down_success_rate:
            time_limit *= level_down_time_multiplier
            if print_progress:
                print('Level down. New time limit increased to {} to allow more successes'.format(
                    int(time_limit)))

    print('Training complete.')

    return {'drift_functions': drift_functions,
            'training_data': training_data}
