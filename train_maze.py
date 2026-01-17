import numpy as np
import datetime
from brownian_motion_obstacle_course import brownian_motion_obstacle_course


def train_next_generation(
        # Simulation parameters:
        # Environmental parameters:
        obstacles, exits, x_mesh, y_mesh,
        # Time parameters:
        time_limit, timestep, simulations,
        # Motion parameters:
        start_point, drift_function, diffusion=1, speed_limit=np.inf, rolling_speed_period=1,
        # Drift function training parameters:
        # mode: `'time'` gives running averages over time, `'steps'` gives running averages over steps
        memory_retention=0.8, running_period=1, running_mode="time",
        collision_penalty=0,
        # UI settings:
        print_progress=True):
    """Train a Brownian motion by running simulations from an initial set of simulation parameters fed into the `brownian_motion_obstacle_course` function to collect training data from a given number of `simulations`. Training parameters include `memory_retention` to show what percentage of previous drift function is retained. Parameter `running_period` describes smoothing of trajectories in training data used to average over this period to train drift functions. Collisions penalized by `collision_penalty` parameter to introduce dividing (1 + `collision_penalty` * `number_of_collisions`) from orginal training score."""
    # Step 1: Run simulations to collect training data

    generation = drift_function['generation']

    training_data = []

    for i in range(simulations):
        if print_progress:
            print('Running iteration {} of {}...'.format(i + 1, simulations))

        data_point = brownian_motion_obstacle_course(obstacles=obstacles, exits=exits, x_mesh=x_mesh, y_mesh=y_mesh,
                                                     time_limit=time_limit, timestep=timestep,
                                                     start_point=start_point, drift=drift_function[
                                                         'drift_value'], diffusion=diffusion,
                                                     speed_limit=speed_limit, rolling_speed_period=rolling_speed_period)

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

        if running_mode == 'time':
            running_steps = int(np.ceil(running_period / timestep))
        elif running_mode == 'steps':
            running_steps = running_period

        for i in range(data_point['path'].shape[1] - 1):
            start = data_point['path'][:, i]
            if i + running_steps < data_point['path'].shape[1]:
                end = data_point['path'][:, i + running_steps]
                steps = running_steps
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
        time_limit=1000, steps=1000, max_step_size=1,
        # Motion parameters:
        start_point=np.array([0, 0]), drift_function=None, diffusion=1, speed_limit=np.inf, rolling_speed_period=1,
        # Drift function training parameters:
        # mode: `'time'` gives running averages over time, `'steps'` gives running averages over steps
        memory_retention=0.8, running_period=1, running_mode='time',
        collision_penalty=0,
        # Generation training parameters:
        # Target goal (stop when achieved):
        target_time=100, target_success_rate=0.9,
        # Adjustment parameters:
        new_time_fraction=0.7, level_up_success_rate=0.95, level_down_success_rate=0.3, level_down_time_multiplier=1.2,
        # Training limits:
        training_generations=100, training_time_limit=3600, simulations=100,
        # UI Settings:
        print_progress=True):
    """Implement the full training process for a single obstacle course with initial parameters. Run a given number of `simulations` up to a number of `training_generations`, until all generations are used or `training_time_limit` (in seconds) is reached. Goal is to achieve a `target_success_rate` of all trajectories under a `target_time`, at which training stops. In training, monitor success rates of current training generation: if `level_up_success_rate` of all trajectories fall within current time limit, level up to reduce time limit according to `new_time_fraction` quartile. If less than `level_down_success_rate` of trajectories fall within time limit, level down by multiplying time limit by `level_down_time_multiplier`. Other parameters are same as in `train_next_generation` function. UPCOMING: Data also includes a log of events."""

    # Initialize data
    drift_functions = []
    training_data = []

    # First drift function
    if drift_function == None:
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
    for iteration in range(training_generations):
        current_time = datetime.datetime.now()
        elapsed_time = current_time - start_time
        if elapsed_time.total_seconds() > training_time_limit:
            # Stop training after the specified time limit
            print('Time limit exceeded. Training stopped.')
            break

        generation = drift_function['generation']

        if print_progress:
            print('Training generation {}...'.format(generation))

        timestep = min(time_limit / steps, max_step_size)

        results = train_next_generation(obstacles=obstacles, exits=exits, x_mesh=x_mesh, y_mesh=y_mesh,
                                        time_limit=time_limit, timestep=timestep, simulations=simulations,
                                        start_point=start_point, drift_function=drift_function, diffusion=diffusion,
                                        speed_limit=speed_limit, rolling_speed_period=rolling_speed_period,
                                        memory_retention=memory_retention, running_period=running_period, running_mode=running_mode,
                                        collision_penalty=collision_penalty,
                                        print_progress=print_progress)

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
