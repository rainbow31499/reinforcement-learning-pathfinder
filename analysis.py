import matplotlib.pyplot as plt


def plot_trajectory_results(obstacles, exits, training_data, trials=None):
    fig, ax = plt.subplots()
    ax.set_aspect(1)
    for obstacle in obstacles:
        ax.plot(obstacle[0], obstacle[1], 'r', linewidth=3)
    for exit in exits:
        ax.plot(exit[0], exit[1], 'g', linewidth=3)
    if trials == None:
        trials = range(len(training_data))
    for trial in trials:
        result = training_data[trial]
        if result['success'] == True:
            ax.plot(result['path'][0], result['path']
                    [1], 'lime', linewidth=0.1)
        elif result['success'] == False:
            ax.plot(result['path'][0], result['path']
                    [1], 'darkorange', linewidth=0.1)


def plot_drift_field(drift_function, obstacles, exits):
    drift_value = drift_function['drift_value']
    x_mesh = drift_function['x_mesh']
    y_mesh = drift_function['y_mesh']

    fig, ax = plt.subplots()

    fig.suptitle('Drift Field For Generation {}'.format(
        drift_function['generation']))

    ax.set_aspect(1)
    for obstacle in obstacles:
        ax.plot(obstacle[0], obstacle[1], 'r', linewidth=3)
    for exit in exits:
        ax.plot(exit[0], exit[1], 'g', linewidth=3)
    ax.quiver(x_mesh, y_mesh, np.transpose(drift_value, (1, 0, 2))[
              :, :, 0], np.transpose(drift_value, (1, 0, 2))[:, :, 1])
    plt.show()
