import numpy as np
import config as cfg
from environment import SearchEnv
from drone import Drone
from gps import GPS
import matplotlib.pyplot as plt

np.random.seed(cfg.SEED)

def initialize_drone(env):
    """Initialize drones at random positions that don't see the science initially
    
    Args:
        env: SearchEnv object
        
    Returns:
        Drone: finalized Drone object
    """
    drone = Drone(env)
    drone.movement_cost = cfg.MOVEMENT_COST
    drone.time_cost = cfg.TIME_COST    
    drone.visited_cells.add(tuple(drone.position))

    return drone

def simulate_astar(trial_num = 0, render=0, save_gif=False):
    # Initialize simulation parameters
    t_0 = cfg.INITIAL_TIME
    dt = cfg.TIME_STEP
    t_f = cfg.MAX_SIMULATION_TIME
    render_pause = cfg.RENDER_PAUSE if render else 0.0
    N = int((t_f - t_0) / dt)

    ### Initialize environment
    env = SearchEnv(grid_size=cfg.GRID_SIZE)
    env.science_value = 10

    env.generate_obstacles(num_large = cfg.NUM_LARGE_OBSTACLES, num_small = cfg.NUM_SMALL_OBSTACLES, large_mu = cfg.LARGE_OBSTACLE_SIZE_MU, large_sigma = cfg.LARGE_OBSTACLE_SIZE_SIGMA, small_mu = cfg.SMALL_OBSTACLE_SIZE_MU, small_sigma = cfg.SMALL_OBSTACLE_SIZE_SIGMA)
    
    if cfg.SMALL_SCIENCE_ENABLED:
        env.generate_small_science(num_small_science = cfg.NUM_SMALL_SCIENCE, min_value = cfg.SMALL_SCIENCE_MIN_VALUE, max_value = cfg.SMALL_SCIENCE_MAX_VALUE)
    
    if save_gif:
        env.record_frames = True

    ### Initialize the drone
    drone = initialize_drone(env)

    ### Initialize the GPS
    gps = GPS(env, drone)
    reconstructed_path, drone_instructions = gps.a_star()
    
    drone.global_path = reconstructed_path
    drone.global_path_index = 0
    
    if drone_instructions is None and cfg.VERBOSE_LOGGING:
        if render: print("\tFAILURE: GPS A* failed to find a path.")
        return

    failure_mode = 2 # Default to "Out of Time"
    time_to_obj = 0

    for i in range(N):
        if render == 2 or save_gif:
            env.render(drone, path=reconstructed_path)
            if render == 2:
                plt.pause(render_pause)
        
        while env.paused:
            env.render(drone, path=reconstructed_path)
            plt.pause(0.1)
        
        # Check for budget failure (Mode 1)
        if drone.budget <= 0:
            failure_mode = 1
            if render == 1 or render == 2 and cfg.VERBOSE_LOGGING: print("\tFAILURE: Max budget exceeded")
            time_to_obj = i
            break
        
        # Check if science has been collected
        if env.science_collected:
            failure_mode = 0
            time_to_obj = i
            if render == 1 or render == 2:
                if cfg.VERBOSE_LOGGING:
                    print(f"\tScience collected! Completed in {time_to_obj*dt} time units")
                if render == 2:
                    env.render(drone, path=reconstructed_path)
                    plt.pause(5)
            break

        # calculates our distance from the end to fuel decision making in our local optimizer
        if hasattr(drone, "global_path"):
            drone.global_path_index = min(range(len(drone.global_path)), key = lambda k: abs(drone.x - drone.global_path[k][0]) + abs(drone.y - drone.global_path[k][1]))
        
        # here, the global action is computed via A*
        global_action = drone_instructions[i] if i < len(drone_instructions) else 1
        
        # then, we see if this leads to one of the local planning optimizers needing to kick in (via an obstacle, science, etc.)
        if drone.local_optimizer_needed(global_action):
            action = drone.local_optimizer.choose_action(drone, env)
        else:
            action = global_action
        
        drone.action(action)
        
        # Check for stuck failure (Mode 3)
        if drone.stuck_count >= 20:
            failure_mode = 3
            time_to_obj = i
            if render == 1 or render == 2 and cfg.VERBOSE_LOGGING:
                print("\tFAILURE: Drones got Stuck")
            break

    # hi, this section is for the monte carlo scheme
    total_cost = cfg.MAX_BUDGET - drone.budget
    total_time = time_to_obj * dt
    small_science_value = drone.small_science_collected_value

    if failure_mode == 2 and (render == 1 or render == 2) and cfg.VERBOSE_LOGGING:
        print("\tFAILURE: Exceeded max sim time")
    
    if save_gif:
        gif_fps = int(2.0 / cfg.RENDER_PAUSE) if cfg.RENDER_PAUSE > 0 else 10
        env.save_gif(f"simulation_trial_{trial_num}.gif", fps=gif_fps)
        env.close(save_gif=save_gif, filename=f"simulation_trial_{trial_num}.gif", fps=gif_fps)
    else:
        env.close()

    return failure_mode, total_cost, total_time, small_science_value


def optimize():
    # Placeholder for a future optimization scheme
    return None


if __name__ == '__main__':
    simulate_astar(render = 2, save_gif=True)
    plt.show(block=True)
    optimize()