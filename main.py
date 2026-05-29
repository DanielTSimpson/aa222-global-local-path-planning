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

def simulate(trial_num = 0, render=False, save_gif=False):
    ### Initialize simulation parameters
    t_0 = cfg.INITIAL_TIME
    dt = cfg.TIME_STEP
    t_f = cfg.MAX_SIMULATION_TIME
    render_pause = cfg.RENDER_PAUSE if render else 0.0
    N = int((t_f - t_0) / dt)

    ### Initialize environment
    env = SearchEnv(grid_size=cfg.GRID_SIZE)
    if save_gif: env.record_frames = True
    env.science_value = 10
    env.generate_obstacles(num_large = cfg.NUM_LARGE_OBSTACLES, num_small = cfg.NUM_SMALL_OBSTACLES, large_mu = cfg.LARGE_OBSTACLE_SIZE_MU, large_sigma = cfg.LARGE_OBSTACLE_SIZE_SIGMA, small_mu = cfg.SMALL_OBSTACLE_SIZE_MU, small_sigma = cfg.SMALL_OBSTACLE_SIZE_SIGMA)
    if cfg.SMALL_SCIENCE_ENABLED:
        env.generate_small_science(num_small_science = cfg.NUM_SMALL_SCIENCE, min_value = cfg.SMALL_SCIENCE_MIN_VALUE, max_value = cfg.SMALL_SCIENCE_MAX_VALUE)

    ### Initialize the drone
    drone = initialize_drone(env)

    ### Initialize the GPS
    gps = GPS(env, drone)
    reconstructed_path, drone_instructions = gps.a_star()
    drone.global_path = reconstructed_path
    drone.global_path_index = 0
    if drone_instructions is None:
        if cfg.VERBOSE_LOGGING: print("\tFAILURE: GPS A* failed to find a path.")
        return

    ### Initialize Failure Trackers
    failure_modes = {"SUCCESS": 0, "BUDGET FAILURE": 1, "TIME FAILURE": 2,"STUCK FAILURE": 3} # 0 = success, 1 = budget failure, 
    time_f = 0 # Final time
    failure_mode = failure_modes.get("TIME FAILURE") # Default failure mode if the loop finishes

    ### Run the Simulation
    for i in range(N):
        time_f = i * dt

        # Start rendering
        if render == True or save_gif:
            env.render(drone, path=reconstructed_path)
            if render == True: plt.pause(render_pause) # Small delay between rendered frames
            while env.paused: # When Pause button is toggled, pause the environment
                env.render(drone, path=reconstructed_path)
                plt.pause(0.1)
        
        # Check for budget failure
        if drone.budget <= 0:
            failure_mode = failure_modes.get("BUDGET FAILURE")
            if cfg.VERBOSE_LOGGING: print("\tFAILURE: Max budget exceeded")
            break
        
        # Check if science has been collected
        if env.science_collected:
            failure_mode = failure_modes.get("SUCCESS")
            if cfg.VERBOSE_LOGGING: print(f"\tScience collected! Completed in {time_f*dt} time units")
            if render == True:
                env.render(drone, path=reconstructed_path)
                plt.pause(5)
            break

        # Check for stuck failure
        if drone.stuck_count >= 20:
            failure_mode = failure_modes.get("STUCK FAILURE")
            if cfg.VERBOSE_LOGGING: print("\tFAILURE: Drones got Stuck")
            break

        # Calculates the distance from the agent to the closest point on the global path
        
        if hasattr(drone, "global_path") and drone.global_path is not None:
            # Using np.linalg.norm so we can swap between different distance calc methods and be consistent w/ A* method
            drone.global_path_index = min(range(len(drone.global_path)), key = lambda k: float(np.linalg.norm(np.array([drone.x, drone.y]) - np.array(drone.global_path[k][0], drone.global_path[k][1]), ord=np.inf)))
        # Calculate the next global action based on path progress, rather than simulation time
        global_action = drone_instructions[drone.global_path_index] if getattr(drone, "global_path_index", 0) < len(drone_instructions) else 1
        # then, we see if this leads to one of the local planning optimizers needing to kick in (via an obstacle, science, etc.)
        if drone.local_optimizer_needed(global_action):
            action = drone.local_optimizer.choose_action(drone, gps, env)
        else:
            action = global_action
        drone.action(action)

    # Monte Carlo vars
    total_cost = cfg.MAX_BUDGET - drone.budget
    total_time = time_f
    small_science_value = drone.small_science_collected_value

    # Check for time failure
    if failure_mode == 2 and cfg.VERBOSE_LOGGING:
        print("\tFAILURE: Exceeded max sim time")
    
    ## Close the Simulation
    if save_gif:
        gif_fps = int(2.0 / cfg.RENDER_PAUSE) if cfg.RENDER_PAUSE > 0 else 10
        env.save_gif(f"simulation_trial_{trial_num}.gif", fps=gif_fps)
        env.close(save_gif=save_gif, filename=f"simulation_trial_{trial_num}.gif", fps=gif_fps)
    else:
        env.close()

    stats = {"TOTAL COST": total_cost, "TOTAL_TIME": total_time, "SMALL_SCIENCE_VALUE": small_science_value}

    return failure_mode, stats


if __name__ == '__main__':
    simulate(render = True, save_gif=True)
    plt.show(block=True)