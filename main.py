import numpy as np
import config as cfg
from environment import SearchEnv
from drone import Drone
from gps import GPS
import matplotlib.pyplot as plt

np.random.seed(cfg.SEED)

def simulate(trial_num = 0, render=False, save_gif=False):
    ### Initialize simulation parameters
    t_0 = cfg.INITIAL_TIME
    dt = cfg.TIME_STEP
    t_f = cfg.MAX_SIMULATION_TIME
    render_pause = cfg.RENDER_PAUSE if render else 0.0
    N = int((t_f - t_0) / dt)
    
    failure_modes = {"SUCCESS": 0, "BUDGET FAILURE": 1, "TIME FAILURE": 2, "STUCK FAILURE": 3}

    ### Initialize environment
    env = SearchEnv(grid_size=cfg.GRID_SIZE)
    if save_gif: env.record_frames = True
    env.science_value = 10
    env.generate_obstacles(num_large = cfg.NUM_LARGE_OBSTACLES, num_small = cfg.NUM_SMALL_OBSTACLES, large_mu = cfg.LARGE_OBSTACLE_SIZE_MU, large_sigma = cfg.LARGE_OBSTACLE_SIZE_SIGMA, small_mu = cfg.SMALL_OBSTACLE_SIZE_MU, small_sigma = cfg.SMALL_OBSTACLE_SIZE_SIGMA)
    if cfg.SMALL_SCIENCE_ENABLED:
        env.generate_small_science(num_small_science = cfg.NUM_SMALL_SCIENCE, min_value = cfg.SMALL_SCIENCE_MIN_VALUE, max_value = cfg.SMALL_SCIENCE_MAX_VALUE)

    ### Initialize the GPS
    gps = GPS(env)

    ### Initialize the drone
    drone = Drone(env, gps)
    drone.movement_cost = cfg.MOVEMENT_COST
    drone.time_cost = cfg.TIME_COST
    drone.visited_cells.add(tuple(drone.position))

    ### Calculate initial global path
    reconstructed_path, drone_instructions = gps.a_star(drone.position)
    drone.global_path = reconstructed_path
    drone.global_instructions = drone_instructions
    drone.global_path_index = 0
    if drone_instructions is None:
        if cfg.VERBOSE_LOGGING: print("\tFAILURE: GPS A* failed to find a path.")
        stats = {"TOTAL COST": cfg.MAX_BUDGET, "TOTAL_TIME": 0, "SMALL_SCIENCE_VALUE": 0}
        return failure_modes["STUCK FAILURE"], stats

    ### Initialize Failure Trackers
    time_f = 0 # Final time
    failure_mode = failure_modes.get("TIME FAILURE") # Default failure mode if the loop finishes

    ### Run the Simulation
    for i in range(N):
        time_f = i * dt

        # Start rendering
        if render == True or save_gif:
            env.render(drone, path=drone.global_path)
            if render == True: plt.pause(render_pause) # Small delay between rendered frames
            while env.paused: # When Pause button is toggled, pause the environment
                env.render(drone, path=drone.global_path)
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

        # Update progress by finding the closest point on the global path
        if drone.global_path is not None:
            current_pos = np.array([drone.x, drone.y])
            distances = [float(np.linalg.norm(current_pos - np.array(p), ord=np.inf)) for p in drone.global_path]
            drone.global_path_index = np.argmin(distances)

        # Get the next action from the global plan, or default to 1 (collect) if finished
        if drone.global_instructions is not None and drone.global_path_index < len(drone.global_instructions):
            global_action = drone.global_instructions[drone.global_path_index]
        else:
            global_action = 1
            
        # Check if local planning is required to collect small science or get unstuck
        if drone.local_optimizer_needed():
            if cfg.VERBOSE_LOGGING: print(f"Running Optimizer: {cfg.LOCAL_PLANNER_TYPE}")
            action = drone.local_optimizer.choose_action(drone, gps, env)
        else:
            action = global_action
        drone.action(action)


    # Monte Carlo vars
    total_cost = cfg.MAX_BUDGET - drone.budget
    total_time = time_f
    small_science_value = drone.small_science_collected_value

    # Check for time failure
    if failure_mode == failure_modes.get("TIME FAILURE") and cfg.VERBOSE_LOGGING:
        print("\tFAILURE: Exceeded max sim time")
    
    ## Close the Simulation
    if save_gif:
        gif_fps = 10
        env.close(save_gif=save_gif, filename=f"simulation_trial_{trial_num}.gif", fps=gif_fps)
    else:
        env.close()

    stats = {"TOTAL COST": total_cost, "TOTAL_TIME": total_time, "SMALL_SCIENCE_VALUE": small_science_value}

    return failure_mode, stats


if __name__ == '__main__':
    simulate(render = True, save_gif=True)
    plt.show(block=True)