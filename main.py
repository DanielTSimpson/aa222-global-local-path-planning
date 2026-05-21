import numpy as np
import config as cfg
from environment import SearchEnv
from drone import Drone
from belief import Belief
from gps import GPS
import matplotlib.pyplot as plt


def initialize_drone(env, window_size):
    """Initialize drones at random positions that don't see the science initially
    
    Args:
        env: SearchEnv object
        window_size: the drone's observation window size
        
    Returns:
        Drone: finalized Drone object
    """
    drone = Drone(env)

    drone.drone_id = 1
    drone.window_size = window_size
    drone.gamma = cfg.GAMMA
    drone.exploration_bonus = cfg.EXPLORATION_BONUS 
    drone.movement_cost = cfg.MOVEMENT_COST
    drone.time_cost = cfg.TIME_COST    

    drone.position = np.array([env.grid_size - 2, env.grid_size - 2])
    drone.visited_cells = {(drone.position[0], drone.position[1])}
    drone.belief_state = Belief(env.grid_size)
    drone.science_found = drone.observe()
    drone.history = [drone.state]

    return drone


def simulate_pomdp(x:list = [], trial_num = 0, render=0, save_gif=False):
    """
    Run the complete Dec-POMDP multi-agent simulation
    Args:
        x: List of disturbances  
        render: Int to show the gridworld plot render (0 - Show nothing, 1 - Show status updates, 2 - Show all)
        save_gif: Boolean to save the set of plots as a .gif
        
    Returns:
        None
    """
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
    
    if cfg.ENABLE_WIND:
        env.wind_speed = cfg.WIND_SPEED
        env.wind_direction = cfg.WIND_DIRECTION
    else:
        env.wind_speed = 0.0
        env.wind_direction = 0.0
    if save_gif:
        env.record_frames = True

    ### Initialize the drone
    drone_window_size = cfg.OBSERVATION_WINDOW_SIZE
    drone = initialize_drone(env, drone_window_size)
    

    failure_mode = 2 # Default to "Out of Time"
    time_to_obj = 0

    # === Inject disturbances, x into Environment's Wind ===
    if cfg.ENABLE_WIND:
        env.wind_speed = np.random.normal(x[3], np.sqrt(x[4]))
        thetas = [np.arctan2(drone.position[1] - env.science_pos[1], drone.position[0] - env.science_pos[0])]
        mean_theta = np.mean(thetas)
        env.wind_direction = x[5] * 2 * np.pi * np.random.random() + (1 - x[5]) * np.random.normal(mean_theta, np.sqrt(x[6]))

    # Main simulation loop
    for i in range(N):
        # Bias the Dynamic Wind every 10 time steps to push the drones away from the science
        if cfg.ENABLE_WIND and i > 0 and i % 10 == 0:
            env.wind_speed = np.random.normal(x[3], np.sqrt(x[4]))
            thetas = np.arctan2(drone.position[1] - env.science_pos[1], drone.position[0] - env.science_pos[0])
            env.wind_direction = x[5] * 2 * np.pi * np.random.random() + (1 - x[5]) * np.random.normal(thetas, np.sqrt(x[6]))
            if render == 1 or render == 2:
                print(f"\tWind changed direction to {env.wind_direction*180/np.pi:.1f} degrees")

        if render == 2 or save_gif:
            env.render([drone])
            if render == 2:
                plt.pause(render_pause)
        
        # Check for budget failure (Mode 1)
        if drone.budget <= 0:
            failure_mode = 1
            if render == 1 or render == 2: print("\tFAILURE: Max budget exceeded")
            time_to_obj = i
            break
        
        # Check if science has been collected
        if env.science_collected:
            failure_mode = 0
            time_to_obj = i
            if render == 1 or render == 2:
                print(f"\tScience collected! Completed in {time_to_obj*dt} time units")
                if render == 2:
                    env.render([drone])
                    plt.pause(5)
            break
        
        # Dec-POMDP decision making and execution
        best_action = drone.decide_action_pomdp()
        drone.action(best_action)
        
        # Check for stuck failure (Mode 3)
        if drone.stuck_count >= 20:
            failure_mode = 3
            time_to_obj = i
            if render == 1 or render == 2:
                print("\tFAILURE: Drones got Stuck")
            break

    if failure_mode == 2 and (render == 1 or render == 2):
        print("\tFAILURE: Exceeded max sim time")
    
    if save_gif:
        gif_fps = int(2.0 / cfg.RENDER_PAUSE) if cfg.RENDER_PAUSE > 0 else 10
        env.save_gif(f"simulation_trial_{trial_num}.gif", fps=gif_fps)

    env.close()


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
    
    if save_gif:
        env.record_frames = True

    ### Initialize the drone
    drone_window_size = cfg.OBSERVATION_WINDOW_SIZE
    drone = initialize_drone(env, drone_window_size)

    ### Initialize the GPS
    gps = GPS(env, drone)
    reconstructed_path, drone_instructions = gps.a_star()
    
    if drone_instructions is None:
        if render: print("\tFAILURE: GPS A* failed to find a path.")
        return



    failure_mode = 2 # Default to "Out of Time"
    time_to_obj = 0

    for i in range(N):
        if render == 2 or save_gif:
            env.render([drone], path=reconstructed_path)
            if render == 2:
                plt.pause(render_pause)
        
        # Check for budget failure (Mode 1)
        if drone.budget <= 0:
            failure_mode = 1
            if render == 1 or render == 2: print("\tFAILURE: Max budget exceeded")
            time_to_obj = i
            break
        
        # Check if science has been collected
        if env.science_collected:
            failure_mode = 0
            time_to_obj = i
            if render == 1 or render == 2:
                print(f"\tScience collected! Completed in {time_to_obj*dt} time units")
                if render == 2:
                    env.render([drone], path=reconstructed_path)
                    plt.pause(5)
            break
        
        drone.action(drone_instructions[i]) if i < len(drone_instructions) else drone.action(1)
        
        # Check for stuck failure (Mode 3)
        if drone.stuck_count >= 20:
            failure_mode = 3
            time_to_obj = i
            if render == 1 or render == 2:
                print("\tFAILURE: Drones got Stuck")
            break

    if failure_mode == 2 and (render == 1 or render == 2):
        print("\tFAILURE: Exceeded max sim time")
    
    if save_gif:
        gif_fps = int(2.0 / cfg.RENDER_PAUSE) if cfg.RENDER_PAUSE > 0 else 10
        env.save_gif(f"simulation_trial_{trial_num}.gif", fps=gif_fps)

    env.close()


def optimize():
    # Placeholder for a future optimization scheme
    return None


if __name__ == '__main__':
    # TODO: Make the drone's observation window forward facing 
        # This implies TODO: Add orientation to the drone's state
    # TODO: Remove exploration POMDP and implement obstacle handling instead
    mu_p = cfg.MU_P
    #simulate_pomdp(x = mu_p.tolist(), render=2, save_gif=False)
    simulate_astar(render = 2)
    plt.show(block=True)
    optimize()