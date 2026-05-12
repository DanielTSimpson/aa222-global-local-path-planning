import numpy as np
import config as cfg
from environment import SearchEnv
from drone import Drone
from belief import Belief
import matplotlib.pyplot as plt


def initialize_drones(num_drones, env, window_size):
    """Initialize drones at random positions that don't see the science initially
    
    Args:
        num_drones: number of drones to create
        env: SearchEnv object
        window_size: the drone's observation window size
        
    Returns:
        list: list of Drone objects
    """
    drones = []

    for drone_id in range(num_drones):
        drone = Drone(env)

        # If drone observes the science upon creation, reshuffle
        while drone.science_found:
            drone = Drone(env)

        drone.drone_id = drone_id
        drone.window_size = window_size
        drone.gamma = cfg.GAMMA
        drone.exploration_bonus = cfg.EXPLORATION_BONUS 
        drone.movement_cost = cfg.MOVEMENT_COST
        drone.time_cost = cfg.TIME_COST    

        drones.append(drone)

    return drones


def run_simulation(x:list = [], trial_num = 0, render=0, save_gif=False):
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

    # Initialize environment
    env = SearchEnv()
    env.science_value = 10
    env.grid_size = cfg.GRID_SIZE

    env.generate_obstacles(num_large = cfg.NUM_LARGE_OBSTACLES, num_small = cfg.NUM_SMALL_OBSTACLES, large_mu = cfg.LARGE_OBSTACLE_SIZE_MU, large_sigma = cfg.LARGE_OBSTACLE_SIZE_SIGMA, small_mu = cfg.SMALL_OBSTACLE_SIZE_MU, small_sigma = cfg.SMALL_OBSTACLE_SIZE_SIGMA)
    
    if cfg.ENABLE_WIND:
        env.wind_speed = cfg.WIND_SPEED
        env.wind_direction = cfg.WIND_DIRECTION
    else:
        env.wind_speed = 0.0
        env.wind_direction = 0.0
    if save_gif:
        env.record_frames = True

    # Initialize the drones
    drone_window_size = cfg.OBSERVATION_WINDOW_SIZE
    drones = initialize_drones(cfg.NUM_DRONES, env, drone_window_size)

    failure_mode = 2 # Default to "Out of Time"
        
    drone.position = np.array([new_x, new_y])
    drone.visited_cells = {(new_x, new_y)}
    drone.belief_state = Belief(env.grid_size)
    drone.science_found = drone.observe()
    drone.history = [drone.state]

    # === Inject disturbances, x into Environment's Wind ===
    if cfg.ENABLE_WIND:
        env.wind_speed = np.random.normal(x[3], np.sqrt(x[4]))
        thetas = [np.arctan2(drone.position[1] - env.science_pos[1], drone.position[0] - env.science_pos[0]) for drone in drones]
        mean_theta = np.mean(thetas)
        env.wind_direction = x[5] * 2 * np.pi * np.random.random() + (1 - x[5]) * np.random.normal(mean_theta, np.sqrt(x[6]))

    # Main simulation loop
    for i in range(N):
        # Bias the Dynamic Wind every 10 time steps to push the drones away from the science
        if cfg.ENABLE_WIND and i > 0 and i % 10 == 0:
            env.wind_speed = np.random.normal(x[3], np.sqrt(x[4]))
            thetas = [np.arctan2(drone.position[1] - env.science_pos[1], drone.position[0] - env.science_pos[0]) for drone in drones]
            mean_theta = np.mean(thetas)
            env.wind_direction = x[5] * 2 * np.pi * np.random.random() + (1 - x[5]) * np.random.normal(mean_theta, np.sqrt(x[6]))
            if render == 1 or render == 2:
                print(f"\tWind changed direction to {env.wind_direction*180/np.pi:.1f} degrees")

        if render == 2 or save_gif:
            env.render(drones)
            if render == 2:
                plt.pause(render_pause)
        
        # Check for budget failure (Mode 1)
        min_budget = min(d.budget for d in drones)
        if min_budget <= 0:
            failure_mode = 1
            if render == 1 or render == 2: print("\tFAILURE: Max budget exceeded")
            time_to_obj = i
            break
        
        # Check if science has been collected
        if env.science_collected:
            failure_mode = 0
            time_to_obj = i
            if render == 1 or render == 2:
                print(f"\tScience collected!")
                if render == 2:
                    env.render(drones)
                    plt.pause(5)
            break
        
        # Dec-POMDP decision making and execution
        for drone in drones:
            best_action = drone.decide_action_pomdp()
            drone.action(best_action)
        
        # Check for stuck failure (Mode 3)
        if any(d.stuck_count >= 20 for d in drones):
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
    # TODO: Implement obstacle handling in Drone's POMDP
    # TODO: Implement A-star to generate a global path
    # TODO: Implement small obstacles
    # TODO: Make the drone's observation window forward facing 
        # This implies TODO: Add orientation to the drone's position
    mu_p = cfg.MU_P
    run_simulation(x = mu_p.tolist(), render=2, save_gif=False)
    optimize()