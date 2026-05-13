"""
Environment module for multi-agent science search simulation
Handles rendering, step execution, and reward computation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from scipy.signal import convolve2d

from gymnasium import Env
import imageio

class SearchEnv(Env):
    """Multi-agent search environment with Dec-POMDP framework"""
    def __init__(self, grid_size=20):
        self.grid_size = grid_size # The side length of the square grid-world
        self.wind_speed = 0.0 # The probability of the wind moving drones
        self.wind_direction = 0.0 # The direction the wind would bias drone movement in radians
        self.terrain = {"FREE": 0, "OBJECTIVE": 1, "BUFFER ZONE": 2, "LARGE OBSTACLE": 3, "SMALL OBSTACLE": 4}
        self.the_grid = np.zeros((self.grid_size, self.grid_size), dtype = int) # NpArray of the grid

        # Objective Definition
        self.science_pos = (1, 1) # Define the objective location in one corner 
        self.the_grid[self.science_pos[0] - 1 : self.science_pos[0] + 1, 
                      self.science_pos[1] - 1 : self.science_pos[1] + 1] = self.terrain["BUFFER ZONE"] # Add a buffer zone around the objective
        self.the_grid[self.science_pos[0], self.science_pos[1]] = self.terrain["OBJECTIVE"] # Add the objective in the grid world
        self.science_value = np.random.randint(1, 10) # how important the science objective is
        self.science_found = False
        self.science_collected = False

        # Start zone definition
        self.the_grid[self.grid_size - 5 : self.grid_size, 
                      self.grid_size - 5 : self.grid_size] = self.terrain["BUFFER ZONE"]

        self.patches = []
        self.fig, self.ax = None, None
        self.status_texts = []
        self.frames = []
        self.record_frames = False


    def reset_obstacles(self):
        # clears existing obstacles from the environment, called before generating a new map
        self.the_grid = np.zeros((self.grid_size, self.grid_size), dtype = bool)
        self.large_obstacles = np.zeros((self.grid_size, self.grid_size), dtype = bool) 
        self.small_obstacles = np.zeros((self.grid_size, self.grid_size), dtype = bool)
        self.obstacle_grid = np.clip(self.large_obstacles + self.small_obstacles, 0, 1)
        self.obstacles = []


    def in_bounds(self, r, c):
        # checks if a given cell, with coordinates row r and column c, is inside the environment
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size
    

    def is_obstacle(self, r, c):
        # just returns whether or not the cell at row r and column c is an obstacle or free/protected space
        return (self.the_grid[r, c] == self.terrain["LARGE OBSTACLE"] or self.the_grid[r, c] == self.terrain["SMALL OBSTACLE"])
    

    def is_free(self, r, c):
        # returns whether or not the cell at row r and column c is a free space
        return self.the_grid[r, c] == self.terrain["FREE"]
    

    def is_buffer(self, r, c):
        # returns whether the cell is in a buffer zone
        return self.the_grid[r, c] == self.terrain["BUFFER ZONE"]


    def _sample_obstacle_size(self, mu, sigma, min_size = 1, max_size = None):
        # lets us draw obstacle sizes from the small or large obstacle size distributions we define in the config file
        if max_size is None:
            max_size = max(1, self.grid_size // 4) # makes it so we don't go a lil too crazy with oversized obstacles

        width = int(round(np.random.normal(mu, sigma)))
        height = int(round(np.random.normal(mu, sigma)))
        clipped = np.clip([width, height], min_size, max_size)
        return int(clipped[0]), int(clipped[1])
    

    def spawn_obstacle(self, obs_type, mu, sigma):
        w, h = self._sample_obstacle_size(mu, sigma)
        blocked_map = (self.the_grid != self.terrain["FREE"]).astype(int)
        footprint = np.ones((h, w), dtype=int)
        overlap_map = convolve2d(blocked_map, footprint, mode='valid')
        valid_y, valid_x = np.where(overlap_map == 0)
        if len(valid_y) == 0:
            return None
        random_idx = np.random.randint(len(valid_y))
        r = valid_y[random_idx]
        c = valid_x[random_idx]
        self.the_grid[r : r+h, c : c+w] = obs_type
        return (r, c, w, h)
    
    
    def generate_obstacles(self, num_large = 5, num_small = 15, large_mu = 4, large_sigma = 1.0, small_mu = 1.5, small_sigma = 0.5):
        [self.spawn_obstacle(self.terrain["LARGE OBSTACLE"], large_mu, large_sigma) for _ in range(num_large)]
        [self.spawn_obstacle(self.terrain["SMALL OBSTACLE"], small_mu, small_sigma) for _ in range(num_small)]
    

    def render(self, drones, path=None):
        grid = np.zeros((self.grid_size, self.grid_size))
        
        large_obs_val = self.terrain.get("LARGE OBSTACLE", self.terrain.get("LARGE OBSTACLE", 3))
        small_obs_val = self.terrain.get("SMALL OBSTACLE", self.terrain.get("SMALL OBSTACLE", 4))

        # Mark explored cells (visited or observed)
        for drone in drones:
            for (r, c) in drone.visited_cells:
                if self.the_grid[r, c] != large_obs_val and self.the_grid[r, c] != small_obs_val: # only mark cells as explored if they're not an obstacle -- otherwise we're overwriting those grids visually
                    grid[r, c] = 1
                    
        grid[self.the_grid == large_obs_val] = 7
        grid[self.the_grid == small_obs_val] = 8


        if not self.science_found:
            grid[tuple(self.science_pos)] = 2

        for idx, drone in enumerate(drones):
            grid[tuple(drone.position)] = idx + 3

        cmap = colors.ListedColormap(['#ffcccc', 'white', '#2ecc71', 'blue', 'green', 'orange', 'purple', 'grey', 'darkgreen'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 7))
            self.im = self.ax.imshow(grid, cmap=cmap, norm=norm)
            self.ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            self.ax.set_xlabel('Y Position')
            self.ax.set_ylabel('X Position')
            self.ax.set_title("Multi-Agent Science Objective Sear", fontsize=12, fontweight='bold')

            plt.ion()
            plt.show(block=False)
        else:
            self.im.set_data(grid)
        
        # first we clear away our old patches
        for p in self.patches:
            p.remove()
        self.patches.clear()
        
        # draw our science value label (how important the science is)
        if not self.science_found:
            sx, sy = self.science_pos
            assert self.ax is not None
            value_text = self.ax.text(sy, sx, str(self.science_value), ha = 'center', va = 'center', color = 'black', fontsize = 12, fontweight = 'bold', zorder = 20)
            self.patches.append(value_text)
        
        # Clear previous status texts
        for t in self.status_texts:
            t.remove()
        self.status_texts = []

        # Display Drone Info (Entropy & Action)
        action_map = {0: 'Stay', 1: 'Collect Science', 2: 'Up', 3: 'Down', 4: 'Left', 
                      5: 'Right', 6: 'Up-Right', 7: 'Up-Left', 8: 'Down-Right', 9: 'Down-Left'}
        
        for i, drone in enumerate(drones):
            entropy = drone.belief_state.get_entropy()
            action_code = drone.last_action
            action_str = action_map.get(action_code, "None")
            
            text_color = 'black' 
            font_weight = 'normal' 
            bg_color = 'white' 
            
            status_str = f"Drone {drone.drone_id} | H: {entropy:.3f} | Action: {action_str}"
            if drone.drifted:
                status_str += " | DRIFTED!"
            
            # Place text below the plot
            assert self.ax is not None
            t = self.ax.text(0.05, -0.12 - (i * 0.06), status_str, 
                             transform=self.ax.transAxes, fontsize=10, 
                             color=text_color, fontweight=font_weight,
                             bbox=dict(facecolor=bg_color, alpha=0.8, edgecolor='gray', boxstyle='round'))
            self.status_texts.append(t)
        
        # TODO Fix later, Just me editing out more wind related stuff
        # Draw wind direction arrow
        # arrow_len = 1.5
        # dx = np.sin(self.wind_direction) * arrow_len
        # dy = np.cos(self.wind_direction) * arrow_len
        # arrow = patches.Arrow(self.grid_size - 2.5, 2.5, dx, dy, width=0.5, color='black', zorder=10)
        # self.ax.add_patch(arrow)
        # self.patches.append(arrow)

        # Draw the planned path if provided
        if path is not None:
            for (r, c) in path:
                # Draw small semi-transparent circles for the path
                dot = patches.Circle(
                    (c, r), 0.2, 
                    color='red', alpha=0.3, zorder=5
                )
                assert self.ax is not None
                self.ax.add_patch(dot)
                self.patches.append(dot)

        for drone in drones:
            corner_x = drone.x - drone.window_size // 2 - 0.5
            corner_y = drone.y - drone.window_size // 2 - 0.5

            rectangle = patches.Rectangle(
                (corner_y, corner_x),
                drone.window_size,
                drone.window_size,
                linewidth=2,
                edgecolor='black',
                facecolor='none'
            )
            assert self.ax is not None
            self.ax.add_patch(rectangle)
            self.patches.append(rectangle)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.record_frames:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas = self.fig.canvas
            assert isinstance(canvas, FigureCanvasAgg)
            buf = canvas.buffer_rgba()
            # Use buffer_rgba() as tostring_rgb() is deprecated/removed in newer Matplotlib
            image = np.frombuffer(buf, dtype='uint8')
            
            # Handle HiDPI scaling by calculating actual buffer dimensions
            w, h = self.fig.canvas.get_width_height()
            if len(image) != w * h * 4:
                scale = (len(image) / (w * h * 4)) ** 0.5
                w = int(w * scale)
                h = int(h * scale)
            
            image = image.reshape((h, w, 4))
            image = image[:, :, :3].copy() # Convert RGBA to RGB
            self.frames.append(image)
        
        return self.fig


    def save_gif(self, filename, fps=5):
        if self.frames:
            imageio.mimsave(filename, self.frames, fps=fps)
            print(f"Animation saved to {filename}")


    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None