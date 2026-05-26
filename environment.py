"""
Environment module for multi-agent science search simulation
Handles rendering, step execution, and reward computation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.widgets import Button
from scipy.signal import convolve2d

from gymnasium import Env
import imageio

import config as cfg

class SearchEnv(Env):
    """Multi-agent search environment with Dec-POMDP framework"""
    def __init__(self, grid_size=20):
        self.grid_size = grid_size # The side length of the square grid-world
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

        # Small Science Definition
        self.small_science = {}
        self.collected_small_science = set()

        # Start zone definition
        self.the_grid[self.grid_size - 5 : self.grid_size, 
                      self.grid_size - 5 : self.grid_size] = self.terrain["BUFFER ZONE"]

        ## Rendering Parameters
        self.patches = []
        self.fig, self.ax = None, None
        self.status_texts = []
        self.frames = []
        self.record_frames = False
        self.buttons = []
        
        # Setting a toggleable option for showing the user small obstacles the drone hasn't observed yet
        self.show_hidden_small_obstacles = False 
        self.show_hidden_science = False
        
        # adding in the small science objectives scattered throughout the map
        self.show_hidden_small_science = False
        
        # adding a pause button
        self.paused = False

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

    def generate_small_science(self, num_small_science = 10, min_value = 1, max_value =5):
        # populates the grid world with small science objectives, forcing the drone to explore more if it comes across one
        placed = 0
        attempts = 0
        max_attempts = 1000 # I'm hard coding this because there's no way we'll hit 1000
        
        while placed < num_small_science and attempts < max_attempts:
            attempts += 1
            
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            
            cell = (r, c)
            
            # these are basically the 3 conditions which would stop us from putting a science objective in a cell
            if not self.is_free(r, c) or cell == tuple(self.science_pos) or cell in self.small_science:
                continue
            
            value = np.random.randint(min_value, max_value + 1)
            self.small_science[cell] = value
            
            placed += 1

    def spawn_obstacle(self, obs_type, mu, sigma, type = None):
        w, h = self._sample_obstacle_size(mu, sigma)
        blocked_map = (self.the_grid != self.terrain["FREE"]).astype(int)
        footprint = np.ones((h, w), dtype=int)
        overlap_map = convolve2d(blocked_map, footprint, mode='valid')
        valid_y, valid_x = np.where(overlap_map == 0)
        if len(valid_y) == 0:
            return None
        if type == None or type == "Random":
            random_idx = np.random.randint(len(valid_y))

        if type == "Gaussian":
            """ I want to choose indexes near the middle to 
             bias more obstacles in the agent's path, but also 
             make sure there's a decent enough spread such that 
             we dont get a massive "rock" in the middle"""
            random_idx = round(np.random.normal(len(valid_x) // 2, 2*np.sqrt(len(valid_x))))
        r = valid_y[random_idx]
        c = valid_x[random_idx]
        self.the_grid[r : r+h, c : c+w] = obs_type
        return (r, c, w, h)
    
    def generate_obstacles(self, num_large = 5, num_small = 15, large_mu = 4, large_sigma = 1.0, small_mu = 1.5, small_sigma = 0.5):
        [self.spawn_obstacle(self.terrain["LARGE OBSTACLE"], large_mu, large_sigma, type = "Gaussian") for _ in range(num_large)]
        [self.spawn_obstacle(self.terrain["SMALL OBSTACLE"], small_mu, small_sigma) for _ in range(num_small)]
    
    def toggle_small_obstacles(self, event):
        # helper function for toggling small obstacle visibility
        self.show_hidden_small_obstacles = not self.show_hidden_small_obstacles
        if self.buttons:
            self.buttons[0].color = "#cf7f7f" if self.show_hidden_small_obstacles else "#f8a3a3"
            self.buttons[0].hovercolor = self.buttons[0].color
        print(f"Show hidden small obstacles: {self.show_hidden_small_obstacles}")
        self.request_redraw()

    def toggle_science(self, event):
        # helper function for toggling science visibility
        self.show_hidden_science = not self.show_hidden_science
        self.show_hidden_small_science = not self.show_hidden_small_science
        if self.buttons:
            self.buttons[1].color = "#cf7f7f" if self.show_hidden_science else "#f8a3a3"
            self.buttons[1].hovercolor = self.buttons[1].color
        print(f"Show hidden science: {self.show_hidden_science}")
        self.request_redraw()
        
    def toggle_pause(self, event):
        self.paused = not self.paused
        if self.buttons:
            self.buttons[2].color = "#cf7f7f" if self.paused else "#f8a3a3"
            self.buttons[2].hovercolor = self.buttons[2].color
        print(f"Paused: {self.paused}")
        self.request_redraw()
        
    def request_redraw(self):
        # if you have the run paused and try to toggle the environment stuff, it won't show until you Unpause
        # so this is gonna fix that
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def render(self, drone, path=None):
        # Reset rendering parameters
        for p in self.patches:
            p.remove()
        self.patches.clear()
        grid = np.zeros((self.grid_size, self.grid_size))        
        large_obs_val = self.terrain.get("LARGE OBSTACLE", 3) #Get the value for large obstacles
        small_obs_val = self.terrain.get("SMALL OBSTACLE", 4) #Get the value for small obstacles
        
        # ==== LABELING OBJECTS IN THE GRID ===
        ## Label large obstacles
        grid[self.the_grid == large_obs_val] = 7
        ## Label small obstacles
        if self.show_hidden_small_obstacles:
            grid[self.the_grid == small_obs_val] = 8
        else:
            for (r, c) in drone.known_small_obstacles:
                grid[r, c] = 8
        ## Label large science
        science_visible_to_drone = tuple(self.science_pos) in getattr(drone, "known_science", set())
        if self.show_hidden_science or science_visible_to_drone:
            grid[tuple(self.science_pos)] = 2
        ## Label small science
        if cfg.SMALL_SCIENCE_ENABLED:
            if self.show_hidden_small_science:
                for (r, c), value in self.small_science.items():
                    if (r, c) not in self.collected_small_science:
                        grid[r, c] = 9
            else:
                for (r, c) in drone.known_small_science:
                    if (r, c) not in self.collected_small_science:
                        grid[r, c] = 9
        ## Label explored cells
        for (r, c) in drone.visited_cells:
            if self.the_grid[r, c] != large_obs_val and self.the_grid[r, c] != small_obs_val: # only mark cells as explored if they're not an obstacle -- otherwise we're overwriting those grids visually
                grid[r, c] = 1   
        ## Label currently cells visible to the agent
        if hasattr(drone, "current_visible_cells"):
            for (r, c) in drone.current_visible_cells:
                if not self.is_obstacle(r, c): # we want to make sure we don't overwrite obstacle colors with the visible cell color
                    grid[r, c] = 6
        ## Label the agent itself
        grid[tuple(drone.position)] = 3


        # ===== FORMATTING THE PLOT =====
        cmap = colors.ListedColormap(['#ffcccc', 'white', '#2ecc71', 'blue', 'green', 'orange', 'purple', 'grey', 'darkgreen', "yellow"])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        if self.fig is None:
            ## Format the plot itself
            self.fig, self.ax = plt.subplots(figsize=(5.5, 6.0))
            self.fig.subplots_adjust(top=0.95, bottom=0.15)
            self.ax.set_anchor('N')
            self.im = self.ax.imshow(grid, cmap=cmap, norm=norm)
            self.ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            self.ax.set_xlabel('Y Position')
            self.ax.set_ylabel('X Position')
            self.ax.set_title("Multi-Agent Science Objective Search", fontsize=12, fontweight='bold')
            ## Format Toggle Buttons
            button_ax1 = self.fig.add_axes([0.07, 0.04, 0.42, 0.05])
            button_ax2 = self.fig.add_axes([0.55, 0.04, 0.35, 0.05])
            button_ax3 = self.fig.add_axes([0.35, 0.10, 0.30, 0.05])
            button1 = Button(button_ax1, 'Toggle Small Obstacle Visibility', color="#f8a3a3", hovercolor="#f8a3a3")
            button2 = Button(button_ax2, 'Toggle Science Visibility', color="#f8a3a3", hovercolor="#f8a3a3")
            button3 = Button(button_ax3, 'Pause/Resume', color="#f8a3a3", hovercolor="#f8a3a3")
            button1.on_clicked(self.toggle_small_obstacles)
            button2.on_clicked(self.toggle_science)
            button3.on_clicked(self.toggle_pause)
            self.buttons = [button1, button2, button3]

            plt.ion()
            plt.show(block=False)
        else:
            self.im.set_data(grid)
        

        # ===== RENDERING THE ENVIRONMENT =====
        ## Render science value label (how important the science is)
        if not self.science_found:
            sx, sy = self.science_pos
            assert self.ax is not None
            value_text = self.ax.text(sy, sx, str(self.science_value), ha = 'center', va = 'center', color = 'black', fontsize = 12, fontweight = 'bold', zorder = 20)
            self.patches.append(value_text)
        ## Render the planned path if provided
        if path is not None:
            for (r, c) in path:
                # Render small semi-transparent circles for the path
                dot = patches.Circle(
                    (c, r), 0.2, 
                    color='red', alpha=0.3, zorder=5
                )
                assert self.ax is not None
                self.ax.add_patch(dot)
                self.patches.append(dot)
        ## Render the whole plot and update any queued tasks 
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.record_frames: self._record_frame()
        

        return self.fig

    def _record_frame(self):
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas = self.fig.canvas
            if isinstance(canvas, FigureCanvasAgg):
                buf = canvas.buffer_rgba()
                image = np.frombuffer(buf, dtype='uint8')
            else:
                import io
                buf_io = io.BytesIO()
                self.fig.savefig(buf_io, format='raw', dpi=self.fig.dpi)
                buf_io.seek(0)
                image = np.frombuffer(buf_io.getvalue(), dtype='uint8')
            
            w, h = self.fig.canvas.get_width_height()
            if len(image) != w * h * 4:
                scale = (len(image) / (w * h * 4)) ** 0.5
                w = int(round(w * scale))
                h = int(round(h * scale))
            
            image = image.reshape((h, w, 4))
            image = image[:, :, :3].copy() # Convert RGBA to RGB
            self.frames.append(image)
        except Exception as e:
            print(f"Warning: Failed to record frame: {e}")

    def save_gif(self, filename, fps=5):
        if self.frames:
            try: # imageio v2
                imageio.mimsave(filename, self.frames, fps=fps)
            except TypeError: # imageio v3 no longer supports 'fps', uses 'duration' in ms
                duration = 1000.0 / fps
                imageio.mimsave(filename, self.frames, duration=duration, loop=0)
            print(f"Animation saved to {filename}")

    def close(self, save_gif=False, filename="simulation.gif", fps=5):
        if save_gif and self.frames:
            self.save_gif(filename, fps=fps)
        if self.fig:
            plt.close(self.fig)
            self.fig = None