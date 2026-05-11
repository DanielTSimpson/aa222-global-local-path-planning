"""
Environment module for multi-agent science search simulation
Handles rendering, step execution, and reward computation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches

from gymnasium import Env
import imageio

class SearchEnv(Env):
    """Multi-agent search environment with Dec-POMDP framework"""
    def __init__(self):
        self.grid_size = 20 # The side length of the square grid-world
        self.wind_speed = 0.0 # The probability of the wind moving drones
        self.wind_direction = 0.0 # The direction the wind would bias drone movement in radians
        
        self.science_pos = np.random.randint(0, self.grid_size, size=2) # The random 2D position of the science objective
        self.science_value = np.random.randint(1, 10) # how important the science objective is
        self.science_found = False
        self.science_collected = False

        # Obstacle stuff
        # Current setup is there's a separate set of variables for small obstacles and large obstacles
        # Might switch this later to where there's just one distribution that's pulled from 
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size), dtype = bool) # just a list for each cell indicating whether an obstacle is in it or not
        self.obstacles = []

        self.patches = []
        self.fig, self.ax = None, None
        self.status_texts = []
        self.frames = []
        self.record_frames = False

    def reset_obstacles(self):
        # clears existing obstacles from the environment, called before generating a new map
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size), dtype = bool) # just a list for each cell indicating whether an obstacle is in it or not
        self.obstacles = []

    def in_bounds(self, r, c):
        # checks if a given cell, with coordinates row r and column c, is inside the environment
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size
    
    def is_obstacle(self, r, c):
        # just returns whether or not the cell at row r and column c is an obstacle or free space
        return self.obstacle_grid[r, c]
    
    def is_free(self, r, c):
        # returns whether or not the cell at row r and column c is a free space
        return self.in_bounds(r, c) and not self.is_obstacle(r, c)
    
    def _sample_obstacle_size(self, mu, sigma, min_size = 1, max_size = 5):
        # lets us draw obstacle sizes from the small or large obstacle size distributions we define in the config file
        if max_size is None:
            max_size = max(1, self.grid_size // 4) # makes it so we don't go a lil too crazy with oversized obstacles

        size = int(round(np.random.normal(mu, sigma)))
        return int(np.clip(size, min_size, max_size))
    
    def _rectangle_cells(self, top_left, height, width):
        # generates all the grid cells covered by a rectangle obstacle
        # TODO need to add in a function for non-rectangular obstacles later
        r0, c0 = top_left
        cells = []
        for r in range(r0, r0 + height):
            for c in range(c0, c0 + width):
                if self.in_bounds(r, c):
                    cells.append((r, c))
        return cells
    
    def _too_close(self, cell, protected_cells, buffer_radius):
        # just tells us if an obstacle cell or objective cell is within a no-go zone
        r, c = cell
        for pr, pc in protected_cells:
            if abs(r - pr) + abs(c - pc) <= buffer_radius:
                return True
        return False
    
    def _can_place_obstacle(self, cells, protected_cells, buffer_radius):
        # compares our list of grid cells with obstacle cells and too-close cells to see where further obstacles can and cannot be placed
        for r, c in cells:
            if self.obstacle_grid[r, c]:
                return False
            if self._too_close((r, c), protected_cells, buffer_radius):
                return False
        return True

    def add_rectangle_obstacle(self, height, width, protected_cells=None, buffer_radius = 0):
        # goes through and determines where we can place a rectangle obstacle of a given height and width, and then updates the obstacle grid and list of obstacles accordingly
        if protected_cells is None:
            protected_cells = []
        for _ in range(500): # TODO add max attempts parameter here rather than hard coding 500
            r0 = np.random.randint(0, self.grid_size - height + 1)
            c0 = np.random.randint(0, self.grid_size - width + 1)

            cells = self._rectangle_cells((r0, c0), height, width)

            if self._can_place_obstacle(cells, protected_cells, buffer_radius):
                for r, c in cells:
                    self.obstacle_grid[r, c] = True
                obstacle = {"shape": "rectangle", "top_left": (r0, c0), "height": height, "width": width, "cells": cells}
                self.obstacles.append(obstacle)
                return obstacle
        return None
    
    def generate_obstacles(self, num_large = 5, num_small = 15, large_mu = 4, large_sigma = 1.0, small_mu = 1.5, small_sigma = 0.5, protected_cells = None, buffer_radius = 2):
        # actually going about generating all of our obstacles in the grid space
        
        # start by clearing existing obstacles from the environment
        self.reset_obstacles()

        if protected_cells is None:
            protected_cells = [tuple(self.science_pos)]

        # start with adding in all of our large obstacles
        for _ in range(num_large):
            h = self._sample_obstacle_size(large_mu, large_sigma, min_size = 2)
            w = self._sample_obstacle_size(large_mu, large_sigma, min_size = 2)
            self.add_rectangle_obstacle(h, w, protected_cells, buffer_radius)

        # now we add in our small obstacles
        for _ in range(num_small):
            h = self._sample_obstacle_size(small_mu, small_sigma, min_size = 1, max_size = 3)
            w = self._sample_obstacle_size(small_mu, small_sigma, min_size = 1, max_size = 3)
            self.add_rectangle_obstacle(h, w, protected_cells, buffer_radius)

        return self.obstacles

    def render(self, drones):
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark explored cells (visited or observed)
        for drone in drones:
            for (r, c) in drone.visited_cells:
                if not self.obstacle_grid[r, c]: # only mark cells as explored if they're not an obstacle -- otherwise we're overwriting those grids visually
                    grid[r, c] = 1
        grid[self.obstacle_grid] = 7


        if not self.science_found:
            grid[tuple(self.science_pos)] = 2

        for idx, drone in enumerate(drones):
            grid[tuple(drone.position)] = idx + 3

        cmap = colors.ListedColormap(['#ffcccc', 'white', '#2ecc71', 'blue', 'green', 'orange', 'purple', 'black'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
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
        action_map = {0: 'Stay', 1: 'Right', 2: 'Left', 3: 'Up', 4: 'Down', 6: 'Collect Science'}
        
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