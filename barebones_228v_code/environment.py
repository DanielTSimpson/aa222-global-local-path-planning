"""
Environment module for multi-agent fire search simulation
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
        self.grid_size = 15 # The side length of the square grid-world
        self.wind_speed = 0.0 # The probability of the wind moving drones
        self.wind_direction = 0 # The direction the wind would bias drone movement in radians
        
        self.fire_pos = np.random.randint(0, self.grid_size, size=2) # The random 2D position of the fire
        self.fire_extinguished = False

        self.patches = []
        self.fig, self.ax = None, None
        self.status_texts = []
        self.frames = []
        self.record_frames = False

    def render(self, drones):
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark explored cells (visited or observed)
        for drone in drones:
            for (r, c) in drone.visited_cells:
                grid[r, c] = 1

        if not self.fire_extinguished:
            grid[tuple(self.fire_pos)] = 2

        for idx, drone in enumerate(drones):
            grid[tuple(drone.position)] = idx + 3

        cmap = colors.ListedColormap(['#ffcccc', 'white', 'red', 'blue', 'green', 'orange', 'purple'])
        bounds = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 7))
            self.im = self.ax.imshow(grid, cmap=cmap, norm=norm)
            self.ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
            self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)
            self.ax.set_xlabel('Y Position')
            self.ax.set_ylabel('X Position')
            self.ax.set_title("Multi-Agent Costly-Comms Objective Search", fontsize=12, fontweight='bold')

            plt.ion()
            plt.show(block=False)
        else:
            self.im.set_data(grid)

        for p in self.patches:
            p.remove()
        self.patches.clear()
        
        # Clear previous status texts
        for t in self.status_texts:
            t.remove()
        self.status_texts = []

        # Display Drone Info (Entropy & Action)
        action_map = {0: 'Stay', 1: 'Right', 2: 'Left', 3: 'Up', 4: 'Down', 5: 'COMMUNICATE', 6: 'Extinguish'}
        
        for i, drone in enumerate(drones):
            entropy = drone.belief_state.get_entropy()
            action_code = drone.last_action
            action_str = action_map.get(action_code, "None")
            
            # Formatting for Communicate
            is_comm = (action_code == 5)
            text_color = 'red' if is_comm else 'black'
            font_weight = 'bold' if is_comm else 'normal'
            bg_color = 'yellow' if is_comm else 'white'
            
            status_str = f"Drone {drone.drone_id} | H: {entropy:.3f} | Action: {action_str}"
            if drone.drifted:
                status_str += " | DRIFTED!"
            
            # Place text below the plot
            t = self.ax.text(0.05, -0.12 - (i * 0.06), status_str, 
                             transform=self.ax.transAxes, fontsize=10, 
                             color=text_color, fontweight=font_weight,
                             bbox=dict(facecolor=bg_color, alpha=0.8, edgecolor='gray', boxstyle='round'))
            self.status_texts.append(t)

        # Draw wind direction arrow
        arrow_len = 1.5
        dx = np.sin(self.wind_direction) * arrow_len
        dy = np.cos(self.wind_direction) * arrow_len
        arrow = patches.Arrow(self.grid_size - 2.5, 2.5, dx, dy, width=0.5, color='black', zorder=10)
        self.ax.add_patch(arrow)
        self.patches.append(arrow)

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
            self.ax.add_patch(rectangle)
            self.patches.append(rectangle)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.record_frames:
            # Use buffer_rgba() as tostring_rgb() is deprecated/removed in newer Matplotlib
            image = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype='uint8')
            
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