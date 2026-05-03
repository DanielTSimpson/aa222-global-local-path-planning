import numpy as np
import config as cfg

class Belief:
    """
    Belief class to manage the probability distribution of the fire location.
    """
    def __init__(self, grid_size = cfg.GRID_SIZE):
        """
        Initialize the belief grid with a uniform prior.
        
        Args:
            grid_size (int): The size of the square grid.
        """
        self.grid_size = grid_size
        # Uniform prior: 1 / total_cells
        self.belief_grid = np.ones((grid_size, grid_size)) / (grid_size * grid_size)

    def copy(self):
        """
        Create a lightweight copy of the Belief object.
        """
        new_obj = Belief(self.grid_size)
        new_obj.belief_grid = self.belief_grid.copy()
        return new_obj

    def get_entropy(self):
        """
        Calculate the Shannon entropy of the current belief distribution.
        """
        p = self.belief_grid.ravel()
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    
    def update_from_observation(self, drone_pos, window_size, fire_found):
        """
        Bayesian update of the belief grid based on observation.
        """
        # Determine the window boundaries
        r, c = drone_pos
        half = window_size // 2
        x_min = max(0, r - half)
        x_max = min(self.grid_size, r + half + 1)
        y_min = max(0, c - half)
        y_max = min(self.grid_size, c + half + 1)

        if fire_found:
            # If fire is found, probability is 1.0 inside the window, 0.0 outside
            mask = np.zeros_like(self.belief_grid)
            mask[x_min:x_max, y_min:y_max] = 1.0
            self.belief_grid *= mask
        else:
            # If no fire, probability is 0.0 inside the window
            self.belief_grid[x_min:x_max, y_min:y_max] = 0.0

        # Normalize to ensure sum is 1.0
        total = np.sum(self.belief_grid)
        if total > 0:
            self.belief_grid /= total
        else:
            # Fallback if something went wrong (shouldn't happen in this setup)
            self.belief_grid = np.ones_like(self.belief_grid) / self.belief_grid.size

    def merge(self, other_belief):
        """
        Merge another belief distribution into this one via element-wise multiplication.
        """
        self.belief_grid *= other_belief.belief_grid
        
        # Normalize to ensure sum is 1.0
        total = np.sum(self.belief_grid)
        if total > 0:
            self.belief_grid /= total
        else:
            # If contradiction (0 probability everywhere), reset to uniform
            self.belief_grid = np.ones_like(self.belief_grid) / self.belief_grid.size