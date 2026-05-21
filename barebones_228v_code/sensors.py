import numpy as np
import config as cfg

class OnlineSensor:
    # the class we'll be using to design our front facing sensors
    
    def __init__(self, view_depth = 4, view_angle = np.pi/2, false_negative_rate = 0.05, false_positive_rate = 0.01):
        # initializing our sensor
        # the view depth and view angle are fairly self-explanatory, showing how far and wide the drone's visual field is
        # the false_negative/positive_rates also just indicate exactly what they're named
        # honestly, I don't know if this description is necessary
        self.view_depth = view_depth
        self.view_angle = view_angle
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate
    
    def get_visible_cells(self, drone_pos, heading, grid_size = cfg.GRID_SIZE):
        # returns the cells within our visible front-facing cone
        r0, c0 = drone_pos # r0 is the column our drone is in (its y coord) and c0 is the column (its x coord)
        visible_cells = []

        for r in range(max(0, r0 - self.view_depth), min(grid_size, r0 + self.view_depth + 1)):
            for c in range(max(0, c0 - self.view_depth), min(grid_size, c0 + self.view_depth + 1)):
                dr = r0 - r
                dc = c0 - c

                dist = np.sqrt(dr**2 + dc**2)

                if dist == 0 or dist > self.view_depth:
                    continue
                
                angle = np.arctan2(dr, dc)

                angle_error = self.wrap_angle(angle - heading)

                if abs(angle_error) <= self.view_angle / 2:
                    visible_cells.append((r, c))

        return visible_cells
    
    def observe(self, drone_pos, heading, environment):
        # here we actually observe the visible cells our drone can see
        visible_cells = self.get_visible_cells(drone_pos, heading, grid_size = environment.grid_size)

        observations = {"visible_cells": visible_cells, "small_obstacles": [], "science": []}

        for cell in visible_cells:
            r, c = cell # defining the row and column number of the cell we're analyzing

            # first we see if that cell actually has any small obstacles in it
            if environment.small_obstacle_grid[r, c]:

                # we incorporate the odds of having a false negative
                # basically, we're just generating a probability and comparing it against our false_negative_rate
                # if it's greater, then we properly add the cell -- if not, then we skip over it like it wasn't there
                if np.random.random() > self.false_negative_rate:
                    observations["small_obstacles"].append(cell)

                # then we incorporate the likelihood of having a false positive
                # again, we generate a probability and compare it against our false_positive_rate
                # if it's Less than the false_positive_rate, then we add the cell
                else:
                    if np.random.random() < self.false_positive_rate:
                        observations["small_obstacles"].append(cell)
                
                # now we do our science checks
                # we see if the cell has a science object in it
                if environment.science_grid[r, c]:
                    
                    # we're also going to include a false_negative option here
                    if np.random.random() > self.false_negative_rate:
                        observations["science"].append(cell)
        return observations
    
    @staticmethod
    def wrap_angle(angle):
        # basically just wraps our angle to be from [-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi