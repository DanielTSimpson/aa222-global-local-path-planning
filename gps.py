"""
GPS module of the path planning algo.
Handles scanning for large obstacles
"""
from environment import SearchEnv
import numpy as np
import config as cfg

class GPS():
    def __init__(self, environment: SearchEnv):
        self.environment = environment
        self.large_obstacle_map = (self.environment.the_grid == self.environment.terrain["LARGE OBSTACLE"]) # GPS can only see the large obstacles
        self.observed_small_obstacles = np.zeros_like(environment.the_grid, dtype=bool) # It can learn about small obstacles through the drone

        self.open_map = (~self.large_obstacle_map).astype(int)
        self.gps_map = self.large_obstacle_map.astype(int)
        self.objective_position = environment.science_pos

    def add_small_obstacle(self, position):
        """Adds a newly observed small obstacle to the map and updates pathfinding maps."""
        if not self.observed_small_obstacles[position]:
            self.observed_small_obstacles[position] = True
            self.gps_map = np.bitwise_or(self.large_obstacle_map, self.observed_small_obstacles).astype(int)
            self.open_map = 1 - self.gps_map
        
    def _get_instructions(self, path):
        instructions = []
        move_to_action = {v: k for k, v in cfg.MOVES.items()}
        
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            if (dx, dy) in move_to_action:
                instructions.append(move_to_action[(dx, dy)])
        return instructions
    
    def a_star(self, drone_position):
        """
        Applies the A-star path search algorithm to determine where a drone should move
        Source: https://www.datacamp.com/tutorial/a-star-algorithm
        input: self
        output: list of actions
        """

        reconstructed_path = []
        drone_instructions = []

        def heuristic(initial_pos, final_pos): # Making this its standalone function for each switching between norm methods
            return float(np.linalg.norm(np.array(initial_pos) - np.array(final_pos), ord=np.inf))

        def reconstruct_path(position):
            path = []
            while position is not None:
                path.append(position)
                position = parent.get(position)
            return path[::-1]

        start = tuple(drone_position)
        finish = tuple(self.objective_position)
        openList = [start]
        closedList = []
        g_score: dict[tuple, float] = {start: 0}
        h_score: dict[tuple, float] = {start: heuristic(drone_position, self.objective_position)}
        f_score: dict[tuple, float] = {start: g_score[start] + h_score[start]}
        parent: dict[tuple, tuple | None] = {start: None}

        while len(openList) != 0:
            current = min(openList, key=lambda node: f_score.get(node, float('inf')))
            if current == finish:
                reconstructed_path = reconstruct_path(current)
                break
            
            openList.remove(current)
            closedList.append(current)

            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0: # dont check current, only neighbors
                        continue
                    neighbor = (current[0] + i, current[1] + j)
                    if neighbor in closedList:
                        continue
                    if neighbor == current: # dont check the same thing twice
                        continue
                    if not (0 <= neighbor[0] < self.open_map.shape[0] and 0 <= neighbor[1] < self.open_map.shape[1]): # check if in bounds
                        continue
                    if self.open_map[neighbor[0], neighbor[1]] == 0: # don't bother checking obstacles
                        continue 

                    tentative_g = g_score[current] + 1 # Every neighbor is 1 move away
                    if neighbor not in openList:
                        openList.append(neighbor)
                    elif tentative_g >= g_score.get(neighbor, float('inf')): # the path isn't better
                        continue

                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_score[neighbor] = heuristic(neighbor, finish)
                    f_score[neighbor] = tentative_g + h_score[neighbor]
        
        if (reconstructed_path == []): 
            if cfg.VERBOSE_LOGGING:
                print("ERR: NO PATH FOUND THROUGH A*")
            return None, None
        drone_instructions = self._get_instructions(reconstructed_path)
        return reconstructed_path, drone_instructions