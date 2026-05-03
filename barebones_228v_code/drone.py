"""
Drone module for Dec-POMDP multi-agent system
Implements intelligent decision-making for fire search and extinguish
"""
import numpy as np
from copy import deepcopy
from environment import SearchEnv
from belief import Belief     
import config as cfg

class Drone():
    """
    Dec-POMDP Agent with belief state and value-based decision making
    """
    def __init__(self, environment: SearchEnv):
        self.drone_id = 0
        self.window_size = 0
        self.env = environment
        self.position = np.random.randint(0, self.env.grid_size, size=2)
        self.budget = cfg.MAX_BUDGET
        self.belief_state = Belief(self.env.grid_size)
        self.lookahead_depth = cfg.LOOKAHEAD_DEPTH
        
        self.time = 0
        self.visited_cells = set() # Track the global visited cells
        self.steps_since_last_comm = 0
        self.last_action = None
        self.fire_found = False
        self.drifted = False
        self.history = []
        self.stuck_count = 0

        # POMDP Params
        self.gamma = 0.0
        self.exploration_bonus = 0.0 
        self.movement_cost = 0.0
        self.comm_cost = 0.0
        self.time_cost = 0.0
        
        self.fire_found = self.observe()
        if not self.history:
            self.history.append(self.state)
        
    @property
    def x(self):
        return self.position[0]
    
    @property
    def y(self):
        return self.position[1]

    @property
    def state(self):
        return [self.x, self.y, self.fire_found]


    def action(self, action):
        """Execute action and update state"""
        self.last_action = action
        
        # Deduct costs from budget
        step_cost = self.time_cost
        if action in [1, 2, 3, 4]: # Movement actions
            step_cost += self.movement_cost
        elif action == 5: # Communication action
            step_cost += self.comm_cost
        self.budget -= step_cost

        prev_position = self.position.copy()

        x = self.x
        y = self.y
        telemetry_packet = None
        
        if action == 1:  # Up
            y = min(self.env.grid_size - 1, self.y + 1)
        elif action == 2:  # Down
            y = max(0, self.y - 1)
        elif action == 3:  # Left
            x = max(0, self.x - 1)
        elif action == 4:  # Right
            x = min(self.env.grid_size - 1, self.x + 1)
        elif action == 5: # Communicate
            telemetry_packet = self.create_telemetry_packet()
            self.steps_since_last_comm = 0
        elif action == 6: # Extinguish the Fire
            self.env.fire_extinguished = True

        self.drifted = False
        # Apply wind drift

        if np.random.random() < abs(self.env.wind_speed):
            cos_wind = np.cos(self.env.wind_direction)
            sin_wind = np.sin(self.env.wind_direction)
            
            dx = int(np.sign(cos_wind)) if np.random.random() < abs(cos_wind) else 0
            dy = int(np.sign(sin_wind)) if np.random.random() < abs(sin_wind) else 0
            
            if dx != 0 or dy != 0:
                self.drifted = True
                x = max(0, min(self.env.grid_size - 1, x + dx))
                y = max(0, min(self.env.grid_size - 1, y + dy))

        self.position = np.array([x, y])
        
        if np.array_equal(self.position, prev_position):
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            
        if not self.env.fire_extinguished: self.fire_found = self.observe() # Observe for free after every action
        self.history.append(self.state)     
        self.time += 1
        self.steps_since_last_comm += 1

        return telemetry_packet


    def observe(self):
        """Update belief based on observation"""
        x_check = (self.x - self.window_size // 2 <= self.env.fire_pos[0] <= self.x + self.window_size // 2)
        y_check = (self.y - self.window_size // 2 <= self.env.fire_pos[1] <= self.y + self.window_size // 2)
        fire_observed = x_check and y_check
        
        self.belief_state.update_from_observation(self.position, self.window_size, fire_observed)        
        
        # Update visited_cells to include all observed cells
        half = self.window_size // 2
        x_min, x_max = max(0, self.x - half), min(self.env.grid_size, self.x + half + 1)
        y_min, y_max = max(0, self.y - half), min(self.env.grid_size, self.y + half + 1)
        for r in range(x_min, x_max):
            for c in range(y_min, y_max):
                self.visited_cells.add((r, c))
        
        if fire_observed: 
            #print(f"Drone {self.drone_id} found fire at position {self.env.fire_pos}!")
            
            if np.array_equal(self.position, self.env.fire_pos):
                self.action(6) # Extinguish the fire

            else: # Move towards the fire
                if self.x < self.env.fire_pos[0]:
                    self.action(4)
                elif self.x > self.env.fire_pos[0]:
                    self.action(3)
                elif self.y < self.env.fire_pos[1]:
                    self.action(1)
                elif self.y > self.env.fire_pos[1]:
                    self.action(2)

        return fire_observed


    def create_telemetry_packet(self):
        """Creates telemetry packet with belief state"""
        packet = {
            'sender_id': self.drone_id,
            'timestamp': self.time,
            'position': self.position.copy(),
            'belief_state': self.belief_state.copy(),
            'visited_cells': self.visited_cells.copy(),
            'history': deepcopy(self.history)
        }
        return packet


    def receive_telemetry(self, packet, communication_noise=0.1):
        """Receive and merge belief states"""
        other_visited = packet['visited_cells']
        self.visited_cells.update(other_visited)
        if 'belief_state' in packet:
            self.belief_state.merge(packet['belief_state'])


    def _get_best_value(self, belief, position, visited, depth):
        """
        Recursive helper to calculate the best Q-value from a given state with limited lookahead.
        """
        if depth == 0:
            return 0.0
            
        max_q = -float('inf')
        moves = { 1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}
        current_entropy = belief.get_entropy()
        
        for _, (dx, dy) in moves.items():
            nx = max(0, min(self.env.grid_size - 1, position[0] + dx))
            ny = max(0, min(self.env.grid_size - 1, position[1] + dy))
            
            # P(o|b, a)
            half = self.window_size // 2
            x_min, x_max = max(0, nx - half), min(self.env.grid_size, nx + half + 1)
            y_min, y_max = max(0, ny - half), min(self.env.grid_size, ny + half + 1)
            
            # R(b, a) - Bonus granted if the new window reveals unvisited cells
            new_cells = sum(1 for r in range(x_min, x_max) for c in range(y_min, y_max) if (r, c) not in visited)
            bonus = self.exploration_bonus if new_cells > 0 else 0
            reward_action = bonus - self.movement_cost - self.time_cost
            
            prob_see_fire = np.sum(belief.belief_grid[x_min:x_max, y_min:y_max])
            prob_see_nothing = 1.0 - prob_see_fire
            
            # U(b'| o = Fire) - Immediate high value (entropy reduction)
            gain_see_fire = current_entropy
            
            # U(b'| o = Nothing)
            temp_belief = belief.copy()
            temp_belief.update_from_observation((nx, ny), self.window_size, fire_found=False)
            gain_see_nothing = current_entropy - temp_belief.get_entropy()
            
            # Recursive Value
            future_val = 0.0
            if depth > 1 and prob_see_nothing > 0:
                new_visited = visited.copy()
                for r in range(x_min, x_max):
                    for c in range(y_min, y_max):
                        new_visited.add((r, c))
                future_val = self._get_best_value(temp_belief, (nx, ny), new_visited, depth - 1)
            
            # Q(b, a)
            val_nothing = gain_see_nothing + self.gamma * future_val
            q = reward_action + self.gamma * (prob_see_fire * gain_see_fire + prob_see_nothing * val_nothing)
            
            if q > max_q:
                max_q = q
                
        return max_q

    def decide_action_pomdp(self):
        """
        Lookahead POMDP Planning (Depth=4).
        Returns the action index (1-5) with the highest Q-value.
        """
        best_actions = [0] 
        max_q_value = -float('inf')
        
        current_entropy = self.belief_state.get_entropy()
        
        # Actions: 0=Stay 1=Up, 2=Down, 3=Left, 4=Right
        moves = { 1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}

        # --- Check the Q-value for performing a movement ---
        for action_idx, (dx, dy) in moves.items():
            nx = max(0, min(self.env.grid_size - 1, self.x + dx))
            ny = max(0, min(self.env.grid_size - 1, self.y + dy))
            
            # --- P(o|b, a) ---
            half = self.window_size // 2
            x_min, x_max = max(0, nx - half), min(self.env.grid_size, nx + half + 1)
            y_min, y_max = max(0, ny - half), min(self.env.grid_size, ny + half + 1)
            
            # --- R(b, a) ---
            # Bonus granted if the new window reveals unvisited cells
            new_cells = sum(1 for r in range(x_min, x_max) for c in range(y_min, y_max) if (r, c) not in self.visited_cells)
            bonus = self.exploration_bonus if new_cells > 0 else 0
            reward_action = bonus - self.movement_cost - self.time_cost
            
            # P(o = Fire)
            prob_see_fire = np.sum(self.belief_state.belief_grid[x_min:x_max, y_min:y_max])
            
            # P(o = Nothing)
            prob_see_nothing = 1.0 - prob_see_fire
            
            # U(b'| o = Fire)
            gain_see_fire = current_entropy

            # U(b'| o = Nothing)
            temp_belief = self.belief_state.copy()
            temp_belief.update_from_observation((nx, ny), self.window_size, fire_found=False)
            gain_see_nothing = current_entropy - temp_belief.get_entropy()
            
            # --- Future Value (Lookahead) ---
            future_val = 0.0
            if self.lookahead_depth > 1 and prob_see_nothing > 0:
                new_visited = self.visited_cells.copy()
                for r in range(x_min, x_max):
                    for c in range(y_min, y_max):
                        new_visited.add((r, c))
                future_val = self._get_best_value(temp_belief, (nx, ny), new_visited, self.lookahead_depth - 1)

            # Q(b, a) = R(b, a) + \gamma*\sum(P(o|b, a)*U(b))
            val_nothing = gain_see_nothing + self.gamma * future_val
            q_value = reward_action + self.gamma*(prob_see_fire*gain_see_fire + prob_see_nothing*val_nothing)
            
            #print(f"Time step {self.time} | Action {action_idx} | Reward: {reward_action:.2f} | Q-Value: {q_value:.2f}")
            #print(f"\tP(o = Fire):    {prob_see_fire:.6f}\tU(b'| o = Fire):    {gain_see_fire:.4f}")
            #print(f"\tP(o = Nothing): {prob_see_nothing:.6f}\tU(b'| o = Nothing): {val_nothing:.4f} (Inc. Future: {future_val:.2f})")
            
            if q_value > max_q_value:
                max_q_value = q_value
                best_actions = [action_idx]
            elif np.isclose(q_value, max_q_value):
                best_actions.append(action_idx)

        best_action = int(np.random.choice(best_actions))
# --- Check the Q-value for performing communication ---
        
        # R(b, a = communicate)
        reward_comm = -self.comm_cost - self.time_cost

        # U(b' | a = communicate)
        max_entropy = np.log(self.env.grid_size * self.env.grid_size)
        information_possessed = max(0.0, 2*(max_entropy - current_entropy))
        
        # Staleness factor forces gain to 0 immediately after communicating, 
        # and asymptotically approaches 1.0 as time passes.
        decay_rate = 0.8
        staleness_factor = 1.0 - (decay_rate ** self.steps_since_last_comm)
        
        estimated_comm_gain = information_possessed * staleness_factor

        # Q(b, a = communicate)
        # Add future value (approximated by max_q_value since state doesn't change)
        q_comm = reward_comm + estimated_comm_gain + self.gamma * max_q_value

        # Only communicate if it's better than moving AND provides tangible information gain
        if q_comm > max_q_value and estimated_comm_gain > 0.5:
            return 5
            
        return best_action