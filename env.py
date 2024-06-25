import numpy as np
import random
from Game2048 import Game2048  # Make sure this import matches your file name

class Game2048Env:
    def __init__(self):
        self.game = Game2048()
        self.action_space = list(range(4))  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = np.zeros((4, 4), dtype=np.int32)
        self.reward_range = (-float('inf'), float('inf'))

    def reset(self):
        tile1 = (np.random.randint(4), np.random.randint(4))
        tile2 = (np.random.randint(4), np.random.randint(4))
        while tile2 == tile1:
            tile2 = (np.random.randint(4), np.random.randint(4))
        self.game.reset(tile1, tile2)
        return self._get_obs()

    def step(self, action):
        old_score = self.game.get_score()
        
        if action == 0:
            move_changed_board = self.game.move_up()
        elif action == 1:
            move_changed_board = self.game.move_down()
        elif action == 2:
            move_changed_board = self.game.move_left()
        elif action == 3:
            move_changed_board = self.game.move_right()
        else:
            raise ValueError('Invalid action')
        
        if move_changed_board:
            self.game.add_tile()
        
        new_score = self.game.get_score(move_changed_board)
        reward = new_score - old_score
        
        done = self.game.is_game_over()
        
        return self.game.flatten(), reward, done

    def _get_obs(self):
        return np.log2(np.array(self.game.board, dtype=np.int32).flatten())

    def render(self, mode='human'):
        print(self.game)

    def close(self):
        pass

    def sample(self):
        return random.choice(self.action_space)