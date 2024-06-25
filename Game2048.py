import random
import math

class Game2048:
    def __init__(self, max=2048, seed=0) -> None:
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        random.seed(seed)
        
    def add_tile(self):
        if self.board_full():
            return
        x, y = random.randint(0, 3), random.randint(0, 3)
        while self.board[x][y] != 0:
            x, y = random.randint(0, 3), random.randint(0, 3)
        self.board[x][y] = 2 if random.random() < 0.9 else 4
                    
    def reset(self, tile1, tile2):
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.board[tile1[0]][tile1[1]] = 2
        self.board[tile2[0]][tile2[1]] = 2

    def flatten(self):
        return [cell for row in self.board for cell in row]

    def board_full(self):
        return all([cell != 0 for row in self.board for cell in row])

    def __str__(self) -> str:
        return '\n'.join([' '.join([str(cell) for cell in row]) for row in self.board])

    def move_up(self):
        initial_state = [row[:] for row in self.board]
        for col in range(4):
            for row in range(1, 4):
                if self.board[row][col] != 0:
                    current_row = row
                    while current_row > 0 and self.board[current_row - 1][col] == 0:
                        self.board[current_row - 1][col] = self.board[current_row][col]
                        self.board[current_row][col] = 0
                        current_row -= 1
                    if current_row > 0 and self.board[current_row - 1][col] == self.board[current_row][col]:
                        self.board[current_row - 1][col] *= 2
                        self.board[current_row][col] = 0
        return self.board != initial_state

    def move_down(self):
        initial_state = [row[:] for row in self.board]
        for col in range(4):
            for row in range(2, -1, -1):
                if self.board[row][col] != 0:
                    current_row = row
                    while current_row < 3 and self.board[current_row + 1][col] == 0:
                        self.board[current_row + 1][col] = self.board[current_row][col]
                        self.board[current_row][col] = 0
                        current_row += 1
                    if current_row < 3 and self.board[current_row + 1][col] == self.board[current_row][col]:
                        self.board[current_row + 1][col] *= 2
                        self.board[current_row][col] = 0
        return self.board != initial_state

    def move_left(self):
        initial_state = [row[:] for row in self.board]
        for row in range(4):
            for col in range(1, 4):
                if self.board[row][col] != 0:
                    current_col = col
                    while current_col > 0 and self.board[row][current_col - 1] == 0:
                        self.board[row][current_col - 1] = self.board[row][current_col]
                        self.board[row][current_col] = 0
                        current_col -= 1
                    if current_col > 0 and self.board[row][current_col - 1] == self.board[row][current_col]:
                        self.board[row][current_col - 1] *= 2
                        self.board[row][current_col] = 0
        return self.board != initial_state

    def move_right(self):
        initial_state = [row[:] for row in self.board]
        for row in range(4):
            for col in range(2, -1, -1):
                if self.board[row][col] != 0:
                    current_col = col
                    while current_col < 3 and self.board[row][current_col + 1] == 0:
                        self.board[row][current_col + 1] = self.board[row][current_col]
                        self.board[row][current_col] = 0
                        current_col += 1
                    if current_col < 3 and self.board[row][current_col + 1] == self.board[row][current_col]:
                        self.board[row][current_col + 1] *= 2
                        self.board[row][current_col] = 0
        return self.board != initial_state
    
    def is_game_over(self):
        if not self.board_full():
            return False

        for row in range(4):
            for col in range(4):
                if col < 3 and self.board[row][col] == self.board[row][col + 1]:
                    return False
                if row < 3 and self.board[row][col] == self.board[row + 1][col]:
                    return False

        return True
    
    def get_highest_tile(self):
        highest = 0
        for row in self.board:
            for cell in row:
                if cell > highest:
                    highest = cell
        return highest
    
    def get_total_score(self):
        return sum([cell for row in self.board for cell in row])
    
    def get_empty_tiles(self):
        empty = 0
        for row in self.board:
            for cell in row:
                if cell == 0:
                    empty += 1
        return empty
    
    def get_score(self, move_changed_board=True):
        highest_tile = self.get_highest_tile()
        empty_tiles = self.get_empty_tiles()
        total_sum = self.get_total_score()
        
        # Calculate smoothness
        smoothness = 0
        for i in range(4):
            for j in range(4):
                if j < 3:
                    smoothness -= abs(self.board[i][j] - self.board[i][j+1])
                if i < 3:
                    smoothness -= abs(self.board[i][j] - self.board[i+1][j])
        
        # Weight factors (you can adjust these)
        w1, w2, w3, w4, w5 = 1.0, 2.0, 1.0, 0.1, -100.0
        
        # Combine factors
        score = (w1 * math.log2(highest_tile) if highest_tile > 0 else 0) + \
                (w2 * empty_tiles) + \
                (w3 * math.log2(total_sum) if total_sum > 0 else 0) + \
                (w4 * smoothness)
        
        # Apply penalty for impossible moves
        if not move_changed_board:
            score += w5
        
        return score