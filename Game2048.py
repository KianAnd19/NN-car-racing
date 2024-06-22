import random

class Game2048:
    def __init__(self, seed) -> None:
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

    def move(self, direction):
        if direction == '00':
            self.move_up()
        elif direction == '01':
            self.move_down()
        elif direction == '10':
            self.move_left()
        elif direction == '11':
            self.move_right()
        else:
            raise ValueError('Invalid direction')
        
        
        if self.board_full() and self.is_game_over():
            return -1

    def move_up(self):
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
                        self.score += self.board[current_row - 1][col]
                        self.board[current_row][col] = 0
        self.add_tile()

    def move_down(self):
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
                        self.score += self.board[current_row + 1][col]
                        self.board[current_row][col] = 0
        self.add_tile()

    def move_left(self):
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
                        self.score += self.board[row][current_col - 1]
                        self.board[row][current_col] = 0
        self.add_tile()

    def move_right(self):
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
                        self.score += self.board[row][current_col + 1]
                        self.board[row][current_col] = 0
        self.add_tile()
        
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
    
    def get_score(self):
        # calculate score
        self.score = 0
        for row in self.board:
            for cell in row:
                if cell > self.score:
                    self.score = cell
        return self.score