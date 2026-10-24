import random
from collections import namedtuple
from enum import Enum
import numpy as np

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

GRID_SIZE = 20
BLOCK_SIZE = GRID_SIZE

# Arena bounds matching your original 1280x720 window
ARENA_LEFT = 40
ARENA_TOP = 40
ARENA_WIDTH = 1200
ARENA_HEIGHT = 640
ARENA_RIGHT = ARENA_LEFT + ARENA_WIDTH
ARENA_BOTTOM = ARENA_TOP + ARENA_HEIGHT


class SnakeGameHeadless:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        
        # Center snake aligned to the grid
        start_x = ARENA_LEFT + (ARENA_WIDTH // (2 * self.grid_size)) * self.grid_size
        start_y = ARENA_TOP + (ARENA_HEIGHT // (2 * self.grid_size)) * self.grid_size
        
        self.head = Point(start_x, start_y)
        self.body = [
            self.head,
            Point(self.head.x - self.grid_size, self.head.y),
            Point(self.head.x - (2 * self.grid_size), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        cols = ARENA_WIDTH // self.grid_size
        rows = ARENA_HEIGHT // self.grid_size
        while True:
            fx = ARENA_LEFT + random.randint(1, cols - 2) * self.grid_size
            fy = ARENA_TOP + random.randint(1, rows - 2) * self.grid_size
            self.food = Point(fx, fy)
            if self.food not in self.body:
                break

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. Update position based on [straight, right, left]
        self._move(action)
        eating = self.head == self.food
        if not eating:
            self.body.pop()  # The tail leaves its cell before collisions are checked
        self.body.insert(0, self.head)

        # 2. Check collision or starvation (looping in circles)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.body):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 3. Eat fruit (starvation timer restarts on every fruit)
        if eating:
            self.score += 1
            reward = 10
            self.frame_iteration = 0
            self._place_food()

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Wall collisions
        if (pt.x < ARENA_LEFT or pt.x >= ARENA_RIGHT or
            pt.y < ARENA_TOP or pt.y >= ARENA_BOTTOM):
            return True

        # Self collision
        if pt in self.body[1:]:
            return True

        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]       # Straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Turn right
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Turn left

        self.direction = new_dir
        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += self.grid_size
        elif self.direction == Direction.LEFT:
            x -= self.grid_size
        elif self.direction == Direction.DOWN:
            y += self.grid_size
        elif self.direction == Direction.UP:
            y -= self.grid_size

        self.head = Point(x, y)
