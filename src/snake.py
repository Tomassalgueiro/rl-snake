from collections import deque
import pygame

OPPOSITES = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}

KEY_MAP = {
    pygame.K_w: "UP",
    pygame.K_UP: "UP",
    pygame.K_s: "DOWN",
    pygame.K_DOWN: "DOWN",
    pygame.K_a: "LEFT",
    pygame.K_LEFT: "LEFT",
    pygame.K_d: "RIGHT",
    pygame.K_RIGHT: "RIGHT",
}

class Snake:
    def __init__(self, start_pos: tuple[int, int], grid_size: int = 20) -> None:
        self.grid_size = grid_size
        self.direction = "RIGHT"
        self.direction_queue = deque(maxlen=2)
        
        # Body is a list of (x, y) coordinates; head is body[0]
        x, y = start_pos
        self.body = [
            (x, y),
            (x - grid_size, y),
            (x - (2 * grid_size), y)
        ]
        self.grow = False

    def change_direction(self, key_pressed: int) -> None:
        target_dir = KEY_MAP.get(key_pressed)
        if not target_dir:
            return

        ref_dir = self.direction_queue[-1] if self.direction_queue else self.direction
        if target_dir != ref_dir and target_dir != OPPOSITES[ref_dir]:
            self.direction_queue.append(target_dir)

    def step(self) -> None:
        # Commit direction from buffer
        if self.direction_queue:
            self.direction = self.direction_queue.popleft()

        head_x, head_y = self.body[0]

        if self.direction == "UP":
            head_y -= self.grid_size
        elif self.direction == "DOWN":
            head_y += self.grid_size
        elif self.direction == "LEFT":
            head_x -= self.grid_size
        elif self.direction == "RIGHT":
            head_x += self.grid_size

        new_head = (head_x, head_y)
        self.body.insert(0, new_head)

        if not self.grow:
            self.body.pop()
        else:
            self.grow = False

    def eat_fruit(self) -> None:
        self.grow = True

    def check_self_collision(self) -> bool:
        return self.body[0] in self.body[1:]

    def draw(self, surface: pygame.Surface) -> None:
        for idx, (x, y) in enumerate(self.body):
            color = "#8EE53F" if idx == 0 else "#61B329"
            pygame.draw.rect(surface, color, (x, y, self.grid_size, self.grid_size))
            pygame.draw.rect(surface, "dark green", (x, y, self.grid_size, self.grid_size), 1)
