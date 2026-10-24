import random
from collections import namedtuple
from enum import Enum
import numpy as np
import pygame
import arena

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
GRID_SIZE = 20
ARENA_RECT = pygame.Rect(40, 40, 1200, 640)
SPEED = 60  # Altera a velocidade visual (ex: 60 ou 120 para treinar mais rápido)


class SnakeGameAI:
    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
        self.w = w
        self.h = h
        self.grid_size = GRID_SIZE
        self.arena_rect = ARENA_RECT
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake RL Playground")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        
        # Posição inicial no centro da arena alinhada à grelha
        start_x = self.arena_rect.left + (self.arena_rect.width // (2 * self.grid_size)) * self.grid_size
        start_y = self.arena_rect.top + (self.arena_rect.height // (2 * self.grid_size)) * self.grid_size
        
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
        cols = self.arena_rect.width // self.grid_size
        rows = self.arena_rect.height // self.grid_size
        while True:
            fx = self.arena_rect.left + random.randint(1, cols - 2) * self.grid_size
            fy = self.arena_rect.top + random.randint(1, rows - 2) * self.grid_size
            self.food = Point(fx, fy)
            if self.food not in self.body:
                break

    def play_step(self, action):
        self.frame_iteration += 1

        # 1. Tratar eventos (fechar janela)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Executar ação e mover
        self._move(action)
        eating = self.head == self.food
        if not eating:
            self.body.pop()  # A cauda sai da célula antes de verificar colisões
        self.body.insert(0, self.head)

        # 3. Verificar colisões ou loop infinito
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.body):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Comer fruta (o temporizador de fome recomeça a cada fruta)
        if eating:
            self.score += 1
            reward = 10
            self.frame_iteration = 0
            self._place_food()

        # 5. Renderização
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Colisão com as paredes da arena
        if (pt.x < self.arena_rect.left or pt.x >= self.arena_rect.right or
            pt.y < self.arena_rect.top or pt.y >= self.arena_rect.bottom):
            return True

        # Colisão com o próprio corpo
        if pt in self.body[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill("black")
        
        # Desenha a arena e grelha com base nas funções do teu arena.py
        arena.draw_arena(self.display, self.arena_rect)
        arena.draw_grid_lines(self.display, self.arena_rect, cell_size=self.grid_size)

        # Desenha a comida
        pygame.draw.rect(
            self.display, 
            "red", 
            (self.food.x, self.food.y, self.grid_size, self.grid_size)
        )

        # Desenha a cobra com o teu esquema de cores
        for idx, pt in enumerate(self.body):
            color = "#8EE53F" if idx == 0 else "#61B329"
            pygame.draw.rect(
                self.display, 
                color, 
                (pt.x, pt.y, self.grid_size, self.grid_size)
            )
            pygame.draw.rect(
                self.display, 
                "dark green", 
                (pt.x, pt.y, self.grid_size, self.grid_size), 
                1
            )

        # Pontuação no ecrã
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, "white")
        self.display.blit(score_text, (self.arena_rect.left, 10))

        pygame.display.flip()

    def _move(self, action):
        # Sentido horário: RIGHT -> DOWN -> LEFT -> UP
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]       # Seguir em frente
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Virar à direita
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Virar à esquerda

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
