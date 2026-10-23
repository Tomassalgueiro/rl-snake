import random
import pygame
import arena
from snake import Snake

pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
GRID_SIZE = 20

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake RL Playground")
clock = pygame.time.Clock()

ARENA_RECT = pygame.Rect(40, 40, 1200, 640)

start_x = ARENA_RECT.left + (ARENA_RECT.width // (2 * GRID_SIZE)) * GRID_SIZE
start_y = ARENA_RECT.top + (ARENA_RECT.height // (2 * GRID_SIZE)) * GRID_SIZE
snake = Snake(start_pos=(start_x, start_y), grid_size=GRID_SIZE)

def spawn_food(snake_body: list[tuple[int, int]]) -> tuple[int, int]:
    cols = ARENA_RECT.width // GRID_SIZE
    rows = ARENA_RECT.height // GRID_SIZE
    while True:
        fx = ARENA_RECT.left + random.randint(1, cols - 2) * GRID_SIZE
        fy = ARENA_RECT.top + random.randint(1, rows - 2) * GRID_SIZE
        if (fx, fy) not in snake_body:
            return (fx, fy)

food_pos = spawn_food(snake.body)

MOVE_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(MOVE_EVENT, 100)

running = True
game_over = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and game_over:
                snake = Snake(start_pos=(start_x, start_y), grid_size=GRID_SIZE)
                food_pos = spawn_food(snake.body)
                game_over = False
            else:
                snake.change_direction(event.key)

        elif event.type == MOVE_EVENT and not game_over:
            snake.step()

            head_x, head_y = snake.body[0]
            if (head_x < ARENA_RECT.left or head_x >= ARENA_RECT.right or
                head_y < ARENA_RECT.top or head_y >= ARENA_RECT.bottom):
                game_over = True

            if snake.check_self_collision():
                game_over = True

            if (head_x, head_y) == food_pos:
                snake.eat_fruit()
                food_pos = spawn_food(snake.body)

    screen.fill("black")
    arena.draw_arena(screen, ARENA_RECT)
    arena.draw_grid_lines(screen, ARENA_RECT, cell_size=GRID_SIZE)

    # Draw food
    pygame.draw.rect(screen, "red", (food_pos[0], food_pos[1], GRID_SIZE, GRID_SIZE))

    # Draw snake
    snake.draw(screen)

    if game_over:
        font = pygame.font.SysFont(None, 48)
        text = font.render("GAME OVER - Press R to Restart", True, "white")
        screen.blit(text, text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
