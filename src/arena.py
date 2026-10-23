# this file contains everything about the area of the playing game
import pygame

def draw_arena(screen: pygame.Surface, play_rect: pygame.Rect) -> None:
    # Fill play area with green
    pygame.draw.rect(screen, "#1B3B1B", play_rect)

    # Draw border walls
    wall_color = pygame.Color("#C14A09")
    pygame.draw.rect(screen, wall_color, play_rect, width=8)

def draw_grid_lines(screen: pygame.Surface, play_rect: pygame.Rect, cell_size: int = 20) -> None:
    line_color = (20, 50, 20)
    for x in range(play_rect.left, play_rect.right, cell_size):
        pygame.draw.line(screen, line_color, (x, play_rect.top), (x, play_rect.bottom))
    for y in range(play_rect.top, play_rect.bottom, cell_size):
        pygame.draw.line(screen, line_color, (play_rect.left, y), (play_rect.right, y))
