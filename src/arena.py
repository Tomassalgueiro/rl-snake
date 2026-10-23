# this file contains everything about the area of the playing game
import pygame

def drawWalls(screen: pygame.Surface, x: int, y: int) -> None:  
    wallColor = pygame.color.Color("#C14A09")
    pygame.draw.line(screen, wallColor, (24,20), (24,y-20), 10)
    pygame.draw.line(screen, wallColor, (20,20), (x-20,20), 10)
    pygame.draw.line(screen, wallColor, (20,y-20), (x-20,y-20), 10)
    pygame.draw.line(screen, wallColor, (x-25,20), (x-25,y-20), 10)

def drawGrid(screen: pygame.Surface, x: int, y: int) -> None:  
    pygame.draw.flood_fill(screen, "dark green", (x/2,y/2))
