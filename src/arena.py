# this file contains everything about the area of the playing game
import pygame

def drawWalls(screen: pygame.Surface, x: int, y: int) -> None:
    
    pygame.draw.line(screen, "red", (2,2), (2,y))
