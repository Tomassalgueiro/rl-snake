# this file contains everything about the area of the playing game
import pygame

def drawWalls(screen: pygame.Surface, x: int, y: int) -> None:  
    pygame.draw.line(screen, "red", (24,20), (24,y-20), 10)
    pygame.draw.line(screen, "red", (20,20), (x-20,20), 10)
    pygame.draw.line(screen, "red", (20,y-20), (x-20,y-20), 10)
    pygame.draw.line(screen, "red", (x-25,20), (x-25,y-20), 10)
