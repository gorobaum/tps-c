#!/usr/bin/python2

import os
import sys
import string
import math
import pygame
from pygame.locals import *

import cairo

# CONSTANTS
SCREEN_SIZE = [1024, 750]
TRACK_SURFACE_SIZE = [2000, 2000]
INITIAL_POINT = (TRACK_SURFACE_SIZE[0] / 2, TRACK_SURFACE_SIZE[1] / 2)
TRACK_WIDTH = 13

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255,255,0)
MAGENTA = (255,0,255)
CYAN = (0,255,255)
ORANGE = (255, 165, 0)
WATER_GREEN = (0, 230, 75)
PURPLE = (165, 0, 255)
BROWN = (100, 100, 0)
DARK_PURPLE = (100, 0, 100)
BLUE_GRAY = (0, 100, 100)
PINK = (255, 210, 255)

FOREST_GREEN = (34, 139, 34)
YELLOW_GREEN = (153, 204, 50)
AQUAMARINE = (112, 219, 147)
ROYALL_BLUE1 = (72, 118, 255)
SLATE_BLUE = (106, 90, 205)
LIME_GREEN = (50, 205, 50)
DARK_ORCHID = (153, 50, 204)

SHORT_STRAIGHT = (110, 255, 255)
MEDIUM_STRAIGHT = (10, 110, 245)
LONG_STRAIGHT = (0, 10, 90)
HAIRPIN = (255, 10, 10)
ELBOW = (255, 130, 0)
SHORT_HARD = (185, 165, 10)
LONG_HARD = (245, 245, 10)
SHORT_MEDIUM = (150, 10, 190)
LONG_MEDIUM = (235, 190, 255)
SHORT_EASY = (5, 100, 5)
LONG_EASY = (10, 240, 10)

error_message = """
ERROR: no input file provided
    Usage: generateKeyPoints.py <FILE1> <FILE2> [SCALE_FACTOR] [WIDTH HEIGHT]
    FILE1: Reference image.
    FILE2: Target image.
    SCALE_FACTOR: the greater the factor, the larger the zoom.
    WIDTH HEIGHT: screen size.
"""

helpMessage = """
INTERACTION:
       Arrow keys: move image;
         + - keys: to scale image;
            v key: to create a screenshot .png file;
    ESC BACKSPACE: quit.
"""


def getReferenceFileName():
    if len(sys.argv) < 3:
        print(error_message)
        sys.exit()

    return sys.argv[1]


def getTargetFileName():
    if len(sys.argv) < 3:
        print(error_message)
        sys.exit()

    return sys.argv[2]


def getScaleFactor():
    scale_factor = 2
    if len(sys.argv) >= 4:
        scale_factor = float(sys.argv[3])

    return scale_factor


def getScreenSize():
    width = SCREEN_SIZE[0]
    height = SCREEN_SIZE[1]
    if len(sys.argv) >= 6:
        width = int(sys.argv[4])
        height = int(sys.argv[5])

    return [width, height]

def draw_points(clear, surface, points, color):
    if clear:
        surface.fill((200, 200, 200))

    last_point = points[0]
    for point in points[1:]:
        if math.isnan(point[0]) or math.isnan(point[1]):
            print("NAN found!")
            continue
        pygame.draw.line(surface, color, last_point, point)
        last_point = point


def draw_grid(clear, surface, data, clusters, right_points, left_points, color):
    if clear:
        surface.fill((200, 200, 200))

    accumulated_distance = data[0][0]
    i = 1

    pygame.draw.line(surface, color, right_points[i], left_points[i])
    while i < len(data):
        accumulated_distance += data[i][0]
        i += 1
        pygame.draw.line(surface, color, right_points[i], left_points[i])


def draw_edges(clear, surface, data, clusters, right_points, left_points,
               cluster_edge_color):
    if clear:
        surface.fill((200, 200, 200))

    accumulated_distance = data[0][0]
    i = 1
    c = 0

    while i < len(data):
        accumulated_distance += data[i][0]
        i += 1

        if c < len(clusters) and accumulated_distance >= clusters[c][0]:
            pygame.draw.line(surface, cluster_edge_color, right_points[i],
                             left_points[i], 3)
            c += 1


def draw_lables(clear, surface, data, clusters, right_points, left_points):
    if clear:
        surface.fill((200, 200, 200))

    accumulated_distance = data[0][0]
    accumulated_angle = data[0][0]
    i = 1
    c = 0
    labels = []

    while i < len(data):
        accumulated_distance += data[i][0]
        accumulated_angle += data[i][1]

        i += 1

        if c < len(clusters) and accumulated_distance >= clusters[c][0]:
            if c == 0:
                length = clusters[c][0]
            else:
                length = clusters[c][0] - clusters[c - 1][0]

            if clusters[c][1] == 0.0:
                radius = -1
            else:
                radius = length / abs(accumulated_angle)

            degrees = accumulated_angle * 180 / math.pi

            labels.append((["D: %d; L: %d" % (accumulated_distance, length),
                            "R: %.1f" % radius,
                            "A: %9f (%.2f*)" % (accumulated_angle, degrees)],
                           right_points[i], left_points[i]))
            c += 1
            accumulated_angle = 0

    # Draw distanceFromStart at the end of each cluster.
    verdana = pygame.font.Font(pygame.font.match_font("Verdana"), 14)
    alt = False
    for label in labels:
        a = label[1]
        b = label[2]
        if alt:
            c = (1.5 * b[0] - 0.5 * a[0], 1.5 * b[1] - 0.5 * a[1])
        else:
            c = (2.2 * b[0] - 1.2 * a[0], 2.2 * b[1] - 1.2 * a[1])
        alt = not alt
        pygame.draw.line(surface, WHITE, b, c)
        label_surface_d = verdana.render(label[0][0], True, WHITE, (100, 100, 100))
        label_surface_l = verdana.render(label[0][1], True, WHITE, (100, 100, 100))
        label_surface_a = verdana.render(label[0][2], True, WHITE, (100, 100, 100))
        surface.blit(label_surface_d, c)
        surface.blit(label_surface_l, (c[0], c[1] + 14))
        surface.blit(label_surface_a, (c[0], c[1] + 28))


def draw_clusters(clear, surface, data, clusters, right_points, left_points):
    if clear:
        surface.fill((200, 200, 200))

    colors = [SHORT_STRAIGHT, MEDIUM_STRAIGHT, LONG_STRAIGHT, HAIRPIN, ELBOW,
              SHORT_HARD, LONG_HARD, SHORT_MEDIUM, LONG_MEDIUM, SHORT_EASY,
              LONG_EASY]

    accumulated_distance = 0
    i = 0
    c = 0

    while i < len(data):
        # Handle the special case when the last piece of track is part of the
        # first cluster.
        j = c
        if j >= len(clusters):
            j = 0

        left_rev = left_points[i: i + 2]
        left_rev.reverse()
        pygame.draw.polygon(surface, colors[abs(clusters[j][2])], \
                            right_points[i: i + 2] + left_rev)

        # If current segment is the last of a cluster, then next one will use a
        # different color.
        accumulated_distance += data[i][0]
        if c < len(clusters) and accumulated_distance >= clusters[c][0]:
            c += 1

        i += 1


def clear(surface):
    if clear:
        surface.fill((255, 255, 255))

def handle_events(offset, scale, second_phase):
    running = True
    mousePos = (-1,-1)
    end_ref_phase = False
    step = 80   # The amount of pixels offset will move.
    factor = 0.2 # Proportion by which scale will increase/decrease.
    segmentation_changed = False
    should_export_screenshot = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == KEYUP:
            if event.key == K_ESCAPE or event.key == K_BACKSPACE or \
               event.key == K_q:

                running = False
        if event.type == MOUSEBUTTONUP:
          mousePos = pygame.mouse.get_pos()

        if event.type == KEYDOWN:
            if event.key == K_RIGHT:
                offset[0] -= step
            elif event.key == K_LEFT:
                offset[0] += step
            if event.key == K_UP:
                offset[1] += step
            elif event.key == K_DOWN:
                offset[1] -= step
            if event.key == K_v:
                should_export_screenshot = True
            if event.key == K_e:
                end_ref_phase = True
                if second_phase:
                  running = False
            if event.unicode == "+":
                scale += factor
                offset[0] -= (SCREEN_SIZE[0] / 2 - offset[0]) * factor
                offset[1] -= (SCREEN_SIZE[1] / 2 - offset[1]) * factor
            elif event.key == K_MINUS or event.key == K_KP_MINUS:
                scale -= factor
                if scale <= 0.0:
                    scale = 0.1
                offset[0] += (SCREEN_SIZE[0] / 2 - offset[0]) * factor
                offset[1] += (SCREEN_SIZE[1] / 2 - offset[1]) * factor

    return running, scale, offset, mousePos, end_ref_phase, segmentation_changed, \
           should_export_screenshot

def scale_surface(surface, scale):
    new_size = [int(x * scale) for x in surface.get_size()]
    surface = pygame.transform.scale(surface, new_size)
    TRACK_SURFACE_SIZE[0] = new_size[0]
    TRACK_SURFACE_SIZE[1] = new_size[1]

    return surface

def scale_points(points, scale):
    scaled_points = [[(val * scale) for val in tuple] for tuple in points]

    return scaled_points

# PNG CARA
def export_screenshot(surface, filename):
    filename = os.path.splitext(filename)[0] + "-result-" + ".png"

    print("Exporting screenshot to file: " + filename)
    pygame.image.save(surface, filename)
    print("Exportation finished.\n")

def drawRedCrosshair(surface, mousePos):
  x = mousePos[0]
  y = mousePos[1]
  for i, j in enumerate(range(-15, 15)):
    surface.set_at((x+i,y-j), RED)
    surface.set_at((x+i+1,y-j), RED)
    surface.set_at((x+i,y-j-1), RED)
    surface.set_at((x+i,y+j), RED)
    surface.set_at((x+i-1,y+j), RED)
    surface.set_at((x+i,y+j+1), RED)

def printVectorForC(vec):
  out = " = {"
  for i in vec:
    out += "{%s, %s}, " %tuple(i)
  out = out[:-2]
  out += "}"
  print out

def main():
    global SCREEN_SIZE

    referenceFileName = getReferenceFileName()
    targetFileName = getTargetFileName()

    referenceSurface = pygame.image.load(referenceFileName)
    targetSurface = pygame.image.load(targetFileName)

    scaleFactor = getScaleFactor()
    # SCREEN_SIZE = referenceSurface.get_size()

    print(helpMessage)

    pygame.init()
    pygame.key.set_repeat(100, 10)

    window = pygame.display.set_mode(SCREEN_SIZE)
    drawSurface = pygame.Surface(referenceSurface.get_size(), pygame.SRCALPHA, 32)
    drawSurface.blit(referenceSurface, (0,0))
    windowSurface = pygame.Surface(referenceSurface.get_size(), pygame.SRCALPHA, 32)
    windowSurface.blit(referenceSurface, (0,0))
    targetKPSurface = pygame.Surface(referenceSurface.get_size(), pygame.SRCALPHA, 32)
    targetKPSurface.blit(targetSurface, (0,0))

    # draw_all(track_surface, data, clusters, points, right_points, left_points, \
    #         show_clusters, show_grid, show_edges, show_labels)

    referencePoints = []
    targetPoints =    []
    currentPoints = referencePoints

    for point in referencePoints:
      drawRedCrosshair(drawSurface, point)

    second_phase = False
    currentKP = len(referencePoints)

    scale = 1.0
    window.blit(windowSurface, (0,0))
    offset = [0, 0]
    running = True
    filename = getReferenceFileName()
    while running:

        running, scale, offset, mousePos, end_ref_phase, segmentation_changed, should_export_screenshot,\
         = handle_events(offset, scale, second_phase)

        window.fill(GREEN)
        windowSurface = scale_surface(drawSurface, scale)

        # if mousePos[0] == -1:
        # print mousePos
        if second_phase:
          if currentKP != len(referencePoints):
            drawRedCrosshair(drawSurface, referencePoints[currentKP])
          else:
            running = False

        if mousePos[0] != -1:
            currentMousePos = [mousePos[0] - int(offset[0]), mousePos[1] - int(offset[1])]
            currentMousePos = [int(currentMousePos[0]/scale), int(currentMousePos[1]/scale)]
            if currentMousePos[0] >= 0 and currentMousePos[1] >= 0 :
              currentPoints.append(currentMousePos)
              drawRedCrosshair(drawSurface, currentMousePos)
              if second_phase:
                drawRedCrosshair(targetKPSurface, currentMousePos)
                currentKP = currentKP + 1


        if end_ref_phase:
          export_screenshot(drawSurface, getReferenceFileName())
          drawSurface.blit(targetSurface, (0,0))
          windowSurface.blit(targetSurface, (0,0))
          currentPoints = targetPoints
          filename = getTargetFileName()
          second_phase = True
          for point in targetPoints:
            drawRedCrosshair(drawSurface, point)

        window.blit(windowSurface, offset)

        if should_export_screenshot:
            export_screenshot(drawSurface, filename)

        pygame.display.update()
    print referencePoints
    export_screenshot(targetKPSurface, getTargetFileName())
    printVectorForC(referencePoints)
    printVectorForC(targetPoints)
    pygame.quit()


main()