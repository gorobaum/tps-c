import sys
import scipy
import math
import numpy as np
from scipy import ndimage

def deformRotate(imagePixels):
  xc = imagePixels.shape[0]/2
  yc = imagePixels.shape[1]/2
  print(xc)
  deformedPixels = np.ndarray(imagePixels.shape)
  deformedPixels.fill(255)
  for x in range(-128, 128):
    for y in range(-128,128):
      X = x*math.sin(-math.pi/72)-y*math.cos(-math.pi/72)
      Y = x*math.cos(-math.pi/72)+y*math.sin(-math.pi/72)
      deformedPixels[x+xc,y+yc] = bilinear(imagePixels, X+xc, Y+yc)
  return deformedPixels

def deformSinusiodal(imagePixels):
  deformedPixels = np.ndarray(imagePixels.shape)
  print(deformedPixels.shape)
  deformedPixels.fill(255)
  for col in range(imagePixels.shape[1]):
    for row in range(imagePixels.shape[0]):
      newCol = col+8.0*math.sin(row/16.0)
      newRow = row-4.0*math.cos(col/32.0)
      if newCol <= 0 or newCol >= imagePixels.shape[1]:
        continue
      if newRow <= 0 or newRow >= imagePixels.shape[0]:
        continue
      deformedPixels[row,col] = bilinear(imagePixels, newCol, newRow)
  return deformedPixels

def bilinear(imagePixels, col, row):
  u = math.trunc(col)
  v = math.trunc(row)
  interpolation = (u+1-col)*(v+1-row)*getPixel(imagePixels,u,v) + (col-u)*(v+1-row)*getPixel(imagePixels,u+1,v) + (u+1-col)*(row-v)*getPixel(imagePixels,u,v+1) + (col-u)*(row-v)*getPixel(imagePixels,u+1,v+1)
  return interpolation

def getPixel(pixels, col, row):
  # print width, height, x, y
  h = pixels.shape[0]
  w = pixels.shape[1]
  if col >= w or col < 0:
    return 0.0
  elif row >= h or row < 0:
    return 0.0
  else:
    return pixels[row, col]

staticImage = scipy.misc.imread(sys.argv[1], True)
movingImage = deformRotate(staticImage)
scipy.misc.imsave(sys.argv[2], movingImage)