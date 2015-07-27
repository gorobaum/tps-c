import sys

numFiles = 0

n = int(sys.argv[1])
for i in range(1, n):
  numFiles = numFiles + 1
  fileName = 'tps.config'+str(numFiles)
  filepath = '../configurations/tps.config'+str(numFiles)
  f = open(filepath,'w')
  f.write('images/vocal/test/IM_0420.png\n')
  porcentage = 0.04
  name = 'result-'+str(i*0.5)+'\n'
  f.write(name)
  f.write(str(porcentage)+'\n')
  distanceMetric = i*0.2
  f.write(str(distanceMetric)+'\n')
  nFeatures = 0
  f.write(str(nFeatures)+'\n')
  nOctavesLayers = 3
  f.write(str(nOctavesLayers)+'\n')
  contrastThreshold = 0.04
  f.write(str(contrastThreshold)+'\n')
  edgeThreshold = 10
  f.write(str(edgeThreshold)+'\n')
  sigma = 1.6
  f.write(str(sigma)+'\n')
  f.close()

f = open('../tps.configs', 'w')
f.write('images/vocal/test/IM_0414.png\n')
f.write('20\n')
for x in range(1, numFiles+1, 1):
  filename = 'configurations/tps.config'+str(x)+'\n'
  f.write(filename)
f.close()