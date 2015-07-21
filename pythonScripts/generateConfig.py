import sys

numFiles = 0

n = int(sys.argv[1])
for i in range(2, n, 2):
  numFiles = numFiles + 1
  fileName = 'tps.config'+str(numFiles)
  filepath = '../configurations/tps.config'+str(numFiles)
  f = open(filepath,'w')
  f.write('images/vocal/test/IM_0415.png.png\n')
  porcentage = i*0.01
  name = 'result-'+str(porcentage)+'\n'
  f.write(name)
  f.write(str(porcentage)+'\n')

f = open('../tps.configs', 'w')
f.write('images/IM_0414.png.png\n')
f.write('20\n')
for x in range(1, numFiles+1, 1):
  filename = 'configurations/tps.config'+str(x)+'\n'
  f.write(filename)