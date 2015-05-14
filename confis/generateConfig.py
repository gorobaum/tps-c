import sys

numFiles = 0

for i in range(2,51, 2):
  numFiles = numFiles + 1
  fileName = 'tps.config'+str(numFiles)
  f = open(fileName,'w')
  f.write('images/grid-Ref.png\n')
  f.write('images/grid-Tar.png\n')
  porcentage = i*0.01
  name = 'result-'+str(porcentage)+'\n'
  f.write(name)
  f.write(str(porcentage)+'\n')

f = open('tps.configs', 'w')
for x in range(1, numFiles+1, 1):
  filename = 'confis/tps.config'+str(x)+'\n'
  f.write(filename)