import sys

numFiles = 0

n = int(sys.argv[1])
for i in range(2, n, 2):
  numFiles = numFiles + 1
  fileName = 'tps.config'+str(numFiles)
  filepath = '../configurations/tps.config'+str(numFiles)
  f = open(filepath,'w')
  f.write('images/'+sys.argv[2]+'\n')
  porcentage = i*0.01
  name = 'result-'+str(porcentage)+'\n'
  f.write(name)
  f.write(str(porcentage)+'\n')

f = open('../tps.configs', 'w')
f.write('images/'+sys.argv[3]+'\n')
for x in range(1, numFiles+1, 1):
  filename = 'configurations/tps.config'+str(x)+'\n'
  f.write(filename)