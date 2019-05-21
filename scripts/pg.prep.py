import sys

if len(sys.argv) < 6:
    print("please specify input file, job name and for each command: time(hours), memory(gb), #cpus #jobsPerScript)")
    exit(1)

def createJob(idx, cmd):
    name = sys.argv[2] + '.' + str(idx)
    print("creating: " + name)
    outFile = open(name, 'w')
    outFile.write("#!/bin/bash\n")
    outFile.write('\n')
    outFile.write("#SBATCH --time=" + sys.argv[3] + ":00:00\n")
    outFile.write("#SBATCH --nodes=1\n")
    outFile.write("#SBATCH --ntasks=1\n")
    outFile.write("#SBATCH --mem=" + sys.argv[4] + 'G\n')
    outFile.write("#SBATCH --cpus-per-task=" + sys.argv[5] + '\n')
    outFile.write("#SBATCH --job-name=" + name + '\n')
    outFile.write("#SBATCH --output=" + name + '.log\n')
    #outFile.write("#SBATCH --mail-type=BEGIN,END,FAIL\n")
    #outFile.write("#SBATCH --mail-user=robvanderg@live.nl\n")
    #outFile.write("#SBATCH --partition=himem\n")
    outFile.write("\n")
    outFile.write("module load DyNet/2.0.3-foss-2018a-Python-2.7.14\n")
    #outFile.write("module load Python/3.6.4-intel-2018a\n")
    outFile.write(cmd + "\n")
    outFile.close()


cmds = []
for line in open(sys.argv[1]):
    line = line.strip()
    if len(line) > 2 and line[0] != '#':
        cmds.append(line)

idx = 1
numCmds = int(sys.argv[6])
splits = int(len(cmds) / numCmds)
for i in range(0, splits):
    cmd = '\n'.join(cmds[i * numCmds:(i+1) * numCmds])
    createJob(idx, cmd)
    idx += 1
if (len(cmds)%numCmds != 0):
    cmd = '\n'.join(cmds[splits * numCmds:])
    createJob(idx, cmd)
    
    


