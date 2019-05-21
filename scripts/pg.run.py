import sys
import os

if len(sys.argv) < 2:
    print("please give path")

for path in sys.argv[1:]:
    os.system('sbatch ' + path)
