import argparse
import os
from os.path import isfile, join

"Handy script to create hpc job files starting from a directory of experiment."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory containing experiment files")
    parser.add_argument("-c", "--cpus", type=int, required=False, default=4, help="Number of cpus")
    parser.add_argument("-m", "--memory", type=int, required=False, default=10, help="Gigabytes of volatile memory")
    parser.add_argument("-q", "--queue", type=str, required=False, default="short_gpuQ", help="Which queue to use")
    parser.add_argument("-w", "--walltime", type=str, required=False, default="06:00:00", help="Walltime")
    args = parser.parse_args()

    output = []
    output.append("#PBS -l select=1:ncpus=%s:mem=%sgb" % (args.cpus, args.memory))
    output.append("#PBS -q %s" % args.queue)
    output.append("#PBS -l walltime=%s" % args.walltime)
    output.append("module load python-3.5.2")
    output.append("module load cuda-9.0")
    output.append("module load BLAS")
    output = "\n".join(output)

    for dirpath, _, filenames in os.walk(args.directory):
        for f in filenames:
            abspath = os.path.abspath(join(dirpath, f))
            name = "job_%s.pbs" % abspath.split("/")[-1].split(".")[0]
            cmd = "python3.5 ~/constrained-adversarial-networks/src/main.py -i %s -f 0.9" % abspath
            cmd = "\n".join([output, cmd])

            with open(name, "w") as file:
                file.write(cmd)
