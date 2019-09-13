import numpy as np
from model.model import AgentMobilityModel

x_size = 30
y_size = 30


def read_iteration(file):
    corr_count = int(file.readline())   # Read the next number of transportations
    corr = np.zeros((corr_count, 4), dtype=np.float32)
    for i in range(corr_count):
        arr = file.readline().split('\t')
        corr[i][0] = int(arr[0])    # Core "from"
        corr[i][1] = int(arr[1])    # Core "to"
        corr[i][2] = int(arr[2])    # Number of agents
        corr[i][3] = float(arr[3])  # Velocity
    return corr


def read_transportations(path):
    file = open(path, 'r')
    iters = int(file.readline())    # Read the number of lines in file
    transportations = list()
    iter = 0
    while iter < iters:
        transportations.append(read_iteration(file))  # Append all next transportations
        iter += 1
    file.close()
    return transportations  # List of groups of transportations.
    # Each group of transportations - all next transp. at next time.
    # Each transportation - 4 values


def read_schedule(sched_path):
    file = open(sched_path, 'r')
    sched = np.zeros((x_size, y_size), dtype=np.int32)
    for line in file.readlines():
        arr = line.split("\t")
        p = r2m(int(arr[0]))    # (x, y)
        v = int(arr[1])         # Number of core
        sched[p[0]][p[1]] = v
    file.close()
    return sched


def r2m(idx):
    return idx // y_size, idx % y_size  # Return "x" and "y" from number


def corresponds_modelling_velo(transportations, schedules, cores, kernels = 0):
    if kernels != 0:
        ammodel = AgentMobilityModel(x_size, y_size, transportations, cores, kernels)
        ammodel.interactive_simulation2_velo_on_kernel(schedules)
    else:
        ammodel = AgentMobilityModel(x_size, y_size, transportations, cores)
        ammodel.interactive_simulation2_velo(schedules)


def main():
    # select a movement scenario from
    transportations_file = "resources\\spb_passengers_center_100k_1"
    transportations = read_transportations(transportations_file)
    cores = 9
    kernels = 5
    schedules = dict()  # Dictionary of 10. Each value - schedule
                        # Each schedule - matrix x on y; each cell - number of core

    # for optimized scenario
    for iters in range(10):
        iter_sched = read_schedule(
            "schedules\\multiple\\10_include\\spb_schedule_velo_{}_{}.sched".format(iters * 144, (iters + 1) * 144))
        schedules[iters * 144] = iter_sched

    # for default case
    # default = read_schedule("schedules\\default")
    # schedules[0] = default
    corresponds_modelling_velo(transportations, schedules, cores, kernels)

if __name__ == "__main__":
    main()