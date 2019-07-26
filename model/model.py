import numpy as np
from numpy import random as rnd
from deap import creator
from sys import stdout
from scipy.spatial import distance
import matplotlib.pyplot as plt

agent_model_time = 0.001
agent_transfer_time = 0.005


def agent_model_time_func(agents):
    stable_time = agents * agent_model_time
    return stable_time + rnd.normal(loc=stable_time * 0.1, scale=stable_time * 0.01)


v_agent_transfer_time = np.vectorize(agent_model_time_func)


def r2m(idx):
    return idx // 30, idx % 30

class AgentMobilityModel:
    def __init__(self, x_size, y_size, transportations, cores):
        self.x_size = x_size
        self.y_size = y_size
        self.transportations = transportations
        self.iterations = len(transportations)
        self.cores = cores
        self.ev_start = 0
        self.ev_end = self.iterations
        self.field = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        self.field_velo = np.zeros((self.x_size, self.y_size), dtype=np.float32)
        # self.init_simulation()



    def init_simulation(self):
        for i in range(self.ev_start):
            # iteration
            iteration_data = self.transportations[i]
            for corr in iteration_data:
                corr0 = int(corr[0])
                corr1 = int(corr[1])
                corr2 = int(corr[2])
                if corr0 == corr1:
                    p = r2m(corr0)
                    self.field[p[0]][p[1]] += corr2
                    self.field_velo[p[0]][p[1]] += corr[3]
                else:
                    p1 = r2m(corr0)
                    p2 = r2m(corr1)
                    self.field[p1[0]][p1[1]] -= corr2
                    self.field_velo[p1[0]][p1[1]] -= corr[3]
                    self.field[p2[0]][p2[1]] += corr2
                    self.field_velo[p2[0]][p2[1]] += corr[3]

    def simulation(self, schedule):
        # initialization
        # model_times = np.zeros(self.iterations)
        # transfer_times = np.zeros(self.iterations)
        total_times = np.zeros(self.ev_end)
        schedule = schedule
        field = np.copy(self.field)
        field_velo = np.copy(self.field_velo)

        for i in range(self.ev_start, self.ev_end):
            # iteration
            iteration_data = self.transportations[i]
            cores_model = np.zeros(self.cores)
            cores_velocities = np.zeros(self.cores)
            cores_transfers = np.zeros(self.cores)

            for corr in iteration_data:
                corr0 = int(corr[0])
                corr1 = int(corr[1])
                corr2 = int(corr[2])
                if corr0 == corr1:
                    p = r2m(corr0)
                    field[p[0]][p[1]] += corr2
                    field_velo[p[0]][p[1]] += corr[3]
                else:
                    p1 = r2m(corr0)
                    p2 = r2m(corr1)
                    field[p1[0]][p1[1]] -= corr2
                    field_velo[p1[0]][p1[1]] -= corr[3]
                    field[p2[0]][p2[1]] += corr2
                    field_velo[p2[0]][p2[1]] += corr[3]
                    core1 = schedule[p1[0]][p1[1]]
                    core2 = schedule[p2[0]][p2[1]]
                    if core1 != core2:
                        cores_transfers[core1 - 1] += corr2
                        cores_transfers[core2 - 1] += corr2
            for x in range(self.x_size):
                for y in range(self.y_size):
                    cores_model[schedule[x][y] - 1] += field[x][y]
                    cores_velocities[schedule[x][y] - 1] += field_velo[x][y]

            cores_transfers = cores_transfers * agent_transfer_time
            cores_model = v_agent_transfer_time(cores_model)
            velo_coef = 90000000.0
            iter_velocities = cores_model + np.abs(cores_velocities / velo_coef)
            total_model_time = iter_velocities + cores_transfers

            # model_times[i] = cores_model.max()
            # transfer_times[i] = cores_transfers.max()
            total_times[i] = total_model_time.max()
        result = total_times[self.ev_start:self.ev_end].sum()
        return result,

    def simulation_backup(self, schedule):
        # initialization
        # model_times = np.zeros(self.iterations)
        # transfer_times = np.zeros(self.iterations)
        total_times = np.zeros(self.ev_end)
        schedule = schedule
        field = np.copy(self.field)

        for i in range(self.ev_start, self.ev_end):
            # iteration
            iteration_data = self.transportations[i]
            cores_model = np.zeros(self.cores)
            cores_transfers = np.zeros(self.cores)

            for corr in iteration_data:
                if corr[0] == corr[1]:
                    p = r2m(corr[0])
                    field[p[0]][p[1]] += corr[2]
                else:
                    p1 = r2m(corr[0])
                    p2 = r2m(corr[1])
                    field[p1[0]][p1[1]] -= corr[2]
                    field[p2[0]][p2[1]] += corr[2]
                    core1 = schedule[p1[0]][p1[1]]
                    core2 = schedule[p2[0]][p2[1]]
                    if core1 != core2:
                        cores_transfers[core1 - 1] += corr[2]
                        cores_transfers[core2 - 1] += corr[2]
            for x in range(self.x_size):
                for y in range(self.y_size):
                    cores_model[schedule[x][y] - 1] += field[x][y]

            cores_transfers = cores_transfers * agent_transfer_time
            cores_model = v_agent_transfer_time(cores_model)
            total_model_time = cores_model + cores_transfers

            # model_times[i] = cores_model.max()
            # transfer_times[i] = cores_transfers.max()
            total_times[i] = total_model_time.max()
        result = total_times[self.ev_start:self.ev_end].sum()
        return result,

    def interactive_simulation(self, schedules, slides=5):
        max_value = 39
        # initialization
        iters = len(self.transportations)
        model_times = np.zeros(iters)
        transfer_times = np.zeros(iters)
        total_times = np.zeros(iters)
        field = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        schedule = schedules[0]
        plt.ion()
        plt.imshow(field, vmax=max_value)
        plt.contour(schedule, alpha=0.5, cmap='Set1')
        plt.tight_layout()
        plt.show()
        plt.pause(0.0000001)

        cores_modeling_time = np.zeros((self.cores, 1440))
        cores_transfer_time = np.zeros((self.cores, 1440))
        cores_total_time = np.zeros((self.cores, 1440))

        for i in range(0, 1440):
            if i in schedules.keys():
                schedule = schedules[i]
            # iteration
            iteration_data = self.transportations[i]
            cores_model = np.zeros(self.cores)
            cores_transfers = np.zeros(self.cores)

            for corr in iteration_data:
                if corr[0] == corr[1]:
                    p = r2m(corr[0])
                    field[p[0]][p[1]] += corr[2]
                else:
                    p1 = r2m(corr[0])
                    p2 = r2m(corr[1])
                    field[p1[0]][p1[1]] -= corr[2]
                    field[p2[0]][p2[1]] += corr[2]
                    core1 = schedule[p1[0]][p1[1]]
                    core2 = schedule[p2[0]][p2[1]]
                    if core1 != core2:
                        cores_transfers[core1] += corr[2]
                        cores_transfers[core2] += corr[2]
            for x in range(self.x_size):
                for y in range(self.y_size):
                    cores_model[schedule[x][y] - 1] += field[x][y]

            cores_transfers = cores_transfers * agent_transfer_time
            cores_model = v_agent_transfer_time(cores_model)
            total_model_time = cores_model + cores_transfers

            cores_modeling_time[:, i] = cores_model
            cores_transfer_time[:, i] = cores_transfers
            cores_total_time[:, i] = total_model_time


            model_times[i] = cores_model.max()
            transfer_times[i] = cores_transfers.max()
            total_times[i] = total_model_time.max()
            # draw output
            if i % slides == 0:
                plt.imshow(field, vmax=max_value)
                plt.contour(schedule, cmap='Set1', alpha=0.4, linestyles="--")
                plt.tight_layout()
                plt.show()
                plt.pause(0.000001)
                plt.clf()
            stdout.write("\riteration=%d" % i)
            stdout.flush()
        plt.close()
        plt.ioff()
        print("\nmodel sum={}".format(model_times.sum()))
        print("transfer sum={}".format(transfer_times.sum()))
        print("total time={}".format(total_times.sum()))
        x_range = np.arange(0, iters)
        plt.figure()
        plt.plot(x_range, model_times, label='model')
        plt.plot(x_range, transfer_times, label='transfer')
        plt.plot(x_range, total_times, label='total')
        plt.legend()
        plt.tight_layout()
        # plt.show()

        for c in range(self.cores):
            plt.figure(figsize=(4,3))
            plt.plot(x_range, cores_modeling_time[c, :], label='model')
            plt.plot(x_range, cores_transfer_time[c, :], label='transfer')
            plt.plot(x_range, cores_total_time[c, :], label='total')
            plt.legend()
            plt.title("Core {}".format(c))
            plt.tight_layout()
        plt.show()

    def interactive_simulation2_velo(self, schedules, slides=50):
        max_value = 39
        # initialization
        iters = len(self.transportations)
        model_times = np.zeros(iters)
        transfer_times = np.zeros(iters)
        total_times = np.zeros(iters)
        velocities = np.zeros(iters)
        field = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        field_velo = np.zeros((self.x_size, self.y_size), dtype=np.float32)
        schedule = schedules[0]
        plt.ion()
        plt.imshow(field, vmax=max_value)
        plt.contour(schedule, alpha=0.5, cmap='Set1')
        plt.tight_layout()
        plt.show()
        plt.pause(0.0000001)

        cores_modeling_time = np.zeros((self.cores, 1440))
        cores_transfer_time = np.zeros((self.cores, 1440))
        cores_total_time = np.zeros((self.cores, 1440))
        cores_velocities = np.zeros((self.cores, 1440))

        for i in range(0, 1440):
            if i in schedules.keys():
                schedule = schedules[i]
            # iteration
            iteration_data = self.transportations[i]
            cores_model = np.zeros(self.cores)
            cores_velo = np.zeros(self.cores, dtype=np.float32)
            cores_transfers = np.zeros(self.cores)

            for corr in iteration_data:
                corr0 = int(corr[0])
                corr1 = int(corr[1])
                corr2 = int(corr[2])
                if corr0 == corr1:
                    p = r2m(corr0)
                    field[p[0]][p[1]] += corr2
                    field_velo[p[0]][p[1]] += corr[3]
                else:
                    p1 = r2m(corr0)
                    p2 = r2m(corr1)
                    field[p1[0]][p1[1]] -= corr2
                    field_velo[p1[0]][p1[1]] -= corr[3]
                    field[p2[0]][p2[1]] += corr2
                    field_velo[p2[0]][p2[1]] += corr[3]
                    core1 = schedule[p1[0]][p1[1]]
                    core2 = schedule[p2[0]][p2[1]]
                    if core1 != core2:
                        cores_transfers[core1] += corr2
                        cores_transfers[core2] += corr2
            for x in range(self.x_size):
                for y in range(self.y_size):
                    cores_model[schedule[x][y] - 1] += field[x][y]
                    cores_velo[schedule[x][y] - 1] += field_velo[x][y]

            cores_transfers = cores_transfers * agent_transfer_time
            cores_model = v_agent_transfer_time(cores_model)
            velo_coef = 90000000.0
            iter_velocities = cores_model + np.abs(cores_velo / velo_coef)
            total_model_time = iter_velocities + cores_transfers

            # cores_modeling_time[:, i] = cores_model
            cores_transfer_time[:, i] = cores_transfers
            cores_total_time[:, i] = total_model_time
            cores_velocities[:, i] = iter_velocities

            # model_times[i] = cores_model.max()
            transfer_times[i] = cores_transfers.max()
            total_times[i] = total_model_time.max()
            velocities[i] = iter_velocities.max()
            # draw output
            if i % slides == 0:
                plt.imshow(field, vmax=max_value)
                plt.contour(schedule, cmap='Set1', alpha=0.4, linestyles="--")
                plt.tight_layout()
                plt.show()
                plt.pause(0.000001)
                plt.clf()
            stdout.write("\riteration=%d" % i)
            stdout.flush()
        plt.close()
        plt.ioff()
        print("\nmodel sum={}".format(model_times.sum()))
        print("transfer sum={}".format(transfer_times.sum()))
        print("total time={}".format(total_times.sum()))
        print("total velocities={}".format(velocities.sum()))
        x_range = np.arange(0, iters)
        plt.figure()
        # plt.plot(x_range, model_times, label='model')
        plt.plot(x_range, transfer_times, label='transfer')
        plt.plot(x_range, velocities, label='model')
        plt.plot(x_range, total_times, label='total')
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Iteration modelling time, s")
        plt.ylim(0, 0.625)
        plt.tight_layout()
        # plt.show()

        ax_root = int(np.ceil(np.sqrt(self.cores)))
        f, axes = plt.subplots(ax_root, ax_root, figsize=(14, 10), sharex='col', sharey='row')
        for c in range(self.cores):
            ax_x = int(c / ax_root)
            ax_y = int(c % ax_root)
            # axes[ax_x][ax_y].plot(x_range, cores_modeling_time[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, cores_transfer_time[c, :], label='transfer')
            axes[ax_x][ax_y].plot(x_range, cores_velocities[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, cores_total_time[c, :], label='total')
            # axes[ax_x][ax_y].legend()
            axes[ax_x][ax_y].set_title("Core {}".format(c))

        plt.tight_layout()
        plt.show()


    def calc_distance(self, point, centers):
        dists = np.zeros(len(centers))
        for c in range(len(centers)):
            dists[c] = distance.euclidean(point, centers[c])
        return dists.min()


    def solution_to_schedule(self, solution):
        x_size = self.x_size
        y_size = self.y_size
        cores = self.cores
        schedule = np.zeros((x_size, y_size), dtype=np.int32)
        for x in range(x_size):
            for y in range(y_size):
                distances = np.zeros(cores)
                for c in range(cores):
                    distances[c] = self.calc_distance((x, y), solution[c])
                core_idx = np.argmin(distances)
                schedule[x][y] = core_idx
        return schedule

    def evaluate_solution(self, solution):
        schedule = self.solution_to_schedule(solution)
        return self.simulation(schedule)

    colors = ['r', 'g', 'b', 'yellow', 'pink', 'white', 'brown', 'aqua']



    def plotsolution(self, solution):
        from matplotlib import cm
        schedule = self.solution_to_schedule(solution)
        plt.figure()
        plt.imshow(schedule, cmap="Pastel1")
        i = 0
        cmap = cm.get_cmap('Set1')
        for agent in solution:
            for (x, y) in agent:
                plt.plot(y-0.5, x-0.5, marker="x", markersize=10, c=cmap(i))
            i += 1
        plt.xlim(-0.5, 29.5)
        plt.ylim(-0.5,29.5)
        plt.show()
        plt.close()
