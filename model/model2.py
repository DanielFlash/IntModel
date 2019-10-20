import numpy as np
from numpy import random as rnd
from deap import creator
from sys import stdout
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import random


agent_model_time = 0.001
agent_transfer_time = 0.005


def agent_model_time_func(agents):      # Add some noise
    stable_time = agents * agent_model_time
    return stable_time + rnd.normal(loc=stable_time * 0.1, scale=stable_time * 0.01)


v_agent_transfer_time = np.vectorize(agent_model_time_func)


def r2m(idx):
    return idx // 30, idx % 30


class AgentMobilityModel_2:
    def __init__(self, x_size, y_size, transportations, schedules, partitions, kernels):
        self.x_size = x_size
        self.y_size = y_size
        self.transportations = transportations
        self.iterations = len(transportations)
        self.schedules = schedules
        self.partitions = partitions
        self.kernels = kernels
        self.ev_start = 0
        self.ev_end = self.iterations
        self.field = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        self.field_velo = np.zeros((self.x_size, self.y_size), dtype=np.float32)
        # self.init_simulation()


    def init_simulation(self):
        self.schedule_total_time = np.zeros(self.partitions)
        for i in range(self.ev_start):
            if i in self.schedules.keys():  # 0, 144, 288, ..., 1296
                schedule = self.schedules[i]  # Change schedule

            iteration_data = self.transportations[i]

            partitions_model = np.zeros(self.partitions)  # How many agents on partitions
            partitions_velo = np.zeros(self.partitions, dtype=np.float32)  # Sum of velocity on partitions
            partitions_transfers = np.zeros(self.partitions)  # How many agents come to/from this partition

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
                    partition1 = schedule[p1[0]][p1[1]]  # Find the partitions
                    partition2 = schedule[p2[0]][p2[1]]
                    if partition1 != partition2:
                        partitions_transfers[partition1] += corr2
                        partitions_transfers[partition2] += corr2
            for x in range(self.x_size):
                for y in range(self.y_size):
                    partitions_model[schedule[x][y] - 1] += self.field[x][y]
                    partitions_velo[schedule[x][y] - 1] += self.field_velo[x][y]

            partitions_transfers = partitions_transfers * agent_transfer_time  # Transfer time
            partitions_model = v_agent_transfer_time(partitions_model)  # Model time
            velo_coef = 90000000.0
            iter_velocities = partitions_model + np.abs(partitions_velo / velo_coef)  # Model time with velocity
            total_model_time = iter_velocities + partitions_transfers  # Total time

            self.schedule_total_time += total_model_time


    def simulation(self, kschedule, flag=False, schedule_number=-1):
        # total_times = np.zeros(self.ev_end)
        schedule_total_time = np.zeros(self.partitions)
        kschedule = kschedule
        field = np.copy(self.field)
        field_velo = np.copy(self.field_velo)

        if flag:
            self.ev_start = 144 * schedule_number
            self.ev_end = 144 * (schedule_number + 1)

        total_times = np.zeros(self.ev_end)

        for i in range(self.ev_start, self.ev_end):
            if i in self.schedules.keys():  # 0, 144, 288, ..., 1296
                schedule = self.schedules[i]  # Change schedule

            iteration_data = self.transportations[i]
            partitions_model = np.zeros(self.partitions)
            partitions_velocities = np.zeros(self.partitions)
            partitions_transfers = np.zeros(self.partitions)

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
                    partition1 = schedule[p1[0]][p1[1]]
                    partition2 = schedule[p2[0]][p2[1]]
                    if partition1 != partition2:
                        partitions_transfers[partition1 - 1] += corr2
                        partitions_transfers[partition2 - 1] += corr2
            for x in range(self.x_size):
                for y in range(self.y_size):
                    partitions_model[schedule[x][y] - 1] += field[x][y]
                    partitions_velocities[schedule[x][y] - 1] += field_velo[x][y]

            partitions_transfers = partitions_transfers * agent_transfer_time
            partitions_model = v_agent_transfer_time(partitions_model)
            velo_coef = 90000000.0
            iter_velocities = partitions_model + np.abs(partitions_velocities / velo_coef)
            iter_velocities = iter_velocities * kschedule
            total_model_time = iter_velocities + partitions_transfers

            total_times[i] = total_model_time.max()

            schedule_total_time += total_model_time

        # result = total_times[self.ev_start:self.ev_end].sum()
        result = max(schedule_total_time) - min(schedule_total_time)

        return result,


    def interactive_simulation(self, kschedules, slides=50):
        max_value = 39  # For the colormap in plot
        # initialization
        iters = len(self.transportations)
        model_times = np.zeros(iters)
        transfer_times = np.zeros(iters)
        total_times = np.zeros(iters)
        total_times_krn = np.zeros(iters)
        velocities = np.zeros(iters)
        velocities_krn = np.zeros(iters)
        field = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        field_velo = np.zeros((self.x_size, self.y_size), dtype=np.float32)
        schedule = self.schedules[0]
        kschedule = kschedules[0]
        plt.ion()
        plt.imshow(field, vmax=max_value)
        plt.contour(schedule, alpha=0.5, cmap='Set1')
        plt.tight_layout()
        plt.show()
        plt.pause(0.0000001)

        partitions_modeling_time = np.zeros((self.partitions, 1440))
        partitions_transfer_time = np.zeros((self.partitions, 1440))
        partitions_total_time = np.zeros((self.partitions, 1440))
        partitions_total_time_krn = np.zeros((self.partitions, 1440))
        partitions_velocities = np.zeros((self.partitions, 1440))
        partitions_velocities_krn = np.zeros((self.partitions, 1440))

        schedule_number = 0

        for i in range(0, 1440):
            if i in self.schedules.keys():  # 0, 144, 288, ..., 1296
                schedule = self.schedules[i]  # Change schedule
                kschedule = kschedules[schedule_number]
                # kschedule = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
                schedule_number += 1
            # iteration
            iteration_data = self.transportations[i]
            partitions_model = np.zeros(self.partitions)  # How many agents on cores
            partitions_velo = np.zeros(self.partitions, dtype=np.float32)  # Sum of velocity on cores
            partitions_transfers = np.zeros(self.partitions)  # How many agents come to/from this core

            for corr in iteration_data:  # For each transp. in all next transp.
                corr0 = int(corr[0])
                corr1 = int(corr[1])
                corr2 = int(corr[2])
                if corr0 == corr1:  # If the same cell
                    p = r2m(corr0)  # Get coordinates "x" and "y"
                    field[p[0]][p[1]] += corr2  # Add this number of agents
                    field_velo[p[0]][p[1]] += corr[3]  # Add this velocity
                else:
                    p1 = r2m(corr0)
                    p2 = r2m(corr1)
                    field[p1[0]][p1[1]] -= corr2
                    field_velo[p1[0]][p1[1]] -= corr[3]
                    field[p2[0]][p2[1]] += corr2
                    field_velo[p2[0]][p2[1]] += corr[3]
                    partition1 = schedule[p1[0]][p1[1]]  # Find the cores
                    partition2 = schedule[p2[0]][p2[1]]
                    if partition1 != partition2:
                        partitions_transfers[partition1] += corr2
                        partitions_transfers[partition2] += corr2
            for x in range(self.x_size):
                for y in range(self.y_size):
                    partitions_model[schedule[x][y] - 1] += field[x][y]
                    partitions_velo[schedule[x][y] - 1] += field_velo[x][y]

            partitions_transfers = partitions_transfers * agent_transfer_time  # Transfer time
            partitions_model = v_agent_transfer_time(partitions_model)  # Model time
            velo_coef = 90000000.0
            iter_velocities = partitions_model + np.abs(partitions_velo / velo_coef)  # Model time with velocity
            iter_velocities_krn = iter_velocities * kschedule
            total_model_time = iter_velocities + partitions_transfers  # Total time
            total_model_time_krn = iter_velocities_krn + partitions_transfers  # Total time

            partitions_modeling_time[:, i] = partitions_model
            partitions_transfer_time[:, i] = partitions_transfers
            partitions_total_time[:, i] = total_model_time
            partitions_total_time_krn[:, i] = total_model_time_krn
            partitions_velocities[:, i] = iter_velocities
            partitions_velocities_krn[:, i] = iter_velocities_krn

            model_times[i] = partitions_model.max()
            transfer_times[i] = partitions_transfers.max()
            total_times[i] = total_model_time.max()
            total_times_krn[i] = total_model_time_krn.max()
            velocities[i] = iter_velocities.max()
            velocities_krn[i] = iter_velocities_krn.max()

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
        print("total time on kernels={}".format(total_times_krn.sum()))
        print("total velocities={}".format(velocities.sum()))
        print("total velocities on kernels={}".format(velocities_krn.sum()))
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

        plt.figure()
        # plt.plot(x_range, model_times, label='model')
        plt.plot(x_range, transfer_times, label='transfer')
        plt.plot(x_range, velocities_krn, label='model on kernels')
        plt.plot(x_range, total_times_krn, label='total on kernels')
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Iteration modelling time, s")
        plt.ylim(0, 0.625)
        plt.tight_layout()
        # plt.show()

        ax_root = int(np.ceil(np.sqrt(self.partitions)))
        f, axes = plt.subplots(ax_root, ax_root, figsize=(14, 10), sharex='col', sharey='row')
        for c in range(self.partitions):
            ax_x = int(c / ax_root)
            ax_y = int(c % ax_root)
            # axes[ax_x][ax_y].plot(x_range, partitions_modeling_time[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, partitions_transfer_time[c, :], label='transfer')
            axes[ax_x][ax_y].plot(x_range, partitions_velocities[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, partitions_total_time[c, :], label='total')
            # axes[ax_x][ax_y].legend()
            axes[ax_x][ax_y].set_title("Core {}".format(c))

        plt.tight_layout()
        # plt.show()

        ax_root = int(np.ceil(np.sqrt(self.partitions)))
        f, axes = plt.subplots(ax_root, ax_root, figsize=(14, 10), sharex='col', sharey='row')
        for c in range(self.partitions):
            ax_x = int(c / ax_root)
            ax_y = int(c % ax_root)
            # axes[ax_x][ax_y].plot(x_range, partitions_modeling_time[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, partitions_transfer_time[c, :], label='transfer')
            axes[ax_x][ax_y].plot(x_range, partitions_velocities_krn[c, :], label='model on kernels')
            axes[ax_x][ax_y].plot(x_range, partitions_total_time_krn[c, :], label='total on kernels')
            # axes[ax_x][ax_y].legend()
            axes[ax_x][ax_y].set_title("Core {}".format(c))

        plt.tight_layout()
        plt.show()


    def interactive_simulation2_velo_on_kernel(self, schedules, slides=50):  # slides - how often update the plot
        """ The same simulation, but with some extra code for mapping field zones with kernels.
        'partition' is the field zone (as it was before); 'Kernel' is the physical kernel in computer.
        Algorithm: we sum the partitions load and map it on kernels for each schedule """
        max_value = 39  # For the colormap in plot
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

        partitions_modeling_time = np.zeros((self.partitions, 1440))
        partitions_transfer_time = np.zeros((self.partitions, 1440))
        partitions_total_time = np.zeros((self.partitions, 1440))
        partitions_velocities = np.zeros((self.partitions, 1440))

        new_schedule = False        # Some flags to track schedule changes
        initial_schedule = True
        schedule_number = 0

        schedule_transfer_time = np.zeros(self.partitions)  # Sum of time during the one schedule
        schedule_model_time = np.zeros(self.partitions)
        schedule_velocities = np.zeros(self.partitions)
        schedule_total_time = np.zeros(self.partitions)

        schedule_partition_transfers = np.zeros((self.partitions, self.partitions))  # Need for finding transfers between partitions on one kernel
        kernels_load = np.zeros((self.partitions, self.kernels))  # Kernel load
        final_kernels_load = np.zeros((len(schedules), self.kernels))  # Sum of kernels loads for every schedule

        for i in range(0, 1440):
            if i in schedules.keys():       # 0, 144, 288, ..., 1296
                schedule = schedules[i]     # Change schedule
                new_schedule = True
            # iteration
            iteration_data = self.transportations[i]
            partitions_model = np.zeros(self.partitions)  # How many agents on partitions
            partitions_velo = np.zeros(self.partitions, dtype=np.float32)  # Sum of velocity on partitions
            partitions_transfers = np.zeros(self.partitions)  # How many agents come to/from this partition

            for corr in iteration_data:  # For each transp. in all next transp.
                corr0 = int(corr[0])
                corr1 = int(corr[1])
                corr2 = int(corr[2])
                if corr0 == corr1:  # If the same cell
                    p = r2m(corr0)  # Get coordinates "x" and "y"
                    field[p[0]][p[1]] += corr2  # Add this number of agents
                    field_velo[p[0]][p[1]] += corr[3]   # Add this velocity
                else:
                    p1 = r2m(corr0)
                    p2 = r2m(corr1)
                    field[p1[0]][p1[1]] -= corr2
                    field_velo[p1[0]][p1[1]] -= corr[3]
                    field[p2[0]][p2[1]] += corr2
                    field_velo[p2[0]][p2[1]] += corr[3]
                    partition1 = schedule[p1[0]][p1[1]]  # Find the partitions
                    partition2 = schedule[p2[0]][p2[1]]
                    if partition1 != partition2:
                        partitions_transfers[partition1] += corr2
                        partitions_transfers[partition2] += corr2
                        schedule_partition_transfers[partition1, partition2] += corr2  # partition matrix (for kernels)
                        schedule_partition_transfers[partition2, partition1] += corr2
            for x in range(self.x_size):
                for y in range(self.y_size):
                    partitions_model[schedule[x][y] - 1] += field[x][y]  # Update the number of agents
                    partitions_velo[schedule[x][y] - 1] += field_velo[x][y]

            partitions_transfers = partitions_transfers * agent_transfer_time     # Transfer time
            partitions_model = v_agent_transfer_time(partitions_model)            # Model time
            velo_coef = 90000000.0
            iter_velocities = partitions_model + np.abs(partitions_velo / velo_coef)  # Model time with velocity
            total_model_time = iter_velocities + partitions_transfers    # Total time

            partitions_modeling_time[:, i] = partitions_model  # Modeling time for all partitions on iteration
            partitions_transfer_time[:, i] = partitions_transfers
            partitions_total_time[:, i] = total_model_time
            partitions_velocities[:, i] = iter_velocities

            model_times[i] = partitions_model.max()  # Max modeling time among all partitions on iteration
            transfer_times[i] = partitions_transfers.max()
            total_times[i] = total_model_time.max()
            velocities[i] = iter_velocities.max()

            if new_schedule:  # If schedule was changed
                if initial_schedule:  # If it is the first change, we do not calculate load
                    initial_schedule = False
                else:  # Calculate load
                    kernels_load = self.greedy_algorithm(kernels_load, schedule_partition_transfers,
                                          schedule_model_time, schedule_transfer_time,
                                          schedule_total_time, schedule_velocities)

                    for kernel in range(self.kernels):
                        print("kernels load={}".format((kernels_load[:, kernel]).sum()))
                        final_kernels_load[schedule_number, kernel] = (kernels_load[:, kernel]).sum()
                    schedule_number += 1

                schedule_transfer_time = np.zeros(self.partitions)  # Reset to zero our parameters
                schedule_model_time = np.zeros(self.partitions)
                schedule_velocities = np.zeros(self.partitions)
                schedule_total_time = np.zeros(self.partitions)
                schedule_partition_transfers = np.zeros((self.partitions, self.partitions))
                kernels_load = np.zeros((self.partitions, self.kernels))
                new_schedule = False

            schedule_transfer_time += partitions_transfers  # Add loads
            schedule_model_time += partitions_model
            schedule_velocities += iter_velocities
            schedule_total_time += total_model_time

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
        print("\nmodel sum={}".format(model_times.sum()))  # Sum of max modeling time for all iterations
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

        ax_root = int(np.ceil(np.sqrt(self.partitions)))
        f, axes = plt.subplots(ax_root, ax_root, figsize=(14, 10), sharex='col', sharey='row')
        for c in range(self.partitions):
            ax_x = int(c / ax_root)
            ax_y = int(c % ax_root)
            # axes[ax_x][ax_y].plot(x_range, partitions_modeling_time[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, partitions_transfer_time[c, :], label='transfer')
            axes[ax_x][ax_y].plot(x_range, partitions_velocities[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, partitions_total_time[c, :], label='total')
            # axes[ax_x][ax_y].legend()
            axes[ax_x][ax_y].set_title("partition {}".format(c))

        plt.tight_layout()
        plt.show()

        # Output the kernels load:
        print("final kernels load=\n{}".format(final_kernels_load))

        plt.figure()
        sched_range = np.arange(0, len(schedules))
        for kernel in range(self.kernels):
            plt.plot(sched_range, final_kernels_load[:, kernel], label='kernel {}'.format(kernel))
        plt.legend()
        plt.xlabel("Schedules")
        plt.ylabel("Iteration modelling time, s")
        # plt.ylim(0, 50)
        plt.tight_layout()
        plt.show()


    def greedy_algorithm(self, kernels_load, schedule_partition_transfers,
                         schedule_model_time, schedule_transfer_time,
                         schedule_total_time, schedule_velocities):

        print("\nmodel sum on schedule=\n{}".format(schedule_model_time))
        print("transfer sum on schedule=\n{}".format(schedule_transfer_time))
        print("total time on schedule=\n{}".format(schedule_total_time))
        print("total velocities on schedule=\n{}".format(schedule_velocities))

        for partition in range(self.partitions):  # Take each partition
            min_load = math.inf
            min_kernel = 0
            for kernel in range(self.kernels):  # Take each kernel
                load = (kernels_load[:, kernel]).sum() + schedule_total_time[partition]  # Overall load
                for set_partition in range(
                        self.partitions):  # Look for partitions transfers between chosen partition and other on chosen kernel
                    if kernels_load[set_partition, kernel] != 0:
                        if schedule_partition_transfers[partition, set_partition] != 0:
                            load -= 2 * schedule_partition_transfers[partition, set_partition] * \
                                    agent_transfer_time  # Subtract transfer load
                if min_load > load:
                    min_load = load  # Find min load
                    min_kernel = kernel  # Find kernel with min load

            kernels_load[partition, min_kernel] += schedule_total_time[partition]  # Add partition on kernel

            for set_partition in range(self.partitions):  # Subtract partitions transfers load
                if kernels_load[set_partition, min_kernel] != 0:
                    if schedule_partition_transfers[partition, set_partition] != 0:
                        kernels_load[partition, min_kernel] -= \
                            schedule_partition_transfers[partition, set_partition] * agent_transfer_time
                        kernels_load[set_partition, min_kernel] -= \
                            schedule_partition_transfers[partition, set_partition] * agent_transfer_time

        print("\nkernels load of partitions=\n{}\n".format(kernels_load))

        return kernels_load


    def interactive_simulation2_velo_on_kernel2(self, slides=50):
        max_value = 39
        # initialization
        iters = len(self.transportations)
        model_times = np.zeros(iters)
        transfer_times = np.zeros(iters)
        total_times = np.zeros(iters)
        velocities = np.zeros(iters)
        field = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        field_velo = np.zeros((self.x_size, self.y_size), dtype=np.float32)
        schedule = self.schedules[0]
        plt.ion()
        plt.imshow(field, vmax=max_value)
        plt.contour(schedule, alpha=0.5, cmap='Set1')
        plt.tight_layout()
        plt.show()
        plt.pause(0.0000001)

        partitions_modeling_time = np.zeros((self.partitions, 1440))
        partitions_transfer_time = np.zeros((self.partitions, 1440))
        partitions_total_time = np.zeros((self.partitions, 1440))
        partitions_velocities = np.zeros((self.partitions, 1440))

        new_schedule = False  # Some flags to track schedule changes
        initial_schedule = True
        schedule_number = 0
        schedule_change_iter = np.zeros(len(self.schedules))

        schedule_transfer_time = np.zeros(self.partitions)  # Sum of time during the one schedule
        schedule_model_time = np.zeros(self.partitions)
        schedule_velocities = np.zeros(self.partitions)
        schedule_total_time = np.zeros(self.partitions)

        kernels_load = np.zeros((self.partitions, self.kernels))  # Kernel load
        all_coefs_list = np.zeros((len(self.schedules), self.partitions))

        for i in range(0, 1440):
            if i in self.schedules.keys():       # 0, 144, 288, ..., 1296
                schedule = self.schedules[i]     # Change schedule
                new_schedule = True
            # iteration
            iteration_data = self.transportations[i]
            partitions_model = np.zeros(self.partitions)  # How many agents on partitions
            partitions_velo = np.zeros(self.partitions, dtype=np.float32)  # Sum of velocity on partitions
            partitions_transfers = np.zeros(self.partitions)  # How many agents come to/from this partition

            for corr in iteration_data:  # For each transp. in all next transp.
                corr0 = int(corr[0])
                corr1 = int(corr[1])
                corr2 = int(corr[2])
                if corr0 == corr1:  # If the same cell
                    p = r2m(corr0)  # Get coordinates "x" and "y"
                    field[p[0]][p[1]] += corr2  # Add this number of agents
                    field_velo[p[0]][p[1]] += corr[3]   # Add this velocity
                else:
                    p1 = r2m(corr0)
                    p2 = r2m(corr1)
                    field[p1[0]][p1[1]] -= corr2
                    field_velo[p1[0]][p1[1]] -= corr[3]
                    field[p2[0]][p2[1]] += corr2
                    field_velo[p2[0]][p2[1]] += corr[3]
                    partition1 = schedule[p1[0]][p1[1]]  # Find the partitions
                    partition2 = schedule[p2[0]][p2[1]]
                    if partition1 != partition2:
                        partitions_transfers[partition1] += corr2
                        partitions_transfers[partition2] += corr2
            for x in range(self.x_size):
                for y in range(self.y_size):
                    partitions_model[schedule[x][y] - 1] += field[x][y]
                    partitions_velo[schedule[x][y] - 1] += field_velo[x][y]

            partitions_transfers = partitions_transfers * agent_transfer_time     # Transfer time
            partitions_model = v_agent_transfer_time(partitions_model)            # Model time
            velo_coef = 90000000.0
            iter_velocities = partitions_model + np.abs(partitions_velo / velo_coef)  # Model time with velocity
            total_model_time = iter_velocities + partitions_transfers    # Total time

            partitions_modeling_time[:, i] = partitions_model
            partitions_transfer_time[:, i] = partitions_transfers
            partitions_total_time[:, i] = total_model_time
            partitions_velocities[:, i] = iter_velocities

            model_times[i] = partitions_model.max()
            transfer_times[i] = partitions_transfers.max()
            total_times[i] = total_model_time.max()
            velocities[i] = iter_velocities.max()

            if new_schedule:  # If schedule was changed
                if initial_schedule:  # If it is the first change, we do not calculate load
                    initial_schedule = False
                else:  # Calculate load
                    schedule_change_iter[schedule_number] = i
                    coefs_list = self.gen_algorithm(kernels_load, schedule_model_time,
                                                    schedule_transfer_time, schedule_total_time,
                                                    schedule_velocities, schedule_number)

                    for partition in range(self.partitions):
                        all_coefs_list[schedule_number, partition] = coefs_list[partition]
                        print("partition coefficients={}".format(coefs_list[partition]))
                    schedule_number += 1

                schedule_transfer_time = np.zeros(self.partitions)  # Reset to zero our parameters
                schedule_model_time = np.zeros(self.partitions)
                schedule_velocities = np.zeros(self.partitions)
                schedule_total_time = np.zeros(self.partitions)
                kernels_load = np.zeros((self.partitions, self.kernels))
                new_schedule = False

            schedule_transfer_time += partitions_transfers  # Add loads
            schedule_model_time += partitions_model
            schedule_velocities += iter_velocities
            schedule_total_time += total_model_time

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

        ax_root = int(np.ceil(np.sqrt(self.partitions)))
        f, axes = plt.subplots(ax_root, ax_root, figsize=(14, 10), sharex='col', sharey='row')
        for c in range(self.partitions):
            ax_x = int(c / ax_root)
            ax_y = int(c % ax_root)
            # axes[ax_x][ax_y].plot(x_range, partitions_modeling_time[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, partitions_transfer_time[c, :], label='transfer')
            axes[ax_x][ax_y].plot(x_range, partitions_velocities[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, partitions_total_time[c, :], label='total')
            # axes[ax_x][ax_y].legend()
            axes[ax_x][ax_y].set_title("partition {}".format(c))

        plt.tight_layout()
        # plt.show()

        print("\nmodel sum={}".format(self.multiply_coefs(partitions_modeling_time, all_coefs_list, schedule_change_iter).sum()))
        print("transfer sum={}".format(self.multiply_coefs(partitions_transfer_time, all_coefs_list, schedule_change_iter).sum()))
        print("total time={}".format(self.multiply_coefs(partitions_total_time, all_coefs_list, schedule_change_iter).sum()))
        print("total velocities={}".format(self.multiply_coefs(partitions_velocities, all_coefs_list, schedule_change_iter).sum()))

        plt.figure()
        # plt.plot(x_range, self.multiply_coefs(partitions_modeling_time, all_coefs_list, schedule_change_iter), label='model')
        plt.plot(x_range, self.multiply_coefs(partitions_transfer_time, all_coefs_list, schedule_change_iter), label='transfer')
        plt.plot(x_range, self.multiply_coefs(partitions_velocities, all_coefs_list, schedule_change_iter), label='model')
        plt.plot(x_range, self.multiply_coefs(partitions_total_time, all_coefs_list, schedule_change_iter), label='total')
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Iteration modelling time, s")
        # plt.ylim(0, 50)
        plt.tight_layout()
        # plt.show()

        ax_root = int(np.ceil(np.sqrt(self.partitions)))
        f, axes = plt.subplots(ax_root, ax_root, figsize=(14, 10), sharex='col', sharey='row')
        for c in range(self.partitions):
            ax_x = int(c / ax_root)
            ax_y = int(c % ax_root)
            # axes[ax_x][ax_y].plot(x_range, self.multiply_coefs(partitions_modeling_time, all_coefs_list, schedule_change_iter, False)[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, self.multiply_coefs(partitions_transfer_time, all_coefs_list, schedule_change_iter, False)[c, :], label='transfer')
            axes[ax_x][ax_y].plot(x_range, self.multiply_coefs(partitions_velocities, all_coefs_list, schedule_change_iter, False)[c, :], label='model')
            axes[ax_x][ax_y].plot(x_range, self.multiply_coefs(partitions_total_time, all_coefs_list, schedule_change_iter, False)[c, :], label='total')
            # axes[ax_x][ax_y].legend()
            axes[ax_x][ax_y].set_title("partition {}".format(c))

        plt.tight_layout()
        plt.show()


    def multiply_coefs(self, data, all_coefs_list, schedule_change_iter, flag = True):
        schedule_number = 0
        for i in range(0, 1440):
            # partitions_transfer_time[:, i] = partitions_transfers
            for partition in range(self.partitions):
                j = schedule_change_iter[schedule_number]

                data[partition, i] = data[partition, i] / all_coefs_list[schedule_number, partition]

                if i == j - 1:
                    schedule_number += 1
                    print(i, j)

        if flag:
            max_data = np.zeros(len(self.transportations))
            for i in range(0, 1440):
                max_data[i] = data[:, i].max()
            return max_data

        return data


    def gen_algorithm(self, kernels_load, schedule_model_time,
                      schedule_transfer_time, schedule_total_time,
                      schedule_velocities, schedule_number):

        print("\nmodel sum on schedule=\n{}".format(schedule_model_time))
        print("transfer sum on schedule=\n{}".format(schedule_transfer_time))
        print("total time on schedule=\n{}".format(schedule_total_time))
        print("total velocities on schedule=\n{}".format(schedule_velocities))

        population_size = 10
        population = list()
        for elem in range(population_size):
            population.append(kernels_load.copy())

        for individual in population:
            check_list = np.zeros(self.kernels)
            for partition in range(self.partitions):
                kernel = random.randrange(0, self.kernels)
                while check_list[kernel] != 0:
                    kernel = random.randrange(0, self.kernels)
                # individual[partition, kernel] = schedule_total_time[partition]
                individual[partition, kernel] = 1
                check_list[kernel] = 1
            for kernel in range(self.kernels):
                if check_list[kernel] == 0:
                    partition = random.randrange(0, self.partitions)
                    # individual[partition, kernel] = schedule_total_time[partition]
                    individual[partition, kernel] = 1

        tmp_iter = 0
        for gen_iter in range(10):
            new_population = list()
            for individual in population:
                part = random.randrange(0, self.partitions)
                krn_list = list()
                krn_list2 = list()
                for krn in range(self.kernels):
                    if individual[part, krn] != 0:
                        krn_list.append(krn)
                while True:
                    part2 = random.randrange(0, self.partitions)
                    while part == part2:
                        part2 = random.randrange(0, self.partitions)
                    krn_list2.clear()
                    for krn in range(self.kernels):
                        if individual[part2, krn] != 0:
                            krn_list2.append(krn)
                    if (len(krn_list) > 1 and len(krn_list2) > 1) or \
                            ((len(krn_list) != len(krn_list2)) and len(krn_list) > 0 and len(krn_list2) > 0):
                        break

                new_individual = individual.copy()
                if len(krn_list) > 1:
                    krn = random.choice(krn_list)
                    new_individual[part2, krn] = new_individual[part, krn]
                    new_individual[part, krn] = 0
                elif len(krn_list2) > 1:
                    krn = random.choice(krn_list2)
                    new_individual[part, krn] = new_individual[part2, krn]
                    new_individual[part2, krn] = 0
                else:
                    print(len(krn_list), len(krn_list2))
                    input()

                new_population.append([individual, 0])
                new_population.append([new_individual, 0])

            load_idces = np.zeros(len(new_population))
            for idx, (individual, _) in enumerate(new_population):
                # total_load = np.zeros(self.partitions)
                # for partition in range(self.partitions):
                    # load, _ = self.load_func(individual, partition)
                    # total_load[partition] = load

                # load_idces[idx] = max(total_load) - min(total_load)
                coefs_list = self.solution_to_schedule(individual)
                result = self.simulation(coefs_list, True, schedule_number)
                load_idces[idx] = result[0]

                print(f"OUT {tmp_iter} of 200")
                tmp_iter += 1

            for elem in range(len(new_population)):
                new_population[elem][1] = load_idces[elem]

            new_population.sort(key=self.takeSecond)
            population.clear()
            for elem in range(population_size):
                population.append(new_population[elem][0])

        print("\nkernels load of partitions=\n{}\n".format(population[0]))

        coefs_list = np.zeros(self.partitions)
        for partition in range(self.partitions):
            _, coef = self.load_func(population[0], partition)
            coefs_list[partition] = coef

        return coefs_list


    def takeSecond(self, elem):
        return elem[1]


    def load_func(self, individual, partition):
        load_list = [individual[partition, kern]
                     for kern in range(self.kernels)
                     if individual[partition, kern] != 0]

        if len(load_list) == 0:
            print("PARTITION WITHOUT ANY KERNEL!!!")
            # input(">>> Press any key >>>")
            # raise Exception("Partition doesn't have any kernel")
            load_list.append(math.inf)

        return load_list[0] / pow(len(load_list), 0.5), pow(len(load_list), -0.5)


    def solution_to_schedule(self, solution):
        coefs_list = np.zeros(self.partitions)
        for partition in range(self.partitions):
            _, coef = self.load_func(solution, partition)
            coefs_list[partition] = coef
        return coefs_list


    def evaluate_solution(self, solution):
        kschedule = self.solution_to_schedule(solution)
        return self.simulation(kschedule)


    colors = ['r', 'g', 'b', 'yellow', 'pink', 'white', 'brown', 'aqua']


    def plotsolution(self, solution):
        from matplotlib import cm
        schedule = self.solution_to_schedule(solution)
        plt.figure()
        plt.imshow(schedule, cmap="Pastel1")
        cmap = cm.get_cmap('Set1')
        x_range = np.arange(0, self.partitions)
        i = 0
        for coefs in solution:
            plt.plot(x_range, coefs, marker="x", markersize=10, c=cmap(i))
            i += 1
        # plt.xlim(-0.5, 29.5)
        # plt.ylim(-0.5,29.5)
        plt.show()
        plt.close()
