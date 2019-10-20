from deap import tools, base
from multiprocessing import Pool
from algorithms.ga_scheme import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
import random

from deap import creator

creator.create("TimeFit", base.Fitness, weights=(-1.0,))
creator.create("ScheduleMixtureIndividual", np.ndarray, fitness=creator.TimeFit)


class IntervalMixtureSchedGA_2:
    def individual(self):
        partitions = self.model.partitions
        kernels = self.model.kernels
        schedule_total_time = self.model.schedule_total_time
        solution = np.zeros((partitions, kernels))

        check_list = np.zeros(kernels)
        for part in range(partitions):
            krn = random.randrange(0, kernels)
            while check_list[krn] != 0:
                krn = random.randrange(0, kernels)
            # solution[part, krn] = schedule_total_time[part]
            solution[part, krn] = 1
            check_list[krn] = 1

            krn = random.randrange(0, kernels)
            while check_list[krn] != 0:
                krn = random.randrange(0, kernels)
            # solution[part, krn] = schedule_total_time[part]
            solution[part, krn] = 1
            check_list[krn] = 1

        for krn in range(kernels):
            if check_list[krn] == 0:
                part = random.randrange(0, partitions)
                # solution[part, krn] = schedule_total_time[part]
                solution[part, krn] = 1

        return solution

    def mutation(self, mutant):
        partitions = self.model.partitions
        kernels = self.model.kernels
        part = random.randrange(0, partitions)
        krn_list = list()
        krn_list2 = list()
        for krn in range(kernels):
            if mutant[part, krn] != 0:
                krn_list.append(krn)
        while True:
            part2 = random.randrange(0, partitions)
            while part2 == part:
                part2 = random.randrange(0, partitions)
            krn_list2.clear()
            for krn in range(kernels):
                if mutant[part2, krn] != 0:
                    krn_list2.append(krn)
            if (np.count_nonzero(mutant[part, :]) > 1 and
                np.count_nonzero(mutant[part2, :]) >= 1) \
                    or (np.count_nonzero(mutant[part, :]) >= 1 and
                        np.count_nonzero(mutant[part2, :]) > 1):
                break

        if len(krn_list) > 1:
            krn = random.choice(krn_list)
            mutant[part2, krn] = mutant[part, krn]
            mutant[part, krn] = 0
        else:
            krn = random.choice(krn_list2)
            mutant[part, krn] = mutant[part2, krn]
            mutant[part2, krn] = 0

        if np.count_nonzero(mutant) < kernels:
            print(f"Mutant \n{mutant}")
            print(np.count_nonzero(mutant))

        return mutant,

    def crossover(self, p1, p2):
        partitions = self.model.partitions
        kernels = self.model.kernels
        # c1 = p1.copy()
        # c2 = p2.copy()
        c1 = creator.ScheduleMixtureIndividual(np.zeros((partitions, kernels)))
        c2 = creator.ScheduleMixtureIndividual(np.zeros((partitions, kernels)))
        point = random.randrange(1, (kernels - 1))
        i = 0
        for krn in range(kernels):
            if i < point:
                for part in range(partitions):
                    c1[part, krn] = p2[part, krn]
                    c2[part, krn] = p1[part, krn]
            else:
                for part in range(partitions):
                    c1[part, krn] = p1[part, krn]
                    c2[part, krn] = p2[part, krn]
            i += 1

        children = list()
        children.append(c1)
        children.append(c2)
        for c in children:
            zero_krn_parts = [part for part in range(partitions) if np.count_nonzero(c[part, :]) == 0]
            several_krn_parts = [part for part in range(partitions) if np.count_nonzero(c[part, :]) > 1]

            while len(zero_krn_parts) > 0:
                for z in range(len(zero_krn_parts)):
                    if z < len(several_krn_parts):
                        krn_list = [krn for krn in range(kernels) if c[several_krn_parts[z], krn] != 0]
                        krn = random.choice(krn_list)
                        c[zero_krn_parts[z], krn] = c[several_krn_parts[z], krn]
                        c[several_krn_parts[z], krn] = 0

                zero_krn_parts = [part for part in range(partitions) if np.count_nonzero(c[part, :]) == 0]
                several_krn_parts = [part for part in range(partitions) if np.count_nonzero(c[part, :]) > 1]

        if np.count_nonzero(c1) < kernels:
            print(f"c1 \n{c1}")
            print(np.count_nonzero(c1))
        if np.count_nonzero(c2) < kernels:
            print(f"c2 \n{c2}")
            print(np.count_nonzero(c2))

        return c1, c2

    def __init__(self, model, outpath, ext_sol=None, ):
        self.pool = Pool(10)
        # base params
        self.pop_size = 32
        self.generations = 100
        self.mut_prob = 0.5
        self.cross_prob = 0.15
        self.model = model
        self.external_sol = ext_sol
        self.outpath = outpath

        toolbox = base.Toolbox()
        toolbox.register("map", self.pool.map)
        # toolbox.register("map", map)

        toolbox.register("individual", tools.initIterate, creator.ScheduleMixtureIndividual, self.individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, self.pop_size)
        toolbox.register("mate", self.crossover)
        toolbox.register("mutate", self.mutation)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.model.evaluate_solution)  # Solution to Schedule and simulate
        toolbox.register("savesched", self.write_solution)  # Save the schedule of the best solution
        toolbox.register("plotsolution", self.model.plotsolution)   # Solution to Schedule and plot it

        self.toolbox = toolbox

    def write_solution(self, best):  # Save the schedule of the best solution
        best_schedule = self.model.solution_to_schedule(best)  # Solution to Schedule
        out_schedule = open(self.outpath, 'w')
        for part in range(self.model.partitions):
            out_schedule.write("{}\t{}\n".format(part, best_schedule[part]))
        out_schedule.close()

    def __call__(self, start, end):
        self.model.ev_start = start
        self.model.ev_end = end
        self.model.init_simulation()
        pop = self.toolbox.population()
        # if self.external_sol is not None:
        #     pop[0] = self.external_sol

        hof = tools.HallOfFame(2, np.array_equal)
        stats = tools.Statistics(lambda ind: np.array(ind.fitness.values))
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        pop, logbook = eaMuPlusLambda(pop, self.toolbox, self.pop_size, self.pop_size, self.cross_prob, self.mut_prob,
                                      self.generations, stats=stats, halloffame=hof)
        return pop, logbook, hof


def main():
    from simulation_launch import read_transportations, read_schedule
    from model.model2 import AgentMobilityModel_2
    for iters in range(9, 10):
        print("Iter = {}".format(iters))
        # Select input scenario file
        transportations_file = "..\\resources\\spb_passengers_center_100k_1"
        transportations = read_transportations(transportations_file)
        # Select input schedule files
        schedules = dict()  # Dictionary of 10. Each value - schedule
        # Each schedule - matrix x on y; each cell - number of core
        for i in range(10):
            iter_sched = read_schedule(
                "..\\schedules\\multiple\\10_include\\spb_schedule_velo_{}_{}.sched".format(
                    i * 144, (i + 1) * 144
                )
            )
            schedules[i * 144] = iter_sched
        x_size = 30
        y_size = 30
        partitions = 9
        kernels = 50
        ammodel = AgentMobilityModel_2(x_size, y_size, transportations, schedules, partitions, kernels)
        outpath = "..\\kernel_schedules\\kernel_50_schedule_output_{}.ksched".format(iters)
        scheduler = IntervalMixtureSchedGA_2(ammodel, outpath, ext_sol=None)
        result = scheduler(iters * 144, (iters + 1) * 144)  # Call "__call__"
        best_solution = result[2].items[0]
        scheduler.write_solution(best_solution)


if __name__ == "__main__":
    main()
