from deap import tools, base
from multiprocessing import Pool
from algorithms.ga_scheme import eaMuPlusLambda
from numpy import random as rnd
import numpy as np

from deap import creator

creator.create("TimeFit", base.Fitness, weights=(-1.0,))
creator.create("ScheduleMixtureIndividual", np.ndarray, fitness=creator.TimeFit)


class IntervalMixtureSchedGA:
    def individual(self):
        x_size = self.model.x_size
        y_size = self.model.y_size
        cores = self.model.cores
        max_centers = self.max_centers
        solution = rnd.random((cores, max_centers, 2)) * x_size  # TODO if x==y
        return solution

    def mutation(self, mutant):
        core_idx = rnd.randint(self.model.cores)
        center_idx = rnd.randint(self.max_centers)
        cord_idx = rnd.randint(2)
        mutant[core_idx][center_idx][cord_idx] += rnd.normal(0, 0.05 * self.model.x_size)  # TODO if x==y
        if mutant[core_idx][center_idx][cord_idx] < 0:
            mutant[core_idx][center_idx][cord_idx] = 0
        if mutant[core_idx][center_idx][cord_idx] > self.model.x_size:
            mutant[core_idx][center_idx][cord_idx] = self.model.x_size
        return mutant,

    def crossover(self, p1, p2):
        x_size = self.model.x_size
        y_size = self.model.y_size
        cores = self.model.cores
        max_centers = self.max_centers
        c1 = creator.ScheduleMixtureIndividual(np.zeros((cores, max_centers, 2)))
        c2 = creator.ScheduleMixtureIndividual(np.zeros((cores, max_centers, 2)))
        a, b = rnd.choice(range(cores), 2)
        a = np.minimum(a, b)
        b = np.maximum(a, b)
        i = 0
        while i < cores:
            if i < a or i > b:
                c1[i] = np.copy(p1[i])
                c2[i] = np.copy(p2[i])
            else:
                c1[i] = np.copy(p2[i])
                c2[i] = np.copy(p1[i])
            i += 1
        return c1, c2

    def __init__(self, model, outpath, max_centers=1, ext_sol=None, ):
        self.pool = Pool(10)
        # base params
        self.pop_size = 32
        self.generations = 100
        self.mut_prob = 0.5
        self.cross_prob = 0.15
        self.model = model
        self.max_centers = max_centers
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
        for x in range(self.model.x_size):
            for y in range(self.model.y_size):
                out_schedule.write("{}\t{}\n".format(x * self.model.y_size + y, best_schedule[x][y]))
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
    from model.model import AgentMobilityModel
    for iters in range(1):
        print("Iter = {}".format(iters))
        # Select input scenario file
        transportations_file = "..\\resources\\spb_passengers_center_100k_1"
        transportations = read_transportations(transportations_file)
        x_size = 30
        y_size = 30
        cores = 9
        ammodel = AgentMobilityModel(x_size, y_size, transportations, cores)
        outpath = "..\\tmp\\schedule_output.sched"
        scheduler = IntervalMixtureSchedGA(ammodel, outpath, max_centers=3, ext_sol=None)
        result = scheduler(iters * 144, (iters + 1) * 144)  # Call "__call__"
        best_solution = result[2].items[0]
        scheduler.write_solution(best_solution)


if __name__ == "__main__":
    main()
