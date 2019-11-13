import numpy as np
from scipy.spatial.distance import pdist, squareform

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from utils import reorder
from problems import Abstract_Prob, TSP
from solver import Abstract_Solver


class GA_Solver(Abstract_Solver):

    _pop_size = None
    _max_iter = None

    _radius = None
    _scaling = 1.0
    _alpha = 1

    def _sharing(self):
        def decorator(func):
            def wrapper(*args, **kargs):
                # calculate sharing factor for each individual
                individuals = list(map(reorder, args[0]))
                pd = squareform(pdist(individuals)) / self._scaling

                mask = pd < self._radius
                share = np.sum(np.multiply(1 - np.power(pd / self._radius, self._alpha), mask), axis=0)

                # rewrite fitnesses
                for i in range(len(individuals)):
                    args[0][i].fitness.setValues([v * share[i] for v in args[0][i].fitness.values])
                return func(*args, **kargs)

            return wrapper

        return decorator

    def __init__(self, pop_size=100, max_iter=100, radius=0.5):
        super().__init__()
        self._pop_size = pop_size
        self._max_iter = max_iter
        self._radius = radius


        # negative weight for minimisation
        if not hasattr(creator, 'FitnessMin'):
            creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create('Individual', np.ndarray, typecode='i', fitness=creator.FitnessMin)

    def solve(self, inst: Abstract_Prob, eval=None, niching=None):
        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register('indices', np.random.permutation, inst.n)

        # Structure initializers
        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register('select', tools.selTournament, tournsize=3)
        if eval is None:
            toolbox.register('mate', tools.cxPartialyMatched)
            toolbox.register('evaluate', inst.eval)
        else:
            # toolbox.register('mate', tools.cxOnePoint)
            # toolbox.register('mate', tools.cxTwoPoint)
            toolbox.register('mate', tools.cxPartialyMatched)
            toolbox.register('evaluate', eval)

        # introduce niching techniques
        if niching == 'fs':
            # fitness sharing as a decorator
            self._scaling = inst.n * (inst.n / 10.0 + 1.0) / 2.0
            toolbox.decorate('select', self._sharing())
        elif niching == 'dc':
            # deterministic crowding
            pass

        pop = toolbox.population(n=self._pop_size)

        # hof = tools.HallOfFame(1)
        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register('avg', np.mean)
        # stats.register('std', np.std)
        # stats.register('min', np.min)
        # stats.register('max', np.max)

        algorithms.eaSimple(pop, toolbox, 0.7, 0.2, self._max_iter, verbose=False)

        return pop


if __name__ == '__main__':
    tsp = TSP(50)
    slr = GA_Solver()
    pop = slr.solve(tsp, niching='fs')
    print(min(map(lambda x: x.fitness.values[0], pop)))
    pop = slr.solve(tsp)
    print(min(map(lambda x: x.fitness.values[0], pop)))
