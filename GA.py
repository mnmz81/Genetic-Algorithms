import numpy as np

import numpy as np
import os
import matplotlib.pyplot as plt


def two_point_crossover(a, b):
    index_1 = np.random.randint(0, len(a))
    index_2 = np.random.randint(0, len(b))
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    child_1 = np.copy(a[index_1:index_2])
    child_2 = np.copy(b[index_1:index_2])
    child_1 = np.concatenate([child_1, [i for i in b if i not in child_1]])
    child_2 = np.concatenate([child_2, [i for i in a if i not in child_2]])
    return child_1, child_2


def single_two_point_crossover(creator):
    index_1, index_2 = np.random.randint(0, len(creator)), np.random.randint(0, len(creator))
    if index_1 > index_2:
        index_1, index_2 = index_2, index_1
    child_1 = np.copy(creator[index_1:index_2])
    child_1 = np.concatenate([child_1, [i for i in creator if i not in child_1]])
    return child_1


def mutation_swap(a):
    """ swap two random indexes """
    index_1 = np.random.randint(0, len(a))
    index_2 = np.random.randint(0, len(a))
    a[index_1], a[index_2] = a[index_2], a[index_1]
    return a


def mutation_pttt(a):
    """ implements 'Push To The Top'  of an random index"""
    index = np.random.randint(0, len(a))
    out = a[index]
    np.delete(a, out)
    np.insert(a, 0, out)
    return a


def creatGraph():
    dirname = ""
    fname = os.path.join(dirname, "tokyo.dat")
    data = []
    with open(fname) as f:
        for line in f:
            data.append(line.split())
    n = len(data)
    G = np.empty([n, n])
    for i in range(n) :
        for j in range(i, n):
            G[i, j] = np.linalg.norm(np.array([float(data[i][1]), float(data[i][2])]) - np.array([float(data[j][1]),
                                                                                                  float(data[j][2])]))
            G[j, i] = G[i, j]
    return G, n


def fitness_selec(Graph, perm):
    tlen = 0.0
    for i in range(len(perm)):
        tlen += Graph[perm[i], perm[np.mod(i+1, len(perm))]]
    return 1 / tlen


def sexual_selection(genome, fitness):
    partial_sums = np.cumsum(fitness)
    best_individuals = np.ravel(np.where(sum(fitness) * np.random.uniform() < partial_sums))
    j = best_individuals[np.random.randint(0,len(best_individuals))]
    k = best_individuals[np.random.randint(0, len(best_individuals))]
    return genome[j, :], genome[k, :]


def choose_n_best(genome, genome_fitness, n):
    """ (mu + lambda) """
    new_genome = np.empty((n, np.size(genome[0, :])), dtype=int)
    for i in range(n):
        index = np.where(genome_fitness == np.max(genome_fitness))[0][0]
        new_genome[i, :] = genome[np.argmax(genome_fitness), :]
        genome_fitness = np.delete(genome_fitness, index)
    return new_genome


def choose_the_best(genome, genome_fitness):
    """ (1 + lambda) """
    return np.copy(genome[np.argmax(genome_fitness), :])


def GA(graph, n, max_evals, population, cross=two_point_crossover, mutation=mutation_swap, fitness=fitness_selec,
       selection=sexual_selection):
    eval_counter = 0
    history = []
    mu = population
    pm = 0.8
    # generate random genome
    genome = np.empty((mu, n), dtype=int)
    for i in range(mu):
        genome[i, :] = np.random.permutation(n)
    # calculate fitness
    fitness_vals = [fitness(graph, individual) for individual in genome]
    eval_counter += mu
    f_curbest = np.max(fitness_vals)
    f_best = f_curbest
    x_best = genome[np.argmax(fitness_vals), :]
    history.append(f_curbest)

    while eval_counter < max_evals:
        new_genome = np.empty((3*mu, n), dtype=int)
        for i in range(mu):
            new_genome[i, :] = np.copy(genome[i, :])
        # sexual selection
        for i in range(mu):
            in1, in2 = selection(genome, fitness_vals)
            in1, in2 = cross(in1), cross(in2)
            # mutation
            if np.random.uniform() < pm:
                in1 = mutation(in1)
            if np.random.uniform() < pm:
                in2 = mutation(in2)
            new_genome[mu + (2*i), :] = np.copy(in1)
            new_genome[mu + (2*i+1), :] = np.copy(in2)
        new_gen_fitness = [fitness(graph, individual) for individual in new_genome]
        genome = np.copy(choose_n_best(new_genome, new_gen_fitness, mu))

        # evaluation
        fitness_vals = [fitness(graph, individual) for individual in genome]
        eval_counter += mu
        f_curbest = np.max(fitness_vals)
        if f_best < f_curbest:
            f_best = f_curbest
            x_best = genome[np.argmax(fitness_vals), :]
        history.append(f_curbest)
       # if np.mod(eval_counter, max_evals // 10000) == 0:
        print(eval_counter, " evals: fmax=", 1/f_best)
    return x_best, f_best, history


def GA_single(graph, n, max_evals, children_amount, cross=single_two_point_crossover, mutation=mutation_swap,
              fitness=fitness_selec, selection=choose_the_best):
    eval_counter = 0
    history = []
    mu = children_amount
    pm = 0.8
    # generate random genome
    parent = np.random.permutation(n)
    # calculate fitness
    fitness_val = fitness(graph, parent)
    # eval_counter += 1
    f_curbest = fitness_val
    f_best = f_curbest
    x_best = parent
    history.append(f_curbest)

    while eval_counter < max_evals:
        new_genome = np.empty((mu+1, n), dtype=int)
        new_genome[-1, :] = np.copy(parent)
        # sexual selection
        for i in range(mu):
            child = cross(parent)
            # mutation
            if np.random.uniform() < pm:
                child = mutation(child)
            new_genome[i, :] = np.copy(child)
        new_gen_fitness = [fitness(graph, individual) for individual in new_genome]
        parent = np.copy(selection(new_genome, new_gen_fitness))

        # evaluation
        fitness_val = fitness(graph, parent)
        eval_counter += mu
        f_curbest = fitness_val
        if f_best < f_curbest:
            f_best = f_curbest
            x_best = parent
        history.append(f_curbest)
       # if np.mod(eval_counter, max_evals // 10000) == 0:
        print(eval_counter, " evals: fmax=", 1/f_best)
    return x_best, f_best, history


if __name__ == "__main__":
    NTrials = 10 ** 6
    graph, n = creatGraph()
    population = 1000
    tourStat = []
    # x_best, f_best, history = GA(graph, n, NTrials, population)
    x_best, f_best, history = GA_single(graph, n, NTrials, population)
    print(f"x: {x_best},\n f: {f_best}, d: { 1 /f_best}")
    plt.semilogy(history)
    plt.show()
