from simulation import * 
from pulse_encoding import * 
from gate_population import * 
from qiskit.circuit import Instruction

import numpy as np 
import matplotlib.pyplot as plt


class GeneticAlgorithm:

    def __init__(self, simulation_class, population_size, generations, mutation_rate=0.1, noise_config=None, verbose=True):
        """
        A genetic algorithm to evolve a population of dynamical decoupling sequences.
        
        - simulation_class: An instance of the IndividualSimulation class to run simulations.
        - population_size: The number of individuals (DD sequences) in the population.
        - generations: The number of generations for the GA to run.
        - mutation_rate: Probability of mutation occurring in offspring.
        - verbose: If True, prints out generation details.

        """
        self.simulation_class = simulation_class
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.noise_config = noise_config
        self.history = {'mitigated': [], 'unmitigated': [], 'max_mitigated': []}

    def generate_population(self):
        return self.simulation_class.generate_population()

    def fitness(self, individual):
        final_circuit, durations = self.simulation_class.generate_circuit(individual)
        if not self.noise_config:
            prob_unmitigated, probability, _ = self.simulation_class.fake_backend_sim()
        else:
            prob_unmitigated, probability = self.simulation_class.simulation(noise_config=self.noise_config)
        return probability, prob_unmitigated

    def selection(self, population, fitnesses):
        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        selected = random.choices(population, weights=probabilities, k=self.population_size)
        return selected

    def crossover(self, parent1, parent2):
        subsequences1 = self.simulation_class.split_into_identity_subsequences(parent1)
        subsequences2 = self.simulation_class.split_into_identity_subsequences(parent2)

        if subsequences1 and subsequences2:
            subseq1 = random.choice(subsequences1)
            subseq2 = random.choice(subsequences2)
        else:
            raise ValueError("Subsequences were not properly formed.")
        
        child = subseq1 + subseq2

        if not self.simulation_class.check_identity(child):
            child = self.simulation_class.fix_to_identity(child)

        return child

    def mutate(self, individual, max_mutation=4):
        if random.random() < self.mutation_rate:
            gate = random.choices(['X', 'Y', 'Z'])
            mutation_amount = random.choice(np.arange(2, max_mutation, 2))
            
            idxs = []
            count = 0
            for operator in range(len(individual)): 
                if individual[operator].name == gate[0].lower() and count < mutation_amount: 
                    idxs.append(operator)
                    count += 1
            if idxs:
                for index in sorted(idxs, reverse=True):
                    del individual[index]
        return individual

    def evolve(self):
        population, _ = self.generate_population()
        population_size = self.population_size

        for generation in range(self.generations):
            fitnesses = [self.fitness(individual) for individual in population]
            mitigated_fitnesses = [f[0] for f in fitnesses]
            unmitigated_fitnesses = [f[1] for f in fitnesses]
            
            avg_mitigated_fitness = sum(mitigated_fitnesses) / len(mitigated_fitnesses)
            avg_unmitigated_fitness = sum(unmitigated_fitnesses) / len(unmitigated_fitnesses)
            max_mitigated_fitness = max(mitigated_fitnesses)
            
            self.history['mitigated'].append(avg_mitigated_fitness)
            self.history['unmitigated'].append(avg_unmitigated_fitness)
            self.history['max_mitigated'].append(max_mitigated_fitness)

            if self.verbose:
                print(f"Generation {generation}: Avg mitigated fitness = {avg_mitigated_fitness}, Avg unmitigated fitness = {avg_unmitigated_fitness}, Max mitigated fitness = {max_mitigated_fitness}")

            remaining_population_size = random.randint((population_size + 1) // 2, max(1, population_size))
            population_size = remaining_population_size

            selected_population = self.selection(population, mitigated_fitnesses)
            selected_population = selected_population[:remaining_population_size]

            next_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[(i + 1) % len(selected_population)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)

            population = next_population

        return population

    def plot_progress(self, save_fig = True):
        plt.plot(range(self.generations), self.history['mitigated'], label='Mitigated Avg.')
        plt.plot(range(self.generations), self.history['unmitigated'], label='Unmitigated Avg.', linestyle='--')
        plt.plot(range(self.generations), self.history['max_mitigated'], label='Max Mitigated', linestyle='-.')
        plt.title('Genetic Algorithm Progress')
        plt.xlabel('Generation')
        plt.ylabel('Success Probability')
        plt.legend()
        
        if save_fig: 
            if self.noise_model: 
                plt.savefig('img/genetic_noise_model_results.png')
            else: 
                plt.savefig('img/genetic_fake_hardware_results.png')
        return 


if __name__ == "__main__": 
    
    ###########################
    ## EXAMPLE FAKE HARDWARE ##
    ###########################

    qc_example = QuantumCircuit(4)
    qc_example.h(0)
    qc_example.cx(0, 1)
    qc_example.cx(1, 2)
    qc_example.cx(2, 3)
    qc_example.barrier()

    simulation_instance = IndividualSimulation(qc_example, num_individuals=100, length_range=[50, 100], max_gate_reps=100, verbose=True)

    ga = GeneticAlgorithm(simulation_instance, population_size=100, generations=100, mutation_rate=0.1, verbose=True)
    final_population = ga.evolve()

    ga.plot_progress()

    print('OPTIMUM INDIVIDUAL')
    print(final_population)

    #########################
    ## EXAMPLE NOISE MODEL ##
    #########################

    noise_config = {
            'thermal_relaxation': {'t1': 50e3, 't2': 75e3, 'time': 100},
            'depolarization': {'probability': 0.01}
    }

    qc_example = QuantumCircuit(4)
    qc_example.h(0)
    qc_example.cx(0, 1)
    qc_example.cx(1, 2)
    qc_example.cx(2, 3)
    qc_example.barrier()

    simulation_instance = IndividualSimulation(qc_example, num_individuals=100, length_range=[50, 100], max_gate_reps=100, verbose=False)

    ga = GeneticAlgorithm(simulation_instance, population_size=100, generations = 20, mutation_rate=0.2, noise_config = noise_config, verbose=False)
    final_population = ga.evolve()

    ga.plot_progress()

    print('OPTIMUM INDIVIDUAL')
    print(final_population)



