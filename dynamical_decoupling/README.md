# Dynamical Decoupling with Genetic Algorithms

This repository provides tools and simulations for generating, customizing, evolving, and optimizing **Dynamical Decoupling (DD)** sequences using **Genetic Algorithms (GA)**. The DD sequences generated help in mitigating decoherence and noise in quantum systems. The repository offers various noise models, including **thermal relaxation**, **amplitude damping**, **phase damping**, and **depolarization**, to simulate and test the effectiveness of DD sequences. A class for running on `FakeBackends` is also available. 

## Features

- **Population-based DD Sequence Generation:** Generate an initial population of DD sequences using specified basis gates.
- **Noise Models:** Simulate under various noise models such as thermal relaxation, amplitude damping, depolarization, phase damping, and coherent unitary errors.
- **Genetic Algorithm Optimization:** Apply crossover and mutation to evolve DD sequences over generations, maximizing success probability under noisy conditions.
- **Simulation Backend Integration:** Simulate the performance of DD sequences on noisy quantum backends using **Qiskit**.
- **Data Analysis:** Track the evolution of the population and visualize performance metrics such as average success probability over generations.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Generate Population](#generate-population)
   - [Run Simulations](#run-simulations)
   - [Use Genetic Algorithm](#use-genetic-algorithm)
3. [File Structure](#file-structure)
4. [Examples](#examples)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run this project, you'll need the following dependencies:

- **Python 3.7+**
- **Qiskit** (for quantum simulations)
- **Numpy** (for numerical computations)
- **Matplotlib** (for plotting)
- **Scipy** (optional, for advanced mathematical computations)

### Install dependencies

```bash
pip install qiskit qiskit_aer numpy matplotlib scipy
```

### Clone the repository

```bash
git clone https://github.com/devdeliw/qiskit.git
cd qiskit/dynamical_decoupling/
```

## Usage

### 1. Generate Population

To generate a population of DD sequences, initialize the `GeneratePopulation` class by specifying the basis gates, number of individuals, and the length of the sequences. You can define your own basis gates or use standard Pauli gates (`XGate`, `YGate`, `ZGate`).

*In `genetic_algorithm/` directory*

```python
from gate_population import GeneratePopulation

population_generator = GeneratePopulation(
    basis_gates=[XGate(), YGate(), ZGate()], 
    num_individuals=100, 
    length_range=(4, 8), 
    max_gate_reps=10
)

population = population_generator.population
```

### 2. Run Simulations

To evaluate the performance of DD sequences, use the `FullSimulation` class, which runs simulations on a full population and calculates the success probability for each individual.

*In `genetic_algorithm/` directory*

```python
from simulation import FullSimulation
from qiskit_aer.noise import NoiseModel

# Define a noise configuration
noise_config = {
    'thermal_relaxation': {'t1': 50e3, 't2': 30e3, 'time': 100},
    'depolarization': {'probability': 0.02},
    'amplitude_damping': {'gamma': 0.01},
}

sim = FullSimulation(
    population=population, 
    noise_config=noise_config, 
    backend=FakeOsaka()
)

# Run the simulation and get results
results = sim.run_simulation()
```

### 3. Use Genetic Algorithm

The genetic algorithm framework optimizes the DD sequences by evolving them through selection, crossover, and mutation operations. This allows the discovery of sequences with improved performance under noisy conditions.

*In `genetic_algorithm/` directory*

```python
from genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm(
    population=population, 
    mutation_rate=0.05, 
    crossover_rate=0.7
)

# Run evolution over several generations
for generation in range(10):
    ga.evolve()
    avg_success = ga.evaluate_population()
    print(f"Generation {generation}: Avg Success = {avg_success}")
```

## File Structure

The main files in this repository in `genetic_algorithm/` include:

- **`simulation.py`**: Handles the simulation of individual DD sequences and populations of sequences under different noise models.
- **`gate_population.py`**: Contains classes for generating the population of DD sequences.
- **`pulse_encoding.py`**: Defines methods for encoding pulse sequences into quantum circuits.
- **`genetic_algorithm.py`**: Implements the genetic algorithm to evolve DD sequences over generations.

### Directory Layout

```text
dynamical-decoupling-ga/
│
├── genetic_algorithm/ 
│   ├── img/                    # images generated
│   ├── simulation.py           # core simulation logic
│   ├── gate_population.py      # DD sequence generation
│   ├── pulse_encoding.py       # pulse encoding for DD sequences
│   └── genetic_algorithm.py    # genetic algorithm
├── notebooks/   
│   ├── XY4.ipynb     
│   ├── dd_genetic_algo.ipynb       
│   ├── dd_pulse_scheduling.ipynb 
│   ├── test_pulse.ipynb
└── └── test_simulation.ipynb                  
                        
```

## Examples

Here is an example of how to use the full pipeline:

1. **Generate DD sequences**: Create a population of sequences using the Pauli basis gates (`X`, `Y`, `Z`).
2. **Simulate under noise**: Test the sequences under thermal relaxation and depolarizing noise.
3. **Apply genetic algorithm**: Evolve the sequences over several generations to improve the success rate.

The following Jupyter notebooks in the `notebooks/` directory provide detailed examples for these steps.

## Contributing

Contributions are welcome. If you'd like to add new features, report issues, or improve the code, feel free to open a pull request.

## License

This project is licensed under the MIT License.
