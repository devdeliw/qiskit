from qiskit_aer.noise import (
    NoiseModel, 
    depolarizing_error, 
    thermal_relaxation_error, 
    amplitude_damping_error, 
    phase_damping_error, 
    coherent_unitary_error
)
from qiskit_ibm_runtime.fake_provider import *
from qiskit_aer import Aer, QasmSimulator
from gate_population import *
from pulse_encoding import * 
import numpy as np 
import pandas as pd
import os, copy

class IndividualSimulation: 

    def __init__(self, circuit, num_individuals, max_gate_reps, 
                 basis_gates=[XGate(), YGate(), ZGate()], 
                 basis_names=['x', 'y', 'z'], 
                 approx_length = None, length_range = None, 
                 verbose=True): 

        self.circuit = circuit
        self.basis_gates = basis_gates
        self.approx_length = approx_length
        self.num_individuals = num_individuals
        self.length_range = length_range
        self.max_gate_reps = max_gate_reps
        self.basis_names = basis_names
        self.verbose = verbose 

    def generate_population(self): 

        if self.length_range:
            class_ = GeneratePopulation(
                basis_gates=self.basis_gates, 
                num_individuals=self.num_individuals, 
                length_range=self.length_range,
                max_gate_reps=self.max_gate_reps, 
                basis_names=self.basis_names,
                verbose=self.verbose
            )
        if self.approx_length: 
             class_ = GeneratePopulation(
                basis_gates=self.basis_gates, 
                num_individuals=self.num_individuals, 
                approx_length = self.approx_length,
                max_gate_reps=self.max_gate_reps, 
                basis_names=self.basis_names,
                verbose=self.verbose
            )
        population, names = class_.population()

        return population, names

    def generate_circuit(self, individual, gate_duration=50): 
        initial_circuit = self.circuit 

        class_ = GenerateConstantDD(
            circuit=initial_circuit, 
            dd_sequence=individual, 
            gate_duration=gate_duration, 
            verbose=self.verbose
        )

        final_circuit, durations = class_.transpile_n_inverse(save_circ = False)

        self.final_circuit = final_circuit
        self.durations = durations

        return final_circuit, durations

    def split_into_identity_subsequences(self, sequence):
        """
        Splits a given sequence into subsequences that net to the identity.
        A subsequence is valid if it results in an even number of X, Y, and Z gates (for now).
        Returns a list of subsequences (each subsequence is a list of gates).
        """
        subsequences = []
        current_subsequence = []
        x_count = y_count = z_count = 0

        for gate in sequence:
            current_subsequence.append(gate)
            
            # Update gate counts
            if gate.name == 'x':
                x_count += 1
            elif gate.name == 'y':
                y_count += 1
            elif gate.name == 'z':
                z_count += 1

            # Check if the current subsequence nets to the identity
            if x_count % 2 == 0 and y_count % 2 == 0 and z_count % 2 == 0:
                # Subsequence nets to identity; store it
                subsequences.append(current_subsequence)
                # Reset for next subsequence
                current_subsequence = []
                x_count = y_count = z_count = 0

        # If there are leftover gates that don't net to identity, append them anyway (may need fixing later)
        if current_subsequence:
            subsequences.append(current_subsequence)

        return subsequences



    def combine_subsequences(self, subsequences):
        """
        Combine identity-preserving subsequences into a full DD sequence.
        
        Args:
        - subsequences: A list of subsequences, each of which nets to the identity.
        
        Returns:
        - combined_sequence: The full DD sequence.
        """
        combined_sequence = []
        for subsequence in subsequences:
            combined_sequence.extend(subsequence)
        
        return combined_sequence

    def check_identity(self, sequence):
        """
        Check if a DD sequence nets to the identity.
        
        Args:
        - sequence: A list of gates (e.g., ['X', 'Y', 'Z']).
        
        Returns:
        - is_identity: True if the sequence nets to the identity, False otherwise.
        """
        # Assuming that your gates are ['X', 'Y', 'Z'], you can check
        # identity by ensuring that the sequence can be simplified to an identity operation.
        x_count = sequence.count('X') % 2
        y_count = sequence.count('Y') % 2
        z_count = sequence.count('Z') % 2

        # Identity occurs when all gate counts are even
        return (x_count == 0 and y_count == 0 and z_count == 0)

    def fix_to_identity(self, sequence):
        """
        Fix a sequence that doesn't net to the identity by adding necessary gates.
        
        Args:
        - sequence: A list of gates (e.g., ['X', 'Y', 'Z']).
        
        Returns:
        - fixed_sequence: The corrected sequence that nets to the identity.
        """
        x_count = sequence.count('X') % 2
        y_count = sequence.count('Y') % 2
        z_count = sequence.count('Z') % 2
        
        # Add gates to ensure that the sequence nets to identity
        if x_count == 1:
            sequence.append('X')
        if y_count == 1:
            sequence.append('Y')
        if z_count == 1:
            sequence.append('Z')
        
        return sequence

    def mutate_subsequence(self, subsequence):
        """
        Mutate a subsequence while ensuring it still nets to identity.
        
        Args:
        - subsequence: A subsequence that nets to identity.
        
        Returns:
        - mutated_subsequence: The mutated subsequence.
        """
        # Mutate by changing one gate at a random position
        mutate_index = random.randint(0, len(subsequence) - 1)
        mutated_gate = random.choice(self.basis_gates)  # Assuming 'basis_gates' is a list like ['X', 'Y', 'Z']
        
        # Swap the gate
        subsequence[mutate_index] = mutated_gate
        
        # Ensure the mutated subsequence still nets to identity
        return self.fix_to_identity(subsequence)

    def fake_backend_sim(self, backend=FakeAthensV2(), shots=1024, save_histogram=False, open_histogram = False): 

        if self.verbose:
            print(f'running simulation with `{backend.name}` simulator backend.')

        pm = PassManager(ALAPSchedule(durations=InstructionDurations(self.durations)))

        base_circuit = copy.deepcopy(self.circuit)
        base_circuit.compose(base_circuit.inverse(), inplace = True)
        base_circuit.measure_all()
   

        scheduled_circuit_unmitigated = pm.run(base_circuit)
        scheduled_circuit_mitigated = pm.run(self.final_circuit)

        job = backend.run(transpile(scheduled_circuit_mitigated, backend, scheduling_method='alap'), shots=shots)
        result = job.result()
        mitigated_counts = result.get_counts()
        successful_probability_mitigated = mitigated_counts['0' * self.circuit.num_qubits] / shots 

        job = backend.run(transpile(scheduled_circuit_unmitigated, backend, scheduling_method='alap'), shots=shots)
        result = job.result()
        unmitigated_counts = result.get_counts()
        successful_probability_unmitigated = unmitigated_counts['0' * self.circuit.num_qubits] / shots 

        if save_histogram: 
            from qiskit.visualization import plot_histogram
            filename = f'{backend.name}_sim_histogram_MITIGATED.png'
            plot_histogram(mitigated_counts).savefig(f'img/{filename}')
            if open_histogram: 
                os.system(f'open img/{filename}') 
            filename = f'{backend.name}_sim_histogram_UNMITIGATED.png'
            plot_histogram(unmitigated_counts).savefig(f'img/{filename}')
            if open_histogram: 
                os.system(f'open img/{filename}')

        if self.verbose: 
            print(f'\nUNMITIGATED SUCCESSFUL PROBABILITY: {successful_probability_unmitigated}')
            print(f'MITIGATED SUCCESSFUL PROBABILITY: {successful_probability_mitigated}\n')

        return successful_probability_unmitigated, successful_probability_mitigated, backend.name

    def noise_model_(self, noise_config): 
        self.noise_model = NoiseModel() 

        # thermal relaxation noise
        if 'thermal_relaxation' in noise_config: 
            t1 = noise_config['thermal_relaxation'].get('t1', None)
            t2 = noise_config['thermal_relaxation'].get('t2', None)
            time = noise_config['thermal_relaxation'].get('time', 100)

            if t1 is not None and t2 is not None: 
                thermal_error = thermal_relaxation_error(t1, t2, time)
                self.noise_model.add_all_qubit_quantum_error(thermal_error, ['id', 'delay'])
            else: 
                raise ValueError('for thermal relaxation, t1 & t2 must be provided.')

        # depolarizing noise
        if 'depolarization' in noise_config: 
            probability = noise_config['depolarization'].get('probability', None)

            if probability is not None: 
                depol_error = depolarizing_error(probability, 1)
                self.noise_model.add_all_qubit_quantum_error(depol_error, ['id', 'delay', 'rx', 'ry', 'rz'])
            else: 
                raise ValueError('depolarization requires a probability parameter.')

        # amplitude damping noise
        if 'amplitude_damping' in noise_config:
            gamma = noise_config['amplitude_damping'].get('gamma', None)

            if gamma is not None:
                amplitude_error = amplitude_damping_error(gamma)
                self.noise_model.add_all_qubit_quantum_error(amplitude_error, ['id', 'delay'])
            else:
                raise ValueError("Amplitude damping requires a gamma parameter.")

        # phase damping noise
        if 'phase_damping' in noise_config:
            gamma = noise_config['phase_damping'].get('gamma', None)

            if gamma is not None:
                phase_error = phase_damping_error(gamma)
                self.noise_model.add_all_qubit_quantum_error(phase_error, ['id', 'delay'])
            else:
                raise ValueError("Phase damping requires a gamma parameter.")

        # coherent unitary errors
        if 'coherent_unitary' in noise_config:
            theta_error = noise_config['coherent_unitary'].get('theta_error', None)

            if theta_error is not None:
                unitary_error = coherent_unitary_error([[1, 0], [0, np.exp(1j * theta_error)]])
                self.noise_model.add_all_qubit_quantum_error(unitary_error, ['rx', 'ry', 'rz'])
            else:
                raise ValueError("Coherent unitary error requires a theta_error parameter.")

        return self.noise_model

    def simulation(self, shots=1024, noise_config=None, save_histogram = False, open_histogram = False): 
        if noise_config is not None:
            self.noise_model_(noise_config)
        if self.noise_model is None: 
            raise ValueError(
                'noise model has not been built. either define noise_config here '
                'or run self.noise_model(noise_config)'
            )

        base_circuit = copy.deepcopy(self.circuit)
        base_circuit.compose(base_circuit.inverse(), inplace = False)
        base_circuit.measure_all()

        pm = PassManager(ALAPSchedule(durations=InstructionDurations(self.durations)))

        scheduled_qc_unmitigated = pm.run(base_circuit)
        scheduled_qc_mitigated = pm.run(self.final_circuit)

        backend = QasmSimulator()
        job = backend.run(
            transpile(scheduled_qc_unmitigated, backend, scheduling_method='alap'), 
            noise_model=self.noise_model, 
            shots=shots
        )
        result = job.result()
        unmitigated_counts = result.get_counts() 
        successful_probability_unmitigated = unmitigated_counts['0' * self.circuit.num_qubits] / shots

        backend = QasmSimulator()
        job = backend.run(
            transpile(scheduled_qc_mitigated, backend, scheduling_method='alap'), 
            noise_model=self.noise_model, 
            shots=shots
        )
        result = job.result()
        mitigated_counts = result.get_counts() 
        successful_probability_mitigated = mitigated_counts['0' * self.circuit.num_qubits] / shots

        if save_histogram: 
            from qiskit.visualization import plot_histogram
            filename = 'Aer_noise_sim_histogram_MITIGATED.png'
            plot_histogram(mitigated_counts).savefig(f'img/{filename}')
            if open_histogram:
                os.system(f'open img/{filename}')
            filename = 'Aer_noise_sim_histogram_UNMITIGATED.png'
            plot_histogram(unmitigated_counts).savefig(f'img/{filename}')
            if open_histogram:
                os.system(f'open img/{filename}')

        if self.verbose: 
            print(f'\n{self.noise_model}\n')
            print(f'UNMITIGATED SUCCESSFUL PROBABILITY: {successful_probability_unmitigated}')
            print(f'MITIGATED SUCCESSFUL PROBABILITY: {successful_probability_mitigated}\n')

        return successful_probability_unmitigated, successful_probability_mitigated

class FullSimulation: 

    def __init__(self, circuit, num_individuals, max_gate_reps, 
                 basis_gates = [XGate(), YGate(), ZGate()], 
                 basis_names = ['x', 'y', 'z'], 
                 length_range = None, approx_length = None, 
                 verbose = True): 

        self.circuit = circuit
        self.num_individuals = num_individuals
        self.basis_gates = basis_gates
        self.max_gate_reps = max_gate_reps
        self.basis_names = basis_names
        self.length_range = length_range
        self.approx_length = approx_length
        self.verbose = verbose

        if self.approx_length:
            self.class_ = IndividualSimulation(circuit=self.circuit, 
                                                   num_individuals=self.num_individuals, 
                                                   max_gate_reps=self.max_gate_reps,
                                                   approx_length=self.approx_length, 
                                                   basis_gates=self.basis_gates, 
                                                   basis_names=self.basis_names, 
                                                   verbose=self.verbose)

            self.population, self.names = self.class_.generate_population()

        if self.length_range:
            self.class_ = IndividualSimulation(circuit=self.circuit, 
                                                   num_individuals=self.num_individuals, 
                                                   max_gate_reps=self.max_gate_reps,
                                                   length_range = self.length_range, 
                                                   basis_gates=self.basis_gates, 
                                                   basis_names=self.basis_names, 
                                                   verbose=self.verbose)

            self.population, self.names = self.class_.generate_population()

    def full_pop_backend_sim(self, backend = [FakeAthensV2(), FakeJakartaV2()], shots = 1024, save_histogram = False): 
        probabilities = {'backend': [], 
                         'mitigated_success': [], 
                         'unmitigated_success': [], 
                         'sequence': []}

        fitness_table = pd.DataFrame(probabilities)
        fitness_table = fitness_table.astype('object')

        count = 0
        for i in range(len(self.population)): 
            self.class_.generate_circuit(individual = self.population[i])
            self.names[i] = ''.join(self.names[i])
            for j in range(len(backend)):
                backend_ = backend[j]


                successful_probability_unmitigated, successful_probability_mitigated, backend_name = self.class_.fake_backend_sim(backend = backend_, 
                                                                                                                                  save_histogram = save_histogram, 
                                                                                                                                  open_histogram = False)

                fitness_table.at[count, 'backend'] = backend_.name
                fitness_table.at[count, 'mitigated_success'] = successful_probability_mitigated
                fitness_table.at[count, 'unmitigated_success'] = successful_probability_unmitigated
                fitness_table.at[count, 'sequence'] = self.names[i]
                count += 1

        return fitness_table

    def full_pop_noise_sim(self, noise_config, shots, open_histogram = False, save_histogram = False): 
        probabilities = {'sequence': [], 
                         'mitigated_success': [], 
                         'unmitigated_success': []}

        fitness_table = pd.DataFrame(probabilities)
        fitness_table = fitness_table.astype('object')

        for i in range(len(self.population)): 
            self.class_.generate_circuit(individual = self.population[i])
            self.names[i] = ''.join(self.names[i])

            successful_probability_unmitigated, successful_probability_mitigated  = self.class_.simulation(shots = shots, 
                                                                                                           noise_config = noise_config, 
                                                                                                           save_histogram = save_histogram, 
                                                                                                           open_histogram = open_histogram)      
            fitness_table.at[i, 'mitigated_success'] = successful_probability_mitigated
            fitness_table.at[i, 'unmitigated_success'] = successful_probability_unmitigated
            fitness_table.at[i, 'sequence'] = self.names[i]

        return fitness_table








if __name__ == '__main__': 
    noise_config = {
        'thermal_relaxation': {'t1': 50e3, 't2': 75e3, 'time': 100},
        'depolarization': {'probability': 0.02}
    }

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.barrier()

    class_ = FullSimulation(
        circuit = qc, num_individuals = 10, length_range=[4, 20], 
        max_gate_reps = 3, basis_gates=[XGate(), YGate(), ZGate()], 
        basis_names=['x', 'y', 'z'], verbose=False
    )

    probabilities = class_.full_pop_noise_sim(noise_config = noise_config, shots = 1024)
    print("\n", probabilities, "\n")


    """
    population = class_.generate_population()
    individual = population[0]

    class_.generate_circuit(individual = individual)
    class_.fake_backend_sim()
    """
