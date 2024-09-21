from qiskit.circuit.library import XGate, YGate, ZGate, UnitaryGate
from qiskit.quantum_info import Operator
import numpy as np
import random 

class GeneratePopulation:

    """
    Generates a population of pulse gate sequences using gates in `HIGH_FIDELITY_GATES`. 
    Single-qubit gates were chosen because of their high-fidelity. 

    The pulse sequences are generated via building arbitrary subsequences using 
    gates defined in `basis_gates`. These subsequences all tensor to the identity 
    matrix, up to an overall phase. 


    Methods
    -------

    def generate_individuals(self):
        generates one randomized sequence of pulse-gates using the provided class 
        parameterso build a single individual.
    
    def infant_product(self, infant):
        calculates the net unitary matrix from a list (`infant`) of matrices. 

    def is_identity(self, matrix): 
        checks whether matrix is quantum mechanically equivalent to an identity
        matrix (up to an overall global phase). 

    def population(self): 
        generates a pulse-gate population of randomized individuals of size 
        `num_individuals`. 

    Example
    -------
    class_ = GeneratePopulation(basis_gates=[XGate(), YGate(), ZGate()], 
                                num_individuals = 100, 
                                length_range = [4, 20],
                                max_gate_reps = 4, 
                                gate_names = ['X', 'Y', 'Z'],
                                verbose = True)

    class_.population()

    - Generates a population of 100 individuals. 
    - Each individual is made up of Pauli X, Y, Z gates. 
    - The gate depth (# gates) within an individual is between 4 and 20. 
    - A subsequence of an individual has at most 4 repetitions of the same gate. 
    
    """

    def __init__(self, basis_gates, num_individuals, max_gate_reps, 
                 approx_length = None, length_range = None, 
                 basis_names = None, verbose = True): 
        """
        Parameters
        ----------

        basis_gates: array-like
            stores the gates to use for generating randomized pulse gate sequences.

        num_individuals: int
            number of individuals in a population to generate.

        max_gate_reps: int 
            the maximum number of repetitions of a basis gate within a subsequence.

        OPTIONAL
        --------
        approx_length: int
            if defined, all individuals / pulse sequences in a population will have
            roughly the same gate depth = approx_length.

        length_range: [int_min, int_max] 
            if defined, all individuals / pulse sequences will be of randomized 
            depth between int_min and int_max.

        *** either approx_length or length_range must be defined ***

        basis_names : [string_1, string_2, ..., string_len(basis_states)]
            names of the operators in basis_states

        verbose: bool 
            to print out additional information when running

        """

        self.basis_gates = basis_gates
        self.num_individuals = num_individuals
        self.max_gate_reps = max_gate_reps
        self.approx_length = approx_length
        self.length_range = length_range
        self.basis_names =  basis_names
        self.verbose = verbose
        self.count = 0

    def generate_individual(self):

        """
        Generates randomized pulse-gate individual/sequence 

        Returns: 
        --------
        individual: list 
            list of matrices (numpy arrays) that make up the individual
        individual_name: 
            if self.basis_names provided, returns a simplified version of 
            the individual sequence using the names of each gate. 
            (i.e. ['X', 'Y', 'X', 'Y'] for XY4). Otherwise []. 

        """ 

        individual = []
        fixed_length = False

        if self.approx_length: 
            fixed_length = True
        elif self.length_range: 
            fixed_length = False
        else: 
            raise Exception('Either `self.approx_length` or `self.length_range` must be defined')

        if fixed_length: 
            while len(individual) < self.approx_length: 
                infant = [] 
                num_reps = random.choice(range(2, self.max_gate_reps, 2))
                basis_idxs = [random.randint(0, len(self.basis_gates) - 1) for j in range(num_reps)]

                for idx in basis_idxs: 
                    infant.append(self.basis_gates[idx])

                if self.is_identity(self.infant_product(infant)):
                    individual.extend(infant)
        else: 
            length = random.randint(self.length_range[0], self.length_range[1])
            while len(individual) < length: 
                infant = [] 
                num_reps = random.choice(range(2, self.max_gate_reps, 2))
                basis_idxs = [random.randint(0, len(self.basis_gates) - 1) for j in range(num_reps)]

                for idx in basis_idxs: 
                    infant.append(self.basis_gates[idx])

                if self.is_identity(self.infant_product(infant)):
                    individual.extend(infant)

        individual_name = []
        if self.basis_names: 
            for i in individual: 
                for j in range(len(self.basis_gates)): 
                    if np.allclose(i, self.basis_gates[j], atol = 1e-6): 
                      individual_name.append(self.basis_names[j])

            if self.verbose: 
                print(f"Individual {self.count+1}")
                print(f" {str(individual_name):>13}")

        elif self.verbose: 
            print(f"Individual {self.count} generated.")

        return individual, individual_name

    def infant_product(self, infant): 

        # Calculates net unitary matrix from infant subsequence

        if not infant: 
            raise Exception('Infant was never born')

        return np.linalg.multi_dot(infant)

    def is_identity(self, matrix, tolerance = 1e-6): 

        # checks whether net unitary matrix from infant subsequence
        # is equivalent to identity matrix (up to overall global phase)

        identity_matrix = np.eye(matrix.shape[0])

        if matrix[0, 0] == 0: 
            return False

        normalized = matrix / matrix[0, 0]

        return np.allclose(normalized, identity_matrix, atol=tolerance)

    def population(self): 
        
        # Generates population of length `num_individuals`
        # Each individual pseudo-randomized based on class parameters.

        population  = []
        names = []
        for i in range(self.num_individuals): 
            individual, individual_name = self.generate_individual()
            individual = self.individual_to_dd(individual)

            if individual not in population: 
                population.append(individual)
                names.append(individual_name)

            self.count += 1

        if self.verbose:
            print("Population generated.\n")

        return population, names

    def individual_to_dd(self, individual):
        dd_individual = []

        for gate_idx in range(len(individual)): 
            for basis_gate_idx in range(len(self.basis_gates)): 
                if np.allclose(Operator(individual[gate_idx]), self.basis_gates[basis_gate_idx], atol = 1e-6):
                    unitary_gate = UnitaryGate(self.basis_gates[basis_gate_idx])
                    unitary_gate.name = self.basis_names[basis_gate_idx]
                    dd_individual.append(unitary_gate)
        return dd_individual





if __name__ == '__main__': 
    class_ = GeneratePopulation(basis_gates=[XGate(), YGate(), ZGate()], 
                                num_individuals = 10, 
                                approx_length = 6,
                                max_gate_reps = 3, 
                                basis_names = ['x', 'y', 'z'],
                                verbose = True)

    population = class_.population()
    print(population)










