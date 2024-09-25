from hypersphere import Hypersphere

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, random_statevector, partial_trace
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile

from PIL import Image

from scipy.linalg import sqrtm

import numpy as np
import secrets, time, uuid, hashlib

class Scramble_WaveFunctions: 

    def __init__(self, image_path, n, depth, save_image_path, encrypted_name, verbose): 

        self.image_path = image_path
        self.depth = depth
        self.n = n
        self.save_image_path = save_image_path
        self.encrypted_name = encrypted_name
        self.verbose = verbose

    def get_image_size(self): 
        im = Image.open(self.image_path, 'r')
        image = im.convert("RGB")

        width, height = image.size

        return width, height

    def get_wave_functions(self): 

        wave_functions, magnitudes = Hypersphere(n = self.n, 
                                                image_path = self.image_path, 
                                                verbose = self.verbose
        ).generate_statevectors()

        return wave_functions

    def generate_circuits(self): 

        wave_functions = self.get_wave_functions()

        # generic seed generation coupling many different encryption mechanisms
        def generate_seed():

            entropy = secrets.token_bytes(16) + time.time_ns().to_bytes(8, 'big') + uuid.uuid4().bytes
            seed = int(hashlib.sha256(entropy).hexdigest(), 16) % 2**32

            return seed

        seeds = []
        circuits = []

        for i in range(len(wave_functions)): 

            num_wave_functions = self.n - 1

            # generate a randomized circuit of a certain depth using 
            # our generated seed
            seed = generate_seed() 
            circuit = random_circuit(num_wave_functions, depth = self.depth, seed = seed)

            circuits.append(circuit)
            seeds.append(seed)

        return circuits, seeds

    def group_wavefunctions(self): 

        wave_functions = self.get_wave_functions()

        groups = []

        for i in range(len(wave_functions)):
            group = []
            for j in range(self.n):
                group.append(wave_functions[(i + j) % len(wave_functions)])
            groups.append(group)

        return groups

    def run_circuits(self): 

        circuits, seeds = self.generate_circuits()
        groups = self.group_wavefunctions()

        # store circuits and seeds universally
        self.circuits = circuits
        self.seeds = seeds

        simulator = Aer.get_backend('statevector_simulator')
        final_wave_functions = []

        def run(wave_func, i):

            result_wf = []
            main_circuit = circuits[i]

            # create a circuit to first prepare the initial_wave_function 
            # from a circuit starting from |0>|0>|0>...
            initialization_circuit = QuantumCircuit(self.n - 1)
            initialization_circuit.initialize(wave_func.data, 
                                             [j for j in range(self.n - 1 )])

            # append the randomized circuit to the initialization_circuit
            full_circuit = initialization_circuit.compose(main_circuit)

            # transpile the circuit onto the simulator backend
            transpiled_circuit = transpile(full_circuit, simulator)
            result = simulator.run(transpiled_circuit).result()

            # output the resultant statevector
            final_wave_function = result.get_statevector()
            result_wf.append(final_wave_function)

            return result_wf

        final_wave_functions = []

        for i in range(len(groups)): 
            wave_functions = groups[i]
            ancillary_wave_functions = []

            for j in wave_functions: 
                print(j)
                ancillary_wave_functions.append(run(j, i))
            
            final_wave_functions.append(ancillary_wave_functions)

        return final_wave_functions

    def density_matrix(self): 

        final_wave_functions = self.run_circuits()
        density_matrices = []

        # converts the resultant pure states into their density matrices
        for i in final_wave_functions: 
            coupled_density_matrices = [] 

            for j in range(len(i)): 
                coupled_density_matrices.append(DensityMatrix(i[j][0]))

            density_matrices.append(coupled_density_matrices)

        return density_matrices

    def fidelities(self): 

        density_matrices = self.density_matrix()

        # generating a new seed based on the original seeds generated 
        # for the randomized circuits
        combined_string = ''.join(map(str, self.seeds))
        hash_object = hashlib.sha256(combined_string.encode())
        hex_dig = hash_object.hexdigest()[:5]
        seed = int(hex_dig, 16)

        # generate a temporary randomized statevector whhich we will 
        # use to calculate fidelities for each of the pure states
        randomized_pure_state = random_statevector(dims = 2**(self.n - 1), seed = seed)
        randomized_density_matrix = DensityMatrix(randomized_pure_state)

        fidelities = []

        def calculate_fidelitiy(rho, sigma): 

            sqrt_rho = sqrtm(rho)
            product = np.dot(sqrt_rho, np.dot(sigma, sqrt_rho))
            sqrt_product = sqrtm(product)
            trace = np.trace(sqrt_product)
            
            return np.real(trace)**2

        # calculate all the fidelities and store them in a list
        for coupled_states in density_matrices: 
            coupled_fidelities = []

            for matrix in coupled_states: 

                fidelity = calculate_fidelitiy(matrix, randomized_density_matrix)
                coupled_fidelities.append(fidelity)

            fidelities.append(coupled_fidelities)

        return fidelities, density_matrices

    def pure_to_mixed(self): 

        fidelities, density_matrices = self.fidelities()

        weights = [] 

        # using the calculated fidelities, generate self.n density matrix
        # coefficients that together sum to one
        for values in fidelities: 
            coupled_weights = []
            total = 0 

            for fidelity in values: 
                total += fidelity 

            for fidelity in values: 
                coupled_weights.append(fidelity / total)

            weights.append(coupled_weights)

        print(weights)

        mixed_density_matrices = []

        # generate mixed_states using the weights for each of the pure states
        # the mixed states are just generated via ∑ weight_n * |Ψ_n><Ψ_n| 
        for coupled_states_idx in range(len(density_matrices)):
            mixed_density_matrix = np.zeros((2**(self.n-1), 2**(self.n-1)), dtype=np.complex128)

            for matrix_idx in range(len(density_matrices[coupled_states_idx])):
                current_matrix = density_matrices[coupled_states_idx][matrix_idx].data

                mixed_density_matrix += weights[coupled_states_idx][matrix_idx] * current_matrix


            mixed_density_matrices.append(DensityMatrix(mixed_density_matrix))

        return mixed_density_matrices

    def spectral_decomposition(self): 

        density_matrices = self.pure_to_mixed()

        full_eigenvalues, full_eigenvectors = [], []

        # perform spectral/eigendecomposition on the resultant mixed states
        # to retrieve the eigenvalues and eigenvectors 
        for matrix in density_matrices: 

            is_unit_trace = np.isclose(np.trace(matrix), 1.0)
            is_hermitian = np.allclose(matrix, np.matrix(matrix.data).getH())

            # implement checks to ensure the provided density matrix is valid
            if not is_unit_trace: 
                raise Exception("Tr(ρ) ≠ 1")
            if not is_hermitian: 
                raise Exception("ρ not Hermitian")

            eigenvalues, eigenvectors = np.linalg.eigh(matrix.data)
            full_eigenvalues.append([eigenvalues])
            full_eigenvectors.append([eigenvectors])

            reconstructed_matrix = sum(eigenvalues[i] * np.outer(eigenvectors[:, i], 
                                       np.conj(eigenvectors[:, i])) for i in range(len(eigenvalues)))

            spectral_decomp_worked = np.allclose(matrix, reconstructed_matrix)

            if not spectral_decomp_worked: 
                raise Exception("spectral decomposition failed")

        return full_eigenvalues, full_eigenvectors 

    def entropy(self): 

        eigenvalues, eigenvectors = self.spectral_decomposition() 
        entropies = []

        # calculating the von-neumann entropy of the mixed_states 
        # using their eigenvalues
        def calculate_entropy(eigenvalues):  

            eigenvalues = np.array(eigenvalues)
            non_zero_eigenvalues = eigenvalues[eigenvalues > 0]

            entropy = -np.sum(non_zero_eigenvalues * np.log(non_zero_eigenvalues))

            return entropy

        for eigenlist in eigenvalues: 
            entropies.append(calculate_entropy(eigenlist))

        return entropies, eigenvalues, eigenvectors

    def organize_wave_functions(self): 

        entropies, eigenvalues, eigenvectors = self.entropy() 

        sorted_indices = np.argsort(entropies)[::-1]
        
        entropies = np.array(entropies)
        eigenvalues = np.array(eigenvalues)
        eigenvectors = np.array(eigenvectors)

        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[sorted_indices]

        organized_eigenvecs = []

        for i in range(len(eigenvalues)):
            group = [] 
            for j in range(len(eigenvalues[i][0])):
                group.append(eigenvalues[i][0][j])
            group_no0 = np.where(np.array(group) > 1e-10)
            minimum = np.where(min(eigenvalues[i][0][group_no0]))
            organized_eigenvecs.append(eigenvectors[i][0][group_no0[0][minimum]])

        statevectors = []
        for i in organized_eigenvecs: 
            statevectors.append(Statevector(i[0]))

        return statevectors

    def unit_vectors(self): 

        statevectors = self.organize_wave_functions()

        n2_dimensional_unit_vectors = []

        def statevector_to_unit_vector(statevector):
            real_parts = np.real(statevector)
            imag_parts = np.imag(statevector)
            combined = np.concatenate([real_parts, imag_parts])
            unit_vector = combined / np.linalg.norm(combined)
            return unit_vector

        for i in statevectors: 
            n2_dimensional_unit_vectors.append(statevector_to_unit_vector(i))

        return n2_dimensional_unit_vectors

    def encrypted_image(self): 

        n2_dimensional_unit_vectors = self.unit_vectors()
        width, height = self.get_image_size()

        hypersphere_class = Hypersphere(self.n, self.image_path)
        hypersphere_class.encrypt_image(unit_vector_groups = n2_dimensional_unit_vectors, 
                                        save_image_path = self.save_image_path, 
                                        width = width, height = height,
                                        name = self.encrypted_name)

        return



"""
image_path = "/Users/devaldeliwala/quantum_image_encryption/images/el_primo_skin.png"
save_image_path = "/Users/devaldeliwala/quantum_image_encryption/images/"
n = 8
depth = 10
encrypted_name = "el_primo_skin_encrypted"
verbose = False

class_ = Scramble_WaveFunctions(image_path = image_path, 
                               n = n, 
                               depth = depth,
                               save_image_path = save_image_path,
                               encrypted_name = encrypted_name,
                               verbose = verbose)

class_.encrypted_image()
"""








        






            


























