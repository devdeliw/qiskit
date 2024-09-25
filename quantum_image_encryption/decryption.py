from hypersphere import Hypersphere 
from wave_functions import Scramble_WaveFunctions 

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, random_statevector, partial_trace
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile

from PIL import Image
from scipy.linalg import sqrtm

import numpy as np
import secrets, time, uuid, hashlib, os


class Decrypt: 

	def __init__(self, image_path, save_image_path, name,
				 width, height, n, depth, magnitudes, 
				 eigenvalues, eigenvectors, 
				 statevectors, weights, seeds): 

		self.image_path = image_path
		self.save_image_path = save_image_path
		self.name = name
		self.width = width
		self.height = height
		self.n = n
		self.depth = depth
		self.magnitudes = magnitudes
		self.eigenvalues = eigenvalues 
		self.eigenvectors = eigenvectors
		self.statevectors = statevectors
		self.weights = weights
		self.seeds = seeds

	def reverse_eigendecomposition(self): 

		# inverse spectral/eigendecomposition to recover 
		# the mixed density matrices from the eigenvalues and eigenvectors
		# of the pure states

		recovered_mixed_density_matrices = []

		for i in range(len(self.eigenvalues)): 

		    D = np.diag(self.eigenvalues[i][0])
		    VD = np.dot(self.eigenvectors[i][0], D)
		    recovered_mixed_density_matrices.append(np.dot(VD, np.conjugate(self.eigenvectors[i][0].T)))

		recovered_mixed_density_matrices = [DensityMatrix(i) for i in recovered_mixed_density_matrices]

		return recovered_mixed_density_matrices

	def recover_pure_states(self): 

		recovered_mixed_density_matrices = self.reverse_eigendecomposition()

		full_pure_states = []

		for idx in range(len(recovered_mixed_density_matrices)): 

			weights = self.weights[idx]
			eigenvalues, eigenvectors = np.linalg.eigh(recovered_mixed_density_matrices[idx])

			# approximate pure states
			pure_states = [np.outer(eigenvectors[:, i], eigenvectors[:, i].conj()) for i in range(len(eigenvalues))]

			reconstructed_mixed_density_matrix = sum(weights[i] * pure_states[i] for i in range(len(weights)))
			print(f" {idx} Eigenvalue Decomposition Difference:\n", recovered_mixed_density_matrices[idx] - reconstructed_mixed_density_matrix)

			pure_states = [DensityMatrix(i) for i in pure_states]


			full_pure_states.append(pure_states)

		return full_pure_states

	def get_statevectors(self): 

		full_pure_states = self.recover_pure_states()

		recovered_statevectors = []

		for i in full_pure_states: 
			ancillary_statevectors = []
			for j in i: 
				ancillary_statevectors.append(j.to_statevector())

			recovered_statevectors.append(ancillary_statevectors)

		return recovered_statevectors

	def generate_inverse_circuits(self): 

		# inverse every circuit used to encrypt the image via their seeds 
		# and qiskit.random.random_circuit()

		circuits = []

		for i in range(len(self.seeds)): 

			qc = random_circuit(self.n - 1, depth = depth, seed = self.seeds[i])
			circuits.append(qc.inverse())

		return circuits 

	def run_inverse_circuits(self)

		recovered_statevectors = self.get_statevectors()
		circuits = self.generate_inverse_circuits()
		statevectors = self.statevectors

		original_wave_functions = []

		def run(wave_func, i):

		    result_wf = []
		    main_circuit = inverse_circuits[i]

		    # create a circuit to first prepare the initial_wave_function 
		    # from a circuit starting from |0>|0>|0>...
		    initialization_circuit = QuantumCircuit(n - 1)
		    initialization_circuit.initialize(wave_func.data, 
		                                        [j for j in range(n - 1 )])

		    # append the randomized circuit to the initialization_circuit
		    full_circuit = initialization_circuit.compose(main_circuit)

		    # transpile the circuit onto the simulator backend
		    transpiled_circuit = transpile(full_circuit, simulator)
		    result = simulator.run(transpiled_circuit).result()

		    # output the resultant statevector
		    final_wave_function = result.get_statevector()
		    result_wf.append(final_wave_function)

		    return result_wf

		for idx in range(len(statevectors)): 
		    for statevector in statevectors[idx][0]: 
		        original_wave_functions.append(run(statevector, idx))

		return original_wave_functions

	def decrypt_image(self):

		original_wave_functions = self.run_inverse_circuits()

		# original image
		decrypted_image = Inverse_Hypersphere(width = self.width, height = self.height, 
                                               statevectors = original_wave_functions, 
                                               magnitudes = self.magnitudes, 
                                               name = self.name, save_image_path = self.save_image_path
                          ).recover_image()

		return



# would not actually be used for decryption, the outputted information would have to be 
# communicated to the receiver though
HypersphereClass = Hypersphere(n = 8, image_path = "images/el_primo_square.jpg", verbose = True)

pixel_values, width, height = HypersphereClass.get_pixels()
statevectors, magnitudes = HypersphereClass.generate_statevectors()



image_path = "/Users/devaldeliwala/quantum_image_encryption/images/encrypted/el_primo_square_516qubits.png"
save_image_path = "/Users/devaldeliwala/quantum_image_encryption/images/decrypted/"
name = "el_primo_square_516_decrypted.png"

width, height = 64,64
n = 4 
depth = 10
magnitudes = magnitudes


Decrypt(image_path, save_image_path, name,
	    width, height, n, depth, magnitudes, 
	    eigenvalues, eigenvectors, 
	    statevectors, weights, seeds
).decrypt_image()










	
