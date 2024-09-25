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

def encrypt(): 

	# set path to image to encrypt
	image_path = "/Users/devaldeliwala/quantum_image_encryption/images/el_primo_2_square.png"

	if not os.path.isfile(image_path): 
		raise Exception("image file does not exist")

	# directory to place encrypted image
	encrypted_name = "el_primo_2_square"
	save_image_path = "/Users/devaldeliwala/quantum_image_encryption/images/encrypted/"

	if not os.path.isdir(save_image_path):
		os.makedirs(save_image_path)

	# starting test `n`
	n = 6

	# depth of circuits to scramble
	depth = 10 

	verbose = False

	success = False
	while not success: 
		try: 
			encrypt = Scramble_WaveFunctions(image_path = image_path, 
											 n = n, 
											 depth = depth, 
											 save_image_path = save_image_path, 
											 encrypted_name = encrypted_name, 
											 verbose = verbose
			)

			# encrypt the image
			encrypt.encrypted_image()
			success = True

		except: 
			n+=1

encrypt()
