from PIL import Image
from qiskit.quantum_info import Statevector
from scipy.interpolate import Rbf

import numpy as np 

class Hypersphere: 

    """
    Generates a 2^n dimensional Hypersphere containing the 
    RGB information of pixels in an image.

    The Hypersphere is afterwards used to reversibly generate 
    n-dimensional quantum wavefunctions/statevectors

    Parameters: 
    -----------

    n           : int 
    # of qubits per statevector you wish to generate 

    image_path  : str
    location of image for hypersphere generation

    verbose     : bool
    to print out information regarding the image and statevectors
    generated 


    Methods: 
    --------

    def get_pixels(self): 
        returns the RGB pixel information and image size 

    def pixel_to_int(self): 
        reversibly converts all the (R,G,B) information per pixel 
        into an single integer

    def hypersphere(self): 
        groups n integers into a set and converts them to unit vectors
        on an n-dimensional sphere

    def statevectors(self): 
        using the hypersphere and the groups of 2^n dimensional unit vectors,
        reversibly generates n-dimensional statevectors 

    """

    def __init__(self, n, image_path, verbose = False): 

        self.image_path = image_path
        self.n = n 
        self.verbose = verbose 

    def get_pixels(self): 

        # load image
        im = Image.open(self.image_path, mode = 'r')
        image = im.convert('RGB')

        width, height = image.size

        # [(r, g, b), ...]
        pixel_values = np.array(list(image.getdata()))

        if self.verbose: 
            print(f"Image Size: {width} x {height}")

        return pixel_values, width, height

    def pixel_to_int(self): 

        pixel_values, width, height = self.get_pixels()

        def rgb_to_int(r, g, b):

            return (r << 16) + (g << 8) + b

        int_values = []

        for r, g, b in pixel_values: 
            int_values.append(rgb_to_int(r, g, b))
        int_values = np.array(int_values)

        return int_values

    def group(self): 

        int_values = self.pixel_to_int() 
        groups = []

        i = 0
        while i < len(int_values): 
            groups.append(int_values[i:i + 2**self.n])
            i += 2**self.n

        # converts 2^n dimensional arrays filled with ints
        # to unit vectors defining a hypersphere
        def to_unit_vector(arr):

            magnitude = np.linalg.norm(arr)

            if magnitude == 0:
                return arr, magnitude

            unit_vector = arr / magnitude
            return unit_vector, magnitude

        unit_vector_groups = []
        magnitudes = [] 

        for group in groups:
            unit_vector, magnitude = to_unit_vector(group)
            magnitudes.append(magnitude)
            unit_vector_groups.append(unit_vector)

        return unit_vector_groups, magnitudes

    def generate_statevectors(self): 

        unit_vector_groups, magnitudes = self.group()

        # reversibly converts 2^n dimensional unit vectors
        # to n-1 dimensional statevectors
        def unit_vector_to_statevector(unit_vector):

            dim = len(unit_vector) // 2
            real_parts = unit_vector[:dim]
            imag_parts = unit_vector[dim:]
            statevector = real_parts + 1j * imag_parts
            statevector /= np.linalg.norm(statevector)

            return statevector

        statevectors = []
        for i in unit_vector_groups: 
            statevectors.append(unit_vector_to_statevector(i))

        # statevector object from qiskit.quantum_info
        statevectors = [Statevector(i) for i in statevectors]

        return statevectors, magnitudes

    def hypersphere(self): 

        pixel_values, width, height = self.get_pixels()
        unit_vector_groups, magnitudes = self.group()

        self.rbf_r_list = []
        self.rbf_g_list = []
        self.rbf_b_list = []

        r_values = [pixel[0] for pixel in pixel_values]
        g_values = [pixel[1] for pixel in pixel_values]
        b_values = [pixel[2] for pixel in pixel_values]

        # set rbf for each dimension separately
        for i in range(2**self.n):
            rbf_r = Rbf(*np.array(unit_vector_groups).T, r_values[i::2**self.n], function='multiquadric')
            rbf_g = Rbf(*np.array(unit_vector_groups).T, g_values[i::2**self.n], function='multiquadric')
            rbf_b = Rbf(*np.array(unit_vector_groups).T, b_values[i::2**self.n], function='multiquadric')
            self.rbf_r_list.append(rbf_r)
            self.rbf_g_list.append(rbf_g)
            self.rbf_b_list.append(rbf_b)

        return

    def recover_pixels(self, unit_vector_groups):

        # initialize hypersphere
        self.hypersphere()

        recovered_pixel_values = []
        for j in unit_vector_groups: 
            r_values = [self.rbf_r_list[i](*j) for i in range(2**self.n)]
            g_values = [self.rbf_g_list[i](*j) for i in range(2**self.n)]
            b_values = [self.rbf_b_list[i](*j) for i in range(2**self.n)]

            rgb_values = np.array([np.clip([r, g, b], 0, 255).astype(int) for r, g, b in zip(r_values, g_values, b_values)])
            recovered_pixel_values.append(rgb_values)

        return recovered_pixel_values

    def encrypt_image(self, unit_vector_groups, width, height, save_image_path, name): 

        recovered_pixel_values = self.recover_pixels(unit_vector_groups)
        num_pixels = width * height
        num_qubits = int((num_pixels / (2**self.n) + 1) * (self.n - 1))

        final_pixels = []
        for i in recovered_pixel_values: 
            for j in i: 
                final_pixels.append(j)
        final_pixels = np.array(final_pixels).reshape(height, width, 3)
        
        reconstructed_image = Image.fromarray(np.uint8(final_pixels), 'RGB')

        reconstructed_image.save(f'{save_image_path}{name}-{num_qubits}_qubits.png')

        return









class Inverse_Hypersphere: 

    """
    Converts the outputted wavefunctions/statevectors from Hypersphere() 
    back into the original image

    Parameters: 
    -----------

    width, height   : int
    width and height of image

    statevectors    : array-like 
    list of n-dimensional statevectors that together store the information 
    of a digital image 

    magnitudes      : array-like 
    list of the normalization magnitudes that converted the 2^n dimensional vectors 
    into 2^n dimensional unit vectors 

    name            : str
    name of outputted image 

    name            : save_image_path
    location of directory to store reconstructed image


    Methods:
    --------

    def recover_hypersphere(self): 
        recovers the 2^n dimensional unit vectors that define the hypersphere
        from the statevectors parameter

    def recover_pixels(self): 
        recovers the pixel (R,G,B) information from the 2^n dimensional unit
        vectors

    def recover_image(self): 
        generates reconstructed image using the outputted pixel information
        saves the reconstructed image to save_image_path parameter

    """

    def __init__(self, width, height, statevectors, magnitudes, name, save_image_path): 

        self.width = width 
        self.height = height
        self.statevectors = statevectors
        self.magnitudes = magnitudes
        self.name = name
        self.save_image_path = save_image_path

    def recover_groups(self): 

        # converts the n-dimensional statevectors back into 
        # 2^n dimensional unit vectors
        def statevector_to_unit_vector(statevector):
            real_parts = np.real(statevector)
            imag_parts = np.imag(statevector)
            combined = np.concatenate([real_parts, imag_parts])
            unit_vector = combined / np.linalg.norm(combined)
            return unit_vector

        recovered_unit_vectors = []

        for i in self.statevectors: 
            recovered_unit_vectors.append(statevector_to_unit_vector(i.data))

        return recovered_unit_vectors

    def recover_pixels(self): 

        recovered_unit_vectors = self.recover_groups()

        def decimal_to_rgb(decimal):
            r = (decimal >> 16) & 0xFF
            g = (decimal >> 8) & 0xFF
            b = decimal & 0xFF

            return r, g, b

        # converts the unit vectors back into their original 
        # 2^n dimensional integer arrays using the stored normalization
        # magnitudes
        int_vectors = []
        for i in range(len(recovered_unit_vectors)): 
            int_vectors.append(recovered_unit_vectors[i] * self.magnitudes[i])

        recovered_int_values = []
        for i in int_vectors: 
            for j in i: 
                recovered_int_values.append(j)

        # recovers the (R,G,B) pixel information from the n-dimensional 
        # int vectors
        recovered_pixel_values = []
        for i in recovered_int_values: 
            recovered_pixel_values.append(decimal_to_rgb(int(i)))
        recovered_pixel_values = np.array(recovered_pixel_values)

        return recovered_pixel_values

    def recover_image(self): 

        recovered_pixel_values = self.recover_pixels()

        reshaped_pixel_values = recovered_pixel_values.reshape((self.height, self.width, 3))
        reconstructed_image = Image.fromarray(reshaped_pixel_values.astype(np.uint8), 'RGB')

        reconstructed_image.save(f'{self.save_image_path}{self.name}_reconstructed.png')

        return 



# ---------------------------------------------------- #


"""
HypersphereClass = Hypersphere(n = 8, image_path = "images/el_primo_square.jpg", verbose = True)

pixel_values, width, height = HypersphereClass.get_pixels()
statevectors, magnitudes = HypersphereClass.generate_statevectors()


Inverse_HypersphereClass = Inverse_Hypersphere(width = width, height = height, 
                                               statevectors = statevectors, 
                                               magnitudes = magnitudes, 
                                               name = "el_primo_rectangle", 
                                               save_image_path = "images/")
recovered_pixel_values = Inverse_HypersphereClass.recover_pixels()
print(len(recovered_pixel_values))
"""



    





