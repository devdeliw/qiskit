from PIL import Image
from scipy.spatial import cKDTree 
from scipy.interpolate import griddata, Rbf

import numpy as np 
import math 
import matplotlib.pyplot as plt


class Sphere: 

    """
    This class involves generating an RGB gradient sphere that consists
    of the color-space of a 2D digital image. 

    --------
    Methods:
    --------

    deLef get_pixels(): 

        Gets the RGB information for each pixel from an `n x n` image 
        and stores then in an array

    def fibonacci(): 

        Evenly distributes `n x n` points along a unit sphere using 
        the fibonacci method. 

        Each point is colored with the same information as each pixel 
        of the image. 

        The pixels are ordered from top-left to bottom-right, going down 
        with rows. 

    def generate_sphere(): 

        Converts the unit sphere from self.fibonacci() into a spherical 
        gradient by diffusing each pixel using np.linspace(). 

    """

    def __init__(self, image_path, show_sphere = False, verbose = False): 

        """
        Parameters: 
        ----------- 

        image_path  : String
        file path to 2D image 

        show_sphere : Bool 
        displays the fibonacci sphere and the gradient sphere

        verbose     : Bool
        if you want additional information

        """

        self.image_path = image_path
        self.show_sphere = show_sphere
        self.verbose = verbose

    def get_pixels(self): 

        # load image
        im = Image.open(self.image_path, mode = 'r')
        image = im.convert("RGB")

        # number of pixels & RGB information in [0, 1] rather than [0, 255]
        n_pixels = np.array(image).shape[0]**2
        pixel_values = np.array(list(image.getdata()))/255

        return n_pixels, pixel_values

    def fibonacci(self, show_sphere = False): 

        n_pixels, pixel_values = self.get_pixels()
        
        # golden angle
        phi = math.pi * (math.sqrt(5) - 1)

        indices = np.arange(0, n_pixels)

        ## y goes from 1 to -1 ##
        y = 1 - (indices / float(n_pixels - 1)) * 2
        radius = np.sqrt(1 - y * y) 

        theta = phi * indices
        x = np.cos(theta) * radius 
        z = np.sin(theta) * radius

        points = np.column_stack((x, y, z))

        if show_sphere:
            ax = plt.figure().add_subplot(projection = '3d')
            ax.scatter(points[:, 0], points[:, 1], 
                       points[:, 2], c = pixel_values, 
                           marker = '+'
                )
            
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')
            
            plt.show()

        if self.verbose: 
            print(f"{len(points)} total pixels")

        return points, pixel_values 

    def angles(self): 

        points, pixel_values = self.fibonacci()

        spherical_coords = []

        for x, y, z in points:
            theta = np.arccos(z)  
            phi = np.arctan2(y, x) 
            spherical_coords.append([theta, phi])

        return spherical_coords

    def generate_sphere(self): 

        points, pixel_values = self.fibonacci(show_sphere = False)

        coords = points
        colors = pixel_values

        # rbf interpolation for each color channel
        rbf_r = Rbf(coords[:, 0], coords[:, 1], coords[:, 2], colors[:, 0], function='linear')
        rbf_g = Rbf(coords[:, 0], coords[:, 1], coords[:, 2], colors[:, 1], function='linear')
        rbf_b = Rbf(coords[:, 0], coords[:, 1], coords[:, 2], colors[:, 2], function='linear')

        # mesh grid
        phi, theta = np.mgrid[0.0:np.pi:200j, 0.0:2.0*np.pi:200j]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # interpolate colors
        interpolated_colors = np.zeros((200, 200, 3))
        interpolated_colors[:, :, 0] = rbf_r(x, y, z)
        interpolated_colors[:, :, 1] = rbf_g(x, y, z)
        interpolated_colors[:, :, 2] = rbf_b(x, y, z)

        interpolated_colors = np.clip(interpolated_colors, 0, 1)

        # plot the spherical gradient
        if self.show_sphere:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, z, facecolors=interpolated_colors, rstride=1, cstride=1, antialiased=True)

            plt .show()

        return

