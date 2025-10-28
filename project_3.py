import numpy as np, matplotlib.pyplot as plt
'''
Module has Monte Carlo simulation class and ...
'''
class Monte_Carlo_simulations:
    '''
    Class for doing Monte Carlo simulations in a 3D space.
    ''' 
    def __init__(self, min_xyz, max_xyz):
        '''
        Initial values.

        Args:
        --------
            min_xyz (array-like): min boundries given in meters of the x-, y- and z-axis measured in nanometers [nm]
            max_xyz (array-like): max boundries given in meters of the x-, y- and z-axis measured in nanometers [nm]
        '''
        assert len(min_xyz) == 3, "Min boundry should have three dimensions"
        assert len(max_xyz) == 3, "Max boundry should have three dimensions"
        for i in range(3):
            assert isinstance(min_xyz[i], (float, int)) and isinstance(max_xyz[i], (float, int)), "Boundries must be floats or integers"
            # Taken from w3school
            min_xyz[i] = min_xyz[i]
            max_xyz[i] = max_xyz[i]
        
        self.min_xyz = np.array(min_xyz)
        self.max_xyz = np.array(max_xyz)


    def place_random_point(self):
        '''
        Places a random point within the simulation space from a uniform distribution.

        Returns:
        --------
            (array): a point in 3D-space (x, y, z)
        '''
        return np.random.uniform(self.min_xyz, self.max_xyz)
    

    def place_random_sphere(self):
        '''
        Creates a sphere and within the boundaries of our simulation space.

        Returns:
        --------
            center of sphere (array): the point in 3D-space (x, y, z) of which is the center of the sphere
            radius (float): radius of the sphere within the boundaries of the simulation space
        '''
        center_of_sphere = self.place_random_point()
        max_radius = np.min(np.append(
                                center_of_sphere - self.min_xyz, 
                                self.max_xyz - center_of_sphere))

        return center_of_sphere, np.random.uniform(1e-16, max_radius)
    

    def in_sphere(self, point, center, radius):
        '''
        Computes the length of a vector from a point to the center of the sphere and compare it to the radius of the sphere to check if a point is within a sphere.

        Args:
        --------
            point (array): the point (x, y, z) to check if it's inside the sphere 
            center (array): (x, y, z,)-coordinates for the center of the sphere
            radius (float): radius of the sphere

        Returns:
        --------
            (bool)
        '''
        return np.sqrt(np.sum((point-center)**2)) <= radius
    

    def calculate_fraction_of_points(self, n_points, center, radius):
        '''
        Creates a random sphere inside the simulation space. Creates n random points and check whether the point is inside the sphere or not.

        Args:
        --------
            n_points (int): number of random points
            center (array): (x, y, z)-coordinate of center of sphere
            radius (float): radius of sphere

        Returns:
        --------
            points_inside (list): (x, y, z)-coordinates of all points inside the sphere
            (float): fraction of points inside
        '''
        points_inside = []
        for _ in range(n_points):
            point = self.place_random_point()
            if self.in_sphere(point, center, radius):
                points_inside.append(point)

        return np.array(points_inside), len(points_inside)/n_points


    def calculate_accuracy(self, n_points, fraction_of_hits):
        '''
        Calculates the accuracy of our Monte Carlo simulation in regards to the analytical expression.

        Args:
        --------
            n_points (int): number of random points
            fraction_of_hits (float): fraction of points inside
            radius (float): radius of the sphere
        
        Returns:
        --------
            est_volume (float): estimated volume of sphere
            std_err (float): standard error
            true_volume (float): analytical volume of the sphere
        '''
        volume_of_simulation_area = (self.max_xyz[0]-self.min_xyz[0]) * (self.max_xyz[1]-self.min_xyz[1]) * (self.max_xyz[2]-self.min_xyz[2]) 
        est_volume = volume_of_simulation_area * fraction_of_hits
        std_err = volume_of_simulation_area * np.sqrt((fraction_of_hits * (1-fraction_of_hits))/n_points)
        return est_volume, std_err
    

    def calculate_analytical(self, radius):
        '''
        Calculates the analytical expression of volume for a sphere.

        Args:
        -------
            radius (float): radius of the sphere

        Return:
            (float): volume of the sphere
        '''
        return 4/3 * np.pi * radius**3
    

    def plot_3d(self, points_inside):
        '''
        Plots 3D scatter-plot of a matrix.

        Args:
        --------
            points_inside (2Darray): (x, y, z)-coordinates of points inside sphere
        '''
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(points_inside[: ,0], points_inside[: ,1], points_inside[: ,2], s=3)
        ax.set_xlim(self.min_xyz[0], self.max_xyz[0])
        ax.set_ylim(self.min_xyz[1], self.max_xyz[1])
        ax.set_zlim(self.min_xyz[2], self.max_xyz[2])


    def estimate_pi(self, n_points, points_inside, radius):
        '''
        Estimates pi as a function of number of randomly generated points.

        Args:
        -------
            n_points (int): number of random points
            points_inside (int): number of hits inside the sphere

        Returns:
        -------
            (float): the estimation of pi
        '''
        volume_of_simulation_area = (self.max_xyz[0]-self.min_xyz[0]) * (self.max_xyz[1]-self.min_xyz[1]) * (self.max_xyz[2]-self.min_xyz[2])

        return (3 * volume_of_simulation_area * (points_inside/n_points)) / (4 * radius**3)
    

    def translate_periodic_table(self, atom):
        '''
        Reads the periodic table and returns the atomic radius (in nanometer [nm]) of the given atom.

        Args:
        --------
            atom (str): atomic symbol

        Returns:
        --------
            (float): atmoic radius [nm]
        '''
        p_table = {
            "H": 0.12,
            "O": 0.152,
            "P": 0.18,
            "C": 0.17,
            "N": 0.155
        }
        return p_table[atom]
        
    

if __name__ == "__main__":
    test = Monte_Carlo_simulations([-3, -3, -3], [3, 3, 3])
    assert test, "Object was not created"
    assert len(test.place_random_point()) == 3, "Random point was not three dimensions"
    test_sphere_center, test_sphere_radius = np.array((0, 0, 0)), 1.5
    test_point_true, test_point_false = np.array((0, 1, 1)), np.array((2.5, 2.5, 2.5))
    assert test.in_sphere(test_point_true, test_sphere_center, test_sphere_radius), "Point is not inside the test sphere"
    assert not test.in_sphere(test_point_false, test_sphere_center, test_sphere_radius), "Point is inside the test sphere"