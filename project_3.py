'''
Module has Monte Carlo simulation-class and accessible volume-class.
'''
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloSimulations:
    '''
    Class for doing Monte Carlo simulations in a 3D space.
    '''
    def __init__(self, min_xyz, max_xyz):
        '''
        Initial values.

        Args:
        --------
            min_xyz (array-like): min boundries of the (x, y, z)-axis measured in Ångstrøm [Å]
            max_xyz (array-like): max boundries of the (x, y, z)-axis measured in Ångstrøm [Å]
        '''
        assert len(min_xyz) == 3, "Min boundry should have three dimensions"
        assert len(max_xyz) == 3, "Max boundry should have three dimensions"
        for i in range(3):
            assert isinstance(min_xyz[i], (float, int)) and isinstance(max_xyz[i], (float, int)),'''
            Boundries must be floats or integers'''
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
            center_of_sphere (array): the point (center of sphere) in 3D-space (x, y, z)
            radius (float): radius of the sphere within the boundaries of the simulation space
        '''
        center_of_sphere = self.place_random_point()
        max_radius = np.min(np.append(
                                center_of_sphere - self.min_xyz,
                                self.max_xyz - center_of_sphere))

        return center_of_sphere, np.random.uniform(0, max_radius)


    def in_sphere(self, point, center, radius):
        '''
        Computes the length of a vector from a point to the center of the sphere and
        compare it to the radius of the sphere to check if a point is within a sphere.

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
        Creates a random sphere inside the simulation space. 
        Creates n random points and check whether the point is inside the sphere or not.

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
        Calculates the accuracy of our Monte Carlo simulation in regards to the analytical 
        expression.

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
        min_xyz, max_xyz = self.min_xyz, self.max_xyz
        vol_sim_area = (max_xyz[0]-min_xyz[0]) * (max_xyz[1]-min_xyz[1]) * (max_xyz[2]-min_xyz[2])
        est_volume = vol_sim_area * fraction_of_hits
        std_err = vol_sim_area * np.sqrt((fraction_of_hits * (1-fraction_of_hits))/n_points)
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
        min_xyz, max_xyz = self.min_xyz, self.max_xyz
        vol_sim_area = (max_xyz[0]-min_xyz[0]) * (max_xyz[1]-min_xyz[1]) * (max_xyz[2]-min_xyz[2])

        return (3 * vol_sim_area * (points_inside/n_points)) / (4 * radius**3)


    def translate_periodic_table(self, atom):
        '''
        Reads the periodic table and returns the atomic radius (in Ångstrøm [Å]) of the given atom.

        Args:
        --------
            atom (str): atomic symbol

        Returns:
        --------
            (float): atmoic radius [Å]
        '''
        p_table = {
            "H": 1.2,
            "O": 1.52,
            "P": 1.8,
            "C": 1.7,
            "N": 1.55
        }
        return p_table[atom]


    def change_boundaries(self, dna):
        '''
        Changes boundaries of the simulation box in terms of a DNA-string.

        Args:
        -------
            dna (list): a list of all atoms in a DNA sequence, with it's respective radius
        '''
        x_coord, y_coord, z_coord = [], [], []
        for atom in dna:
            x_coord.append(atom[0][0])
            y_coord.append(atom[0][1])
            z_coord.append(atom[0][2])

        x_min = [np.min(x_coord), dna[np.argmin(x_coord)][1]]
        y_min = [np.min(y_coord), dna[np.argmin(y_coord)][1]]
        z_min = [np.min(z_coord), dna[np.argmin(z_coord)][1]]
        x_max = [np.max(x_coord), dna[np.argmax(x_coord)][1]]
        y_max = [np.max(y_coord), dna[np.argmax(y_coord)][1]]
        z_max = [np.max(z_coord), dna[np.argmax(z_coord)][1]]
        buffer = 5

        self.min_xyz = np.array((x_min[0]-x_min[1]-buffer, y_min[0]-y_min[1]-buffer, z_min[0]-z_min[1]-buffer))
        self.max_xyz = np.array((x_max[0]+x_max[1]+buffer, y_max[0]+y_max[1]+buffer, z_max[0]+z_max[1]+buffer))


    def random_walker(self, n_steps, n_walkers):
        '''
        Creates a set of random walkers that each starts from a random point.

        Args:
        --------
            n_steps (int): number of steps the walkers will take
            n_walkers (int): number of walkers

        Returns:
        --------
            set_of_walkers (list): list of coordinates that each walker has traveled
        '''
        set_of_walkers = []
        for _ in range(n_walkers):
            start_point = self.place_random_point()
            x, y, z = [start_point[0]], [start_point[1]], [start_point[2]]
            for _ in range(n_steps):
                x.append(x[-1] + 2*np.random.randint(0,2)-1)
                y.append(y[-1] + 2*np.random.randint(0,2)-1)
                z.append(z[-1] + 2*np.random.randint(0,2)-1)
            set_of_walkers.append(np.array((x, y, z)))
        return set_of_walkers


    def fast_random_walker(self, n_steps, n_walkers):
        '''
        Creates a set of random walkers that each starts from a random point that
        runs faster with less iterations.

        Args:
        --------
            n_steps (int): number of steps the walkers will take
            n_walkers (int): number of walkers

        Returns:
        --------
            set_of_walkers (list): list of coordinates that each walker has traveled
        '''
        set_of_walkers = []
        for _ in range(n_walkers):
            start_point = self.place_random_point()
            x = start_point[0] + np.cumsum(2*np.random.randint(0, 2, size=n_steps)-1)
            y = start_point[1] + np.cumsum(2*np.random.randint(0, 2, size=n_steps)-1)
            z = start_point[2] + np.cumsum(2*np.random.randint(0, 2, size=n_steps)-1)
            set_of_walkers.append(np.array((x, y, z)))
        return set_of_walkers


class AccessibleVolume:
    '''
    A class for calculating the accessible volume around a DNA in a 3D space.
    '''
    def __init__(self, dna, radius_of_point):
        '''
        Initial values.

        Args:
        --------
            dna (list): 1. element is the (x, y, z)-coordinates of each atom in the DNA and 2. element is the radius in Ångstrøm [Å]
            radius_of_point (float): radius of the point of the walker in Ångstrøm [Å]
        '''
        self.dna = dna
        self.radius_of_point = radius_of_point

        x_coord, y_coord, z_coord = [], [], []
        for atom in dna:
            x_coord.append(atom[0][0])
            y_coord.append(atom[0][1])
            z_coord.append(atom[0][2])

        x_min = [np.min(x_coord), dna[np.argmin(x_coord)][1]]
        y_min = [np.min(y_coord), dna[np.argmin(y_coord)][1]]
        z_min = [np.min(z_coord), dna[np.argmin(z_coord)][1]]
        x_max = [np.max(x_coord), dna[np.argmax(x_coord)][1]]
        y_max = [np.max(y_coord), dna[np.argmax(y_coord)][1]]
        z_max = [np.max(z_coord), dna[np.argmax(z_coord)][1]]
        buffer = 5

        self.min_xyz = np.array((x_min[0]-x_min[1]-buffer, y_min[0]-y_min[1]-buffer, z_min[0]-z_min[1]-buffer))
        self.max_xyz = np.array((x_max[0]+x_max[1]+buffer, y_max[0]+y_max[1]+buffer, z_max[0]+z_max[1]+buffer))


    def random_walker_with_no_collison(self, init_xyz, n_steps):
        '''
        Random walker with a set initial position that also checks that the points are 
        collision-free.

        Args:
        --------
            init_xyz (array): (x, y, z)-coordinates of the starting point of the walker
            n_steps (int): number of steps

        Returns:
        --------
            x (list): x-coordinates of the path of the walker
            y (list): y-coordinates of the path of the walker
            z (list): z-coordinates of the path of the walker
        '''
        x, y, z = [init_xyz[0]], [init_xyz[1]], [init_xyz[2]]
        path = []
        for _ in range(n_steps):
            x_step = x[-1] + 2*np.random.randint(0,2)-1
            y_step = y[-1] + 2*np.random.randint(0,2)-1
            z_step = z[-1] + 2*np.random.randint(0,2)-1
            point = self.pbc(np.array((x_step, y_step, z_step)))
            if not self.collision(point):
                x.append(point[0])
                y.append(point[1])
                z.append(point[2])
                path.append(np.array((point[0], point[1], point[2])))
        return np.array(path)


    def collision(self, point):
        '''
        Checks whether a point collides with any of the atoms initialized in the class.

        Args:
        --------
            point (list): (x, y, z)-coordinates for a point

        Return:
        --------
            (bool)
        '''
        for atom in self.dna:
            distance_vector = np.sqrt(np.sum((point-atom[0])**2))
            if self.radius_of_point + atom[1] > distance_vector:
                return True
        return False


    def pbc(self, point):
        '''
        Function to prevent a walker to leave the simulation box.

        Args:
        --------
            point (list): (x, y, z)-coordinates for a point

        Return:
        ---------
            point (list): (x, y, z)-coordinates for a point
        '''
        if point[0] < self.min_xyz[0]:
            point[0] = np.ceil(self.max_xyz[0])
        elif point[0] > self.max_xyz[0]:
            point[0] = np.floor(self.min_xyz[0])

        if point[1] < self.min_xyz[1]:
            point[1] = np.ceil(self.max_xyz[1])
        elif point[1] > self.max_xyz[1]:
            point[1] = np.floor(self.min_xyz[1])

        if point[2] < self.min_xyz[2]:
            point[2] = np.ceil(self.max_xyz[2])
        elif point[2] > self.max_xyz[2]:
            point[2] = np.floor(self.min_xyz[2])

        return point



if __name__ == "__main__":
    test = MonteCarloSimulations([-3, -3, -3], [3, 3, 3])
    assert test, "Object was not created"
    assert len(test.place_random_point()) == 3, "Random point was not three dimensions"
    test_sphere_center, test_sphere_radius = np.array((0, 0, 0)), 1.5
    test_point_true, test_point_false = np.array((0, 1, 1)), np.array((2.5, 2.5, 2.5))
    assert test.in_sphere(test_point_true, test_sphere_center, test_sphere_radius), "Point is not inside the test sphere"
    assert not test.in_sphere(test_point_false, test_sphere_center, test_sphere_radius), "Point is inside the test sphere"
