import numpy as np

class K_means(object):   
    def __init__(self, Points, k):        
        self.points = Points
        self.k = k
        self.centroids = self.initialize_centroids(Points, k)

    def initialize_centroids(self, points, k):
        '''
        Selects k random points as initial
        points from dataset
        '''
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[:k]    


    def closest_centroid(self, points, centroids):
        '''
        Returns an array containing the index to the nearest centroid for each point
        '''
        dists = np.sqrt((points - centroids[:,np.newaxis])**2).sum(axis=2)
        return np.argmin(dists, axis = 0)


    def move_centroids(self, points, closest, centroids):
        '''
        Returns the new centroids assigned from the points closest to them
        '''
        return np.array([points[closest==self.k].mean(axis=0) for self.k in range(centroids.shape[0])])


    def train(self, num_iterations):
        # Run iterative process
        for i in range(num_iterations):
            closest = self.closest_centroid(self.points, self.centroids)
            self.centroids = self.move_centroids(self.points, closest, self.centroids)
        return self.centroids
