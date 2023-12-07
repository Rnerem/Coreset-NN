
import numpy as np
from scipy.spatial import ConvexHull as ConvexHull
import dataset
import matplotlib.pyplot as plt 
import ipdb
import random

load_data = dataset.load_hdf5_data

def uniform_points(n, d = 2):
         P = [[ np.random.rand() for i in range(d)] for j in range(n)]
         P = np.array(P)
         return P

def concatenate_CH_col(P):
    hull = ConvexHull(P)
    vert = hull.vertices
    indicator = []
    for i in range(len(P)):
           indicator += [[int(i in vert)]]
    indicator = np.array(indicator)
    return np.concatenate((P, indicator), axis = 1)

def generate_data(num_sets, num_points, dimension =2):
       data = [  concatenate_CH_col(uniform_points(num_points, dimension)) for i in range(num_sets)]
       data = np.array(data)
       np.save(f'for_sam/{num_points}_uniform_points', data )

def generate_data_range(num_sets, low_num, high_num, dimension =2):

       data = [  concatenate_CH_col(uniform_points(random.randrange(low_num,high_num,1), dimension)) for i in range(num_sets)]
       data = np.array(data, dtype = object)
       np.save(f'for_sam/{low_num}_to_{high_num}_uniform_points', data )

def concatenate_h5(filename):
       A = load_data(filename)
       data = [concatenate_CH_col(A[0][i]) for i in range(len(A[0]))]
       data = np.array(data)
       np.save('for_sam/ply_data_train0_CH', data )


def plot_h5(filename):
       A = np.array(load_data(filename)[0])
       ipdb.set_trace()
       for i in range(len(A[0])):
              fig = plt.figure(figsize=(12, 12))
              ax = fig.add_subplot(projection='3d')
              ax.scatter(A[i,:,0], A[i,:,1], A[i,:,2])
              plt.show()    

# concatenate_h5('data/modelnet40_ply_hdf5_2048/ply_data_train0.h5') 
#generate_data(3000,50,2)
generate_data_range(3000, 10, 100, dimension =2)




