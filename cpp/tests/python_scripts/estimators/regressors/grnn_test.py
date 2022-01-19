# /*    Copyright (c) 2010-2018, Delft University of Technology
#  *    All rights reserved
#  *
#  *    This file is part of the Tudat. Redistribution and use in source and
#  *    binary forms, with or without modification, are permitted exclusively
#  *    under the terms of the Modified BSD license. You should have received
#  *    a copy of the license with this file. If not, please or visit:
#  *    http://tudat.tudelft.nl/LICENSE.
#  */

import numpy as np

if __name__ == '__main__':
  np.random.seed(0)

  center_points = np.array([
    [0.548814, 0.715189, 0.602763, 0.544883, 0.423655, 0.645894, 0.437587],
    [0.891773, 0.963663, 0.383442, 0.791725, 0.528895, 0.568045, 0.925597],
    [0.071036, 0.087129, 0.020218, 0.832620, 0.778157, 0.870012, 0.978618],
    [0.799159, 0.461479, 0.780529, 0.118274, 0.639921, 0.143353, 0.944669],
    [0.521848, 0.414662, 0.264556, 0.774234, 0.456150, 0.568434, 0.018790],
    [0.617635, 0.612096, 0.616934, 0.943748, 0.681820, 0.359508, 0.437032],
    [0.697631, 0.060225, 0.666767, 0.670638, 0.210383, 0.128926, 0.315428],
    [0.363711, 0.570197, 0.438602, 0.988374, 0.102045, 0.208877, 0.161310],
    [0.653108, 0.253292, 0.466311, 0.244426, 0.158970, 0.110375, 0.656330],
    [0.138183, 0.196582, 0.368725, 0.820993, 0.097101, 0.837945, 0.096098]
  ])

  labels = np.array([
    [0.976459, 0.468651],
    [0.976761, 0.604846],
    [0.739264, 0.039188],
    [0.282807, 0.120197],
    [0.296140, 0.118728],
    [0.317983, 0.414263],
    [0.064147, 0.692472],
    [0.566601, 0.265389],
    [0.523248, 0.093941],
    [0.575946, 0.929296]
  ])

  sigma = 0.318569

  inputs = np.array([
    [0.667410, 0.131798, 0.716327, 0.289406, 0.183191, 0.586513, 0.020108],
    [0.828940, 0.004695, 0.677817, 0.270008, 0.735194, 0.962189, 0.248753],
    [0.576157, 0.592042, 0.572252, 0.223082, 0.952749, 0.447125, 0.846409]
  ])

  input_distance_matrix = np.sqrt(np.sum((inputs[:, np.newaxis, :] - center_points[np.newaxis, :, :]) **2, axis=-1))
  print("Input Distance Matrix:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in input_distance_matrix))

  gaussian_distance_matrix = np.exp(- (input_distance_matrix / sigma)**2 )
  print("Gaussian Distance Matrix:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_distance_matrix))

  sum_of_distances = np.sum([gaussian_distance_matrix], axis=2)
  print("Sum of Distances:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in sum_of_distances))

  outputs = np.matmul(gaussian_distance_matrix, labels) / sum_of_distances.T
  print("Outputs:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in outputs))

  # IMPLEMENT OUTPUTS IN A DIFFERENT WAY!
  outputs_other = np.zeros((inputs.shape[0], labels.shape[1]))
  # for i in range(inputs.shape[0]):              # for every input
  #   for j in range(labels.shape[1]):            # for every output dimension
  #     for k in range(center_points.shape[0]):   # for every center point
  #       num = 0
  #       den = 0
  #       for l in range(center_points.shape[1]): # for every input dimension
  #         num += labels[i,j] * np.exp(-(input[i,l] - center_points[k,l])**2 / sigma**2)
  #         den +=               np.exp(-(input[i,l] - center_points[k,l])**2 / sigma**2)

  #       outputs_other[i, j] = num / den
  dist = np.zeros((inputs.shape[0], center_points.shape[0]))
  for i in range(inputs.shape[0]):              # for every input
    for j in range(labels.shape[1]):            # for every output dimension
      
      num = 0
      den = 0
      for k in range(center_points.shape[0]):   # for every center point
      
        for l in range(center_points.shape[1]): # for every input dimension
          dist[i,k] += (inputs[i,l] - center_points[k,l])**2

        dist[i,k] = np.exp(-dist[i,k] / sigma**2)
      
        num += labels[k, j] * dist[i, k]
        den +=                dist[i, k]
    
      outputs_other[i,j] = num / den
  
  print("Outputs nested for-loops:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in outputs_other))

  print("Distance nested for-loops:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in dist))