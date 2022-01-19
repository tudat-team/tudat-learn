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
    # precision = 10

    center_points = np.around(np.random.rand(10, 7), 6)
    print("Center Points:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in center_points))

    output = np.around(np.random.rand(10, 2), 6)
    print("Output:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in output))

    distance_matrix = np.sqrt(np.sum((center_points[:, np.newaxis, :] - center_points[np.newaxis, :, :]) **2, axis=-1))
    print("Distance Matrix:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in distance_matrix))



    # equivalent distance matrix
    # dm = np.zeros((10, 10))
    # for i in range(10):
    #     for j in range(10):
    #         dm[i, j] = np.linalg.norm(center_points[i,:] - center_points[j, :])

    # computing parameters (except sigma)

    cubic_distance_matrix = distance_matrix ** 3
    print("Cubic Distance Matrix:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in cubic_distance_matrix))

    sigma = np.around(np.random.rand(1), 6)
    print("Sigma is", str(format(sigma[0], '.6f')))

    gaussian_distance_matrix = np.exp(- (distance_matrix / sigma)**2 )
    print("Gaussian Distance Matrix:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_distance_matrix))

    cubic_coefficients = np.linalg.solve(cubic_distance_matrix, output)
    print("Cubic Coefficients:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in cubic_coefficients))

    gaussian_coefficients = np.linalg.solve(gaussian_distance_matrix, output)
    print("Gaussian Coefficients:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_coefficients))


    # generating random inputs to evaluate after fitting
    inputs = np.around(np.random.rand(3, 7), 6)
    print("Inputs:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in inputs))

    input_distance_matrix = np.sqrt(np.sum((inputs[:, np.newaxis, :] - center_points[np.newaxis, :, :]) **2, axis=-1))
    print("Input Distance Matrix:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in input_distance_matrix))

    gaussian_input_distance_matrix = np.exp(- (input_distance_matrix / sigma)**2 )
    print("Gaussian Input Distance Matrix:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_input_distance_matrix))

    cubic_input_distance_matrix = input_distance_matrix ** 3
    print("Cubic Input Distance Matrix:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in cubic_input_distance_matrix))

    expected_gaussian_output = np.matmul(gaussian_input_distance_matrix, gaussian_coefficients) # EXPRESSION?
    print("Expected Gaussian Output:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in expected_gaussian_output))

    expected_cubic_output = np.matmul(cubic_input_distance_matrix, cubic_coefficients) # EXPPRESSION?
    print("Expected Cubic Output:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in expected_cubic_output))


    print("Expected Gaussian Output:")
    print(',\n'.join((', '.join(np.format_float_scientific(nn, precision=6) for nn in n)) for n in expected_gaussian_output))

    # TO USE IN LATEX TABLES

    # cubic_distance_matrix = distance_matrix ** 3
    # print("Cubic Distance Matrix:")
    # print('\n'.join((' '.join(str(format(nn, '.6f')) for nn in n)) for n in cubic_distance_matrix))

    # gaussian_distance_matrix = np.exp(- (distance_matrix / sigma)**2 )
    # print("Gaussian Distance Matrix:")
    # print('\n'.join((' '.join('{:.6e}'.format(nn) for nn in n)) for n in gaussian_distance_matrix))

    # gaussian_distance_matrix = np.exp(- (distance_matrix / sigma)**2 )
    # print("Gaussian Distance Matrix:")
    # print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_distance_matrix))

    # cubic_coefficients = np.linalg.solve(cubic_distance_matrix, output)
    # print("Cubic Coefficients:")
    # print('\n'.join((' '.join(str(format(nn, '.6f')) for nn in n)) for n in cubic_coefficients))

    # gaussian_coefficients = np.linalg.solve(gaussian_distance_matrix, output)
    # print("Gaussian Coefficients:")
    # print('\n'.join((' '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_coefficients))


    # Testing RBFNPolynomial
    output_poly = np.concatenate((output, np.zeros((center_points.shape[1] + 1, output.shape[1]))))
    print("Output Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in output_poly))

    gaussian_matrix_poly = np.zeros((
        gaussian_distance_matrix.shape[0] + center_points.shape[1] + 1,
        gaussian_distance_matrix.shape[1] + center_points.shape[1] + 1
    ))
    gaussian_matrix_poly[    :gaussian_distance_matrix.shape[0],       :gaussian_distance_matrix.shape[1]  ] = gaussian_distance_matrix
    gaussian_matrix_poly[    :gaussian_distance_matrix.shape[0],        gaussian_distance_matrix.shape[1]  ] = np.ones((gaussian_distance_matrix.shape[0]))
    gaussian_matrix_poly[     gaussian_distance_matrix.shape[0],       :gaussian_distance_matrix.shape[1]  ] = np.ones((gaussian_distance_matrix.shape[1]))
    gaussian_matrix_poly[    :gaussian_distance_matrix.shape[0],   (1 + gaussian_distance_matrix.shape[1]):] = center_points
    gaussian_matrix_poly[(1 + gaussian_distance_matrix.shape[0]):,     :gaussian_distance_matrix.shape[1]  ] = center_points.T
    print("Gaussian Distance Matrix Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_matrix_poly))

    gaussian_coefficients_poly = np.linalg.solve(gaussian_matrix_poly, output_poly)
    print("Gaussian Coefficients Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_coefficients_poly))


    cubic_matrix_poly = np.zeros((
        cubic_distance_matrix.shape[0] + center_points.shape[1] + 1,
        cubic_distance_matrix.shape[1] + center_points.shape[1] + 1
    ))
    cubic_matrix_poly[    :cubic_distance_matrix.shape[0],       :cubic_distance_matrix.shape[1]  ] = cubic_distance_matrix
    cubic_matrix_poly[    :cubic_distance_matrix.shape[0],        cubic_distance_matrix.shape[1]  ] = np.ones((cubic_distance_matrix.shape[0]))
    cubic_matrix_poly[     cubic_distance_matrix.shape[0],       :cubic_distance_matrix.shape[1]  ] = np.ones((cubic_distance_matrix.shape[1]))
    cubic_matrix_poly[    :cubic_distance_matrix.shape[0],   (1 + cubic_distance_matrix.shape[1]):] = center_points
    cubic_matrix_poly[(1 + cubic_distance_matrix.shape[0]):,     :cubic_distance_matrix.shape[1]  ] = center_points.T
    print("Cubic Distance Matrix Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in cubic_matrix_poly))

    cubic_coefficients_poly = np.linalg.solve(cubic_matrix_poly, output_poly)
    print("Cubic Coefficients Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in cubic_coefficients_poly))


    # inputs
    gaussian_input_distance_matrix_poly = np.ones((gaussian_input_distance_matrix.shape[0], gaussian_input_distance_matrix.shape[1] + 1 + inputs.shape[1]))
    gaussian_input_distance_matrix_poly[:gaussian_input_distance_matrix.shape[0], :gaussian_input_distance_matrix.shape[1]] = gaussian_input_distance_matrix
    gaussian_input_distance_matrix_poly[                                       :, (gaussian_input_distance_matrix.shape[1] + 1):] = inputs 
    print("Gaussian Input Distance Matrix Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_input_distance_matrix_poly))

    expected_gaussian_output_poly = np.matmul(gaussian_input_distance_matrix_poly, gaussian_coefficients_poly)
    print("Expected Gaussian Output Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in expected_gaussian_output_poly))

    cubic_input_distance_matrix_poly = np.ones((cubic_input_distance_matrix.shape[0], cubic_input_distance_matrix.shape[1] + 1 + inputs.shape[1]))
    cubic_input_distance_matrix_poly[:cubic_input_distance_matrix.shape[0], :cubic_input_distance_matrix.shape[1]] = cubic_input_distance_matrix
    cubic_input_distance_matrix_poly[                                       :, (cubic_input_distance_matrix.shape[1] + 1):] = inputs 
    print("Cubic Input Distance Matrix Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in cubic_input_distance_matrix_poly))

    expected_cubic_output_poly = np.matmul(cubic_input_distance_matrix_poly, cubic_coefficients_poly)
    print("Expected Cubic Output Polynomial:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in expected_cubic_output_poly))