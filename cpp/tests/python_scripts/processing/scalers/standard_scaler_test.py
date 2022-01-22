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

    data = np.around(np.random.rand(10, 7), 6)
    print("Data:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in data))

    mean = np.mean(data, axis=0)
    print("Mean:")
    print(', '.join(str(format(n, '.6f')) for n in mean))

    standard_deviation = np.std(data, axis=0)
    print("Standard Deviation:")
    print(', '.join(str(format(n, '.6f')) for n in standard_deviation))

    variance = np.var(data, axis=0)
    print("Variance:")
    print(', '.join(str(format(n, '.6f')) for n in variance))

    scaled_data = (data - mean) / standard_deviation
    print("Scaled Data:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in scaled_data))