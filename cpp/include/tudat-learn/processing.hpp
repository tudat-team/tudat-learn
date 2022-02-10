/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_PROCESSING_HPP
#define TUDAT_LEARN_PROCESSING_HPP

namespace tudat_learn
{

/**
 * @brief Base Processing class. \n 
 * Currently has no functionality, but serves as the base class for classes with processing purposes, such as samplers.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @tparam Label_t The type of a single label. Like the Datum_t, can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t>
class Processing {
  protected:
    /**
     * @brief Protected default constructor, ensuring it is only called from classes that derive from it.
     * 
     */
    Processing( ) = default;

  public:
    /**
     * @brief Virtual destructor, as the class will likely have virtual methods.
     * 
     */
    virtual ~Processing( ) { }
};
  
} // namespace tudat_learn


#endif // TUDAT_LEARN_PROCESSING_HPP
