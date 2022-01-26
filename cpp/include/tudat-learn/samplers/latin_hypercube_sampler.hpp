/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP
#define TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP

#include "tudat-learn/sampling.hpp"

namespace tudat_learn
{

template <typename Datum_t>
class LatinHypercubeSampler : public Sampler<Datum_t> {

};

} // namespace tudat_learn

#include "tudat-learn/samplers/latin_hypercube_sampler.tpp"

#endif // TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP