/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include "tudat-learn/io.hpp"

namespace tudat_learn
{
  
// // CSV implementation based on/taken from https://stackoverflow.com/a/1120224/17210123
// void CSVRow::read_next_row(std::istream &str) {
//   std::getline(str, m_line);

//   m_data.clear();
//   m_data.emplace_back(-1);
//   std::string::size_type pos = 0;
//   while((pos = m_line.find(',', pos)) != std::string::npos)
//   {
//       m_data.emplace_back(pos);
//       ++pos;
//   }
//   // This checks for a trailing comma with no data after it.
//   pos   = m_line.size();
//   m_data.emplace_back(pos);
// }

} // namespace tudat_learn