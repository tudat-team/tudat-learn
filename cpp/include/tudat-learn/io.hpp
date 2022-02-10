/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_IO_HPP
#define TUDAT_LEARN_IO_HPP

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace tudat_learn
{

// CSV implementation based on/taken from https://stackoverflow.com/a/1120224/17210123
// struct CSVRow {
//   std::string_view operator[](std::size_t index) const {
//       return std::string_view(&m_line[m_data[index] + 1], m_data[index + 1] -  (m_data[index] + 1));
//   }

//   std::size_t size() const {
//       return m_data.size() - 1;
//   }

//   void read_next_row(std::istream &str);

//   private:
//     std::string       m_line;
//     std::vector<int>  m_data;
// };

// std::istream& operator>>(std::istream& str, CSVRow& data)
// {
//     data.read_next_row(str);
//     return str;
// } 

} // namespace tudat_learn


#endif // TUDAT_LEARN_IO_HPP
