# `tudat-learn`

# Building and Installing

Creating an [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) environment with the required dependencies:

```bash
conda env create -f environment.yaml
conda activate tudat-learn
```


Creating the `_build` directory, where `tudat-learn` will be built and selecting an `_install` directory, in which it will be installed.
```bash
mkdir _build
cd _build
cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCMAKE_INSTALL_PREFIX=../_install ..
cmake --build . --target install
```

To build documentation, set the `BUILD_DOC` flag as `ON` when running `cmake`:
```bash
cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCMAKE_INSTALL_PREFIX=../_install -DBUILD_DOC=ON ..
```

# Example project:

Hierarchy of a very simple project, in which `main.cpp` uses `tudat-learn`:
```
project
├── CMakeLists.txt
└── main.cpp
```

Having the `CMakeLists.txt` file containing the following:
```cmake
cmake_minimum_required(VERSION 3.10)

project(main)

find_package(tudat-learn)

add_executable(main main.cpp)

target_link_libraries(main tudat-learn::tudat-learn)
```

And the `main.cpp` file:
```cpp
#include <iostream>
#include <utility>

#include <tudat-learn/random.hpp>
#include <tudat-learn/samplers/random_samplers/latin_hypercube_sampler.hpp>

int main() {
  tudat_learn::Random::set_seed(42);

  tudat_learn::LatinHypercubeSampler<double> sampler(std::make_pair(1.0,2.0), 3);

  std::cout << "Hello, World!" << std::endl;
  for(const auto &sample: sampler.sample())
    std::cout << sample << ", ";
  std::cout << "\n";

  return 0;
}
```


The project can be built with the commands below. Notice how one needs to provide the path to the directory with `tudat-learn`'s `.cmake` files within its `_install` directory.
```bash
cd project
mkdir _build
cd _build
cmake -DCMAKE_PREFIX_PATH="/path/to/tudat-learn/cpp/_install/lib/cmake/tudat-learn" ..
cmake --build . 
```

Results in the following output:
```bash
$ ./_build/main
Hello, World!
1.65024, 1.244, 1.86622, 
```

# Documentation

To view the Doxygen documentation, after it was built, one can open the `./_build/doc_doxygen/html/index.html` file with any modern browser.

# Acknowledgements

Ideas for the library setup taken from:
- https://github.com/pablospe/cmake-example-library


Ideas for the Doxygen taken from:
- https://vicrucann.github.io/tutorials/quick-cmake-doxygen/