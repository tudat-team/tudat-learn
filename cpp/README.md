# tudat-learn

# Building and Installing

```bash
conda env create -f environment.yaml
conda activate tudat-learn
```

```bash
mkdir _build
cd _build
cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DCMAKE_INSTALL_PREFIX=../_install ..
cmake --build . --target install
```

# Example project:

(Hierarchy of project)

```cmake
cmake_minimum_required(VERSION 3.10)

project(main)

find_package(tudat-learn)

add_executable(main main.cpp)

target_link_libraries(main tudat-learn::tudat-learn)
```

Build with (on project's directory with the base `CMakeLists.txt` file):
```bash
mkdir _build
cd _build
cmake -DCMAKE_PREFIX_PATH="/path/to/tudat-learn/cpp/_install/lib/cmake/tudat-learn" ..
cmake --build . 
```

# Documentation


# Acknowledgements

Ideas for the library setup taken from:
- https://github.com/pablospe/cmake-example-library


Ideas for the Doxygen taken from:
- https://vicrucann.github.io/tutorials/quick-cmake-doxygen/