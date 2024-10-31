#include <pybind11/pybind11.h>

#include "main.h"
#include <complex>
#include <cstdio>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// -------------
// pure C++ code
// -------------

int get_num_threads() {
    py::gil_scoped_acquire acquire;
    int n = 0;
#pragma omp parallel
    { n = omp_get_num_threads(); }
    return n;
}

int return1() { return 1; }

double lerp(double frac, double val1, double val2) { return (val1 + ((val2 - val1) * frac)); }

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

typedef py::array_t<double> t_array;

PYBIND11_MODULE(_blocks, m) {
    m.doc() = R"pbdoc(
        module description
        -----------------------

        .. currentmodule:: _blocks

        .. autosummary::
           :toctree: _generate

           blerp
    )pbdoc";

    m.def("test_omp",
          []() {
              py::gil_scoped_release release;
              return get_num_threads();
          },
          "Number of OMP threads");
    m.def("return1", &return1, "testfunc");
    m.def("bin_volume", &bin_volume<float>, "bin_volume");
    m.def("bin_tensor", &bin_tensor<float>, "bin_tensor");
    m.def("upscale_volume", &upscale_volume<float>, "upscale_volume");
    m.def("upscale_tensor", &upscale_tensor<float>, "upscale_tensor");

    // m.def("pb_divide", &pb_divide, "divide limits");
    // m.def("pb_combine_volume", &pb_combine_volume<float>, "pb_combine_volume");

// broke in vs 2022 for some reason
// #ifdef VERSION_INFO
//     m.attr("__version__") = VERSION_INFO;
// #else
//     m.attr("__version__") = "dev";
// #endif
}
