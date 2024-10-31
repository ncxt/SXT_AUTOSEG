#include <cmath>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

template <typename T>
py::array_t<T> bin_volume(py::array_t<T, py::array::c_style | py::array::forcecast> volume,
                          int binning) {
    auto data = volume.unchecked<3>();
    int nx = static_cast<int>(data.shape(0));
    int ny = static_cast<int>(data.shape(1));
    int nz = static_cast<int>(data.shape(2));

    int nx_binned = nx / binning;
    int ny_binned = ny / binning;
    int nz_binned = nz / binning;

    py::array_t<T> retval({nx_binned, ny_binned, nz_binned});
    std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
    auto data_out = retval.mutable_unchecked<3>();

    double scale = 1.0 / (binning * binning * binning);
#pragma omp parallel for
    for (int ib = 0; ib < nx_binned; ib++) {
        for (int di = 0; di < binning; di++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int i = binning * ib + di;
                    int jb = j / binning;
                    int kb = k / binning;
                    data_out(ib, jb, kb) += data(i, j, k);
                }
            }
        }
    }
    std::transform(retval.mutable_data(), retval.mutable_data() + retval.size(),
                   retval.mutable_data(), [scale](auto &c) { return c * scale; });
    return retval;
}

template <typename T>
py::array_t<T> bin_tensor(py::array_t<T, py::array::c_style | py::array::forcecast> tensor,
                          int binning) {
    auto data = tensor.unchecked<4>();
    int channels = static_cast<int>(data.shape(0));
    int nx = static_cast<int>(data.shape(1));
    int ny = static_cast<int>(data.shape(2));
    int nz = static_cast<int>(data.shape(3));

    int nx_binned = nx / binning;
    int ny_binned = ny / binning;
    int nz_binned = nz / binning;

    py::array_t<T> retval({channels, nx_binned, ny_binned, nz_binned});
    std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
    auto data_out = retval.mutable_unchecked<4>();

    double scale = 1.0 / (binning * binning * binning);
#pragma omp parallel for
    for (int ch = 0; ch < channels; ch++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    int ib = i / binning;
                    int jb = j / binning;
                    int kb = k / binning;
                    data_out(ch, ib, jb, kb) += data(ch, i, j, k);
                }
            }
        }
    }
    std::transform(retval.mutable_data(), retval.mutable_data() + retval.size(),
                   retval.mutable_data(), [scale](auto &c) { return c * scale; });
    return retval;
}

template <typename T>
py::array_t<T> upscale_volume(py::array_t<T, py::array::c_style | py::array::forcecast> volume,
                              int binning) {
    auto data = volume.unchecked<3>();
    int nx = static_cast<int>(data.shape(0));
    int ny = static_cast<int>(data.shape(1));
    int nz = static_cast<int>(data.shape(2));

    int nx_scaled = nx * binning;
    int ny_scaled = ny * binning;
    int nz_scaled = nz * binning;

    py::array_t<T> retval;
    retval.resize({nx_scaled, ny_scaled, nz_scaled});
    std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
    auto data_out = retval.mutable_unchecked<3>();

    for (int is = 0; is < nx_scaled; is++) {
        for (int js = 0; js < ny_scaled; js++) {
            for (int ks = 0; ks < nz_scaled; ks++) {
                int i = is / binning;
                int j = js / binning;
                int k = ks / binning;
                data_out(is, js, ks) += data(i, j, k);
            }
        }
    }
    return retval;
}

template <typename T>
py::array_t<T> upscale_tensor(py::array_t<T, py::array::c_style | py::array::forcecast> tensor,
                              int binning) {
    auto data = tensor.unchecked<4>();
    int channels = static_cast<int>(data.shape(0));
    int nx = static_cast<int>(data.shape(1));
    int ny = static_cast<int>(data.shape(2));
    int nz = static_cast<int>(data.shape(3));

    int nx_scaled = nx * binning;
    int ny_scaled = ny * binning;
    int nz_scaled = nz * binning;

    py::array_t<T> retval;
    retval.resize({channels, nx_scaled, ny_scaled, nz_scaled});
    std::fill(retval.mutable_data(), retval.mutable_data() + retval.size(), 0.);
    auto data_out = retval.mutable_unchecked<4>();

    for (int ch = 0; ch < channels; ch++) {
        for (int is = 0; is < nx_scaled; is++) {
            for (int js = 0; js < ny_scaled; js++) {
                for (int ks = 0; ks < nz_scaled; ks++) {
                    int i = is / binning;
                    int j = js / binning;
                    int k = ks / binning;
                    data_out(ch, is, js, ks) += data(ch, i, j, k);
                }
            }
        }
    }
    return retval;
}
