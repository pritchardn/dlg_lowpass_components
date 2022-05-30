"""
dlg_lowpass_components filters
"""
import logging

import numpy as np
import pyfftw
from dlg import droputils
from dlg.common.reproducibility.constants import system_summary
from dlg.drop import BarrierAppDROP
from dlg.meta import (
    dlg_batch_input,
    dlg_batch_output,
    dlg_bool_param,
    dlg_component,
    dlg_streaming_input,
)

from dlg_lowpass_components.utils import determine_size

logger = logging.getLogger(__name__)


##
# @brief LPFilterfftNP
# @details Implements a lowpass filter via fft with numpy
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_lowpass_components.LPFilterFFTNP/String/readonly/
#     \~English Import direction for application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/doubleprecision Double Precision/false/Boolean/readwrite/
#     \~English Whether to use double (true) or float (false) precision.
# @param[in] port/signal Signal/float/
#     \~English Numpy array containing incoming signal (string dump of floats)
# @param[in] port/window Signal/float/
#     \~English Numpy array containing the filter window (string dump of floats)
# @param[out] port/signal Signal/Complex/
#     \~English Numpy array containing final signal (complex)
# @par EAGLE_END
class LPFilterFFTNP(BarrierAppDROP):
    """
    Uses numpy to filter a nosiy signal.
    """

    component_meta = dlg_component(
        "LP_filter_np",
        "Filters a signal with " "a provided window using numpy",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    PRECISIONS = {
        "double": {"float": np.float64, "complex": np.complex128},
        "single": {"float": np.float32, "complex": np.complex64},
    }
    precision = {}
    # default values
    double_prec = dlg_bool_param("doublePrec", True)
    series = []
    output = np.zeros([1])

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        if self.double_prec:
            self.precision = self.PRECISIONS["double"]
        else:
            self.precision = self.PRECISIONS["single"]

    def get_inputs(self):
        """
        Reads input arrays into numpy array
        :return: Sets class series variable.
        """
        ins = self.inputs
        if len(ins) != 2:
            raise Exception("Precisely two input required for %r" % self)

        array = [np.frombuffer(droputils.allDropContents(inp)) for inp in ins]
        self.series = array

    def filter(self):
        """
        Actually performs the filtering
        :return: Numpy array of filtered signal.
        """
        signal = self.series[0]
        window = self.series[1]
        nfft = determine_size(len(signal) + len(window) - 1)
        sig_zero_pad = np.zeros(nfft, dtype=self.precision["float"])
        win_zero_pad = np.zeros(nfft, dtype=self.precision["float"])
        sig_zero_pad[0 : len(signal)] = signal
        win_zero_pad[0 : len(window)] = window
        sig_fft = np.fft.fft(sig_zero_pad)
        win_fft = np.fft.fft(win_zero_pad)
        out_fft = np.multiply(sig_fft, win_fft)
        out = np.fft.ifft(out_fft)
        return out.astype(self.precision["complex"])

    def run(self):
        """
        Called by DALiuGE to start execution
        :return:
        """
        outs = self.outputs
        if len(outs) < 1:
            raise Exception("At least one output required for %r" % self)
        self.get_inputs()
        self.output = self.filter()
        data = self.output.tobytes()
        for output in outs:
            output.len = len(data)
            output.write(data)

    def generate_recompute_data(self):
        return {
            "precision_float": str(self.precision["float"]),
            "precision_complex": str(self.precision["complex"]),
            "system": system_summary(),
            "status": self.status,
        }


##
# @brief LPFilterFFTFFTW
# @details Implements a lowpass filter via fft with FFTW
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_lowpass_components.LPFilterFFTFFTW/String/readonly/
#     \~English Import direction for application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/doubleprecision Double Precision/false/Boolean/readwrite/
#     \~English Whether to use double (true) or float (false) precision.
# @param[in] port/signal Signal/float/
#     \~English Numpy array containing incoming signal (string dump of floats)
# @param[in] port/window Signal/float/
#     \~English Numpy array containing the filter window (string dump of floats)
# @param[out] port/signal Signal/Complex/
#     \~English Numpy array containing final signal (complex)
# @par EAGLE_END
class LPFilterFFTFFTW(LPFilterFFTNP):
    """
    Uses fftw to implement a low-pass filter
    """

    component_meta = dlg_component(
        "LP_filter_fftw",
        "Filters a signal with " "a provided window using FFTW",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    def filter(self):
        """
        Actually performs the filtering
        :return: Filtered signal as numpy array.
        """
        pyfftw.interfaces.cache.disable()
        signal = self.series[0]
        window = self.series[1]
        nfft = determine_size(len(signal) + len(window) - 1)
        sig_zero_pad = pyfftw.empty_aligned(len(signal), dtype=self.precision["float"])
        win_zero_pad = pyfftw.empty_aligned(len(window), dtype=self.precision["float"])
        sig_zero_pad[0 : len(signal)] = signal
        win_zero_pad[0 : len(window)] = window
        sig_fft = pyfftw.interfaces.numpy_fft.fft(sig_zero_pad, n=nfft)
        win_fft = pyfftw.interfaces.numpy_fft.fft(win_zero_pad, n=nfft)
        out_fft = np.multiply(sig_fft, win_fft)
        out = pyfftw.interfaces.numpy_fft.ifft(out_fft, n=nfft)
        return out.astype(self.precision["complex"])


##
# @brief LPFilterFFTCuda
# @details Implements a lowpass filter via fft with cuda
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_lowpass_components.LPFilterFFTCuda/String/readonly/
#     \~English Import direction for application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/doubleprecision Double Precision/true/Boolean/readwrite/
#     \~English Whether to use double (true) or float (false) precision.
# @param[in] port/signal Signal/float/
#     \~English Numpy array containing incoming signal (string dump of floats)
# @param[in] port/window Signal/float/
#     \~English Numpy array containing the filter window (string dump of floats)
# @param[out] port/signal Signal/Complex/
#     \~English Numpy array containing final signal (complex)
# @par EAGLE_END
class LPFilterFFTCuda(LPFilterFFTNP):
    """
    Uses pycuda to implement a low-pass filter
    """

    component_meta = dlg_component(
        "LPFilterFFTCuda",
        "Filters a signal with " "a provided window using cuda",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    def filter(self):
        """
        Actually performs the filtering
        :return:
        """
        import pycuda.gpuarray as gpuarray
        import skcuda.fft as cu_fft
        import skcuda.linalg as linalg
        import pycuda.driver as cuda
        from pycuda.tools import make_default_context

        cuda.init()
        context = make_default_context()
        device = context.get_device()
        signal = self.series[0]
        window = self.series[1]
        linalg.init()
        nfft = determine_size(len(signal) + len(window) - 1)
        # Move data to GPU
        sig_zero_pad = np.zeros(nfft, dtype=self.precision["float"])
        win_zero_pad = np.zeros(nfft, dtype=self.precision["float"])
        sig_gpu = gpuarray.zeros(sig_zero_pad.shape, dtype=self.precision["float"])
        win_gpu = gpuarray.zeros(win_zero_pad.shape, dtype=self.precision["float"])
        sig_zero_pad[0 : len(signal)] = signal
        win_zero_pad[0 : len(window)] = window
        sig_gpu.set(sig_zero_pad)
        win_gpu.set(win_zero_pad)

        # Plan forwards
        sig_fft_gpu = gpuarray.zeros(nfft, dtype=self.precision["complex"])
        win_fft_gpu = gpuarray.zeros(nfft, dtype=self.precision["complex"])
        sig_plan_forward = cu_fft.Plan(
            sig_fft_gpu.shape, self.precision["float"], self.precision["complex"]
        )
        win_plan_forward = cu_fft.Plan(
            win_fft_gpu.shape, self.precision["float"], self.precision["complex"]
        )
        cu_fft.fft(sig_gpu, sig_fft_gpu, sig_plan_forward)
        cu_fft.fft(win_gpu, win_fft_gpu, win_plan_forward)

        # Convolve
        out_fft = linalg.multiply(sig_fft_gpu, win_fft_gpu, overwrite=True)
        linalg.scale(2.0, out_fft)

        # Plan inverse
        out_gpu = gpuarray.zeros_like(out_fft)
        plan_inverse = cu_fft.Plan(
            out_fft.shape, self.precision["complex"], self.precision["complex"]
        )
        cu_fft.ifft(out_fft, out_gpu, plan_inverse, True)
        out_np = np.zeros(len(out_gpu), self.precision["complex"])
        out_gpu.get(out_np)
        context.pop()
        return out_np


##
# @brief LPFilterPointwiseNP
# @details Implements a lowpass filter via pointwise convolution with numpy
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_lowpass_components.LPFilterPointwiseNP/String/readonly/
#     \~English Import direction for application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/doubleprecision Double Precision/false/Boolean/readwrite/
#     \~English Whether to use double (true) or float (false) precision.
# @param[in] port/signal Signal/float/
#     \~English Numpy array containing incoming signal (string dump of floats)
# @param[in] port/window Signal/float/
#     \~English Numpy array containing the filter window (string dump of floats)
# @param[out] port/signal Signal/Complex/
#     \~English Numpy array containing final signal (complex)
# @par EAGLE_END
class LPFilterPointwiseNP(LPFilterFFTNP):
    """
    Uses raw numpy to implement a low-pass filter
    """

    component_meta = dlg_component(
        "LPFilterPointwiseNP",
        "Filters a signal with " "a provided window using cuda",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    def filter(self):
        return np.convolve(self.series[0], self.series[1], mode="full").astype(
            self.precision["complex"]
        )
