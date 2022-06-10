import pickle

import numpy as np
from dlg.drop import BarrierAppDROP
from dlg import droputils
from dlg.common.reproducibility.constants import system_summary
from dlg.meta import (
    dlg_batch_input,
    dlg_batch_output,
    dlg_component,
    dlg_float_param,
    dlg_int_param,
    dlg_bool_param,
    dlg_streaming_input,
    dlg_dict_param,
)

PRECISIONS = {
    "double": {"float": np.float64, "complex": np.complex128},
    "single": {"float": np.float32, "complex": np.complex64},
}


def determine_size(length):
    """
    :param length:
    :return: Computes the next largest power of two needed to contain |length| elements
    """
    return int(2 ** np.ceil(np.log2(length))) - 1


def normalize_signal(series):
    astd = np.std(series)
    series /= astd
    return series


def correlate_signals(series_a, series_b):
    return np.absolute(np.correlate(series_a, series_b, mode="valid") / len(series_a))


##
# @brief LPCorrelate
# @details Component to compute correlation between two numpy series
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_lowpass_components.LPCorrelate/String/readonly/
#     \~English Import direction for application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] aparam/normalize Normalize Signal/false/Boolean/readwrite/
#     \~English Whether to normalize the input signals (True) or not (False).
# @param[in] aparam/doubleprecision Double Precision/false/Boolean/readwrite/
#     \~English Whether to use double (true) or float (false) precision.
# @param[in] port/signal Signal A/Complex/
#     \~English Numpy array containing incoming signal (string dump of floats)
# @param[in] port/signal Signal B/Complex/
#     \~English Numpy array containing incoming signal (string dump of floats)
# @param[out] port/correlation Correlation/float/
#     \~English Numpy array containing a single value, the (normalized) cross correlation between two series.
# @par EAGLE_END
class LPCorrelate(BarrierAppDROP):
    """
    Component to compute correlation between two numpy series
    """

    component_meta = dlg_component(
        "LPCorrelate",
        "Computes cross correlation betweeen two series",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    normalize = dlg_bool_param("normalize", False)
    precision = {}
    # default values
    doubleprecision = dlg_bool_param("doubleprecision", True)

    def _get_inputs(self):
        ins = self.inputs
        if len(ins) != 2:
            raise Exception("Needs two inputs to function")
        self.signal_a = pickle.loads(
            droputils.allDropContents(ins[0])
        )
        self.signal_b = pickle.loads(
            droputils.allDropContents(ins[1])
        )

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        if self.doubleprecision:
            self.precision = PRECISIONS["double"]
        else:
            self.precision = PRECISIONS["single"]

    def run(self):
        outs = self.outputs
        if len(outs) < 1:
            raise Exception("At least one output required for %r" % self)
        self._get_inputs()
        if self.normalize:
            ncc = correlate_signals(
                normalize_signal(self.signal_a), normalize_signal(self.signal_b)
            )
        else:
            ncc = correlate_signals(self.signal_a, self.signal_b)
        ncc = np.round(ncc, int(np.ceil(np.log10(len(self.signal_a)))))
        ncc = pickle.dumps(ncc)
        for output in outs:
            output.len = len(ncc)
            output.write(ncc)

    def generate_recompute_data(self):
        return {"normalize": self.normalize, "status": self.status}
