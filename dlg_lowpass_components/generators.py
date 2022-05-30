"""
dlg_lowpass_components generators
"""
import logging

import numpy as np
from dlg import droputils
from dlg.common.reproducibility.constants import system_summary
from dlg.drop import BarrierAppDROP
from dlg.meta import (
    dlg_batch_input,
    dlg_batch_output,
    dlg_component,
    dlg_float_param,
    dlg_int_param,
    dlg_streaming_input,
    dlg_dict_param,
)

logger = logging.getLogger(__name__)


##
# @brief LP_SignalGenerator
# @details Generates a noisy sine signal for filtering. Effectively an input generator.
#
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_lowpass_components.LPSignalGenerator/String/readonly/
#     \~English Import direction for application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/length Signal length/ /Integer/readwrite/
#     \~English Length of the output signal
# @param[in] cparam/samplerate Sample rate/ /Integer/readwrite/
#     \~English The sample rate of the signal
# @param[in] cparam/frequencies Signal frequencies/ /Json/readwrite/
#     \~English A dictionary containing a single list of values - the frequencies incorporated in the original signal.
# @param[in] cparam/noise_params Noise parameters/ /Json/readwrite/
#     \~English A dictionary containing several values defining the properties of an interleaved noise. mean, std-deviation, frequency, random seed, alpha
# @param[out] port/signal Signal/float/
#     \~English Numpy array containing final signal (purely real (floats))
# @par EAGLE_END
class LPSignalGenerator(BarrierAppDROP):
    """
    Generates a noisy sine signal for filtering. Effectively an input generator.
    """

    component_meta = dlg_component(
        "LPSignalGen",
        "Low-pass filter example signal generator",
        [None],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    # default values
    length = dlg_int_param("length", 256)
    srate = dlg_int_param("samplerate", 5000)
    freqs = dlg_dict_param("frequencies", {"values": [440, 800, 1000, 2000]})
    noise = dlg_dict_param(
        "noise", {}
    )  # {'mean': 0.0, 'std': 1.0, 'freq': 666, 'seed': 42})
    series = None

    def initialize(self, **kwargs):
        super(LPSignalGenerator, self).initialize(**kwargs)

    def add_noise(
        self, series: np.array, mean, std, freq, sample_rate, seed, alpha=0.1
    ):
        """
        A noise to the provided signal by producing random values of a given frequency
        :param series: The input (and output) numpy array signal series
        :param mean: The average value
        :param std: The standard deviation of the value
        :param freq: The frequency of the noisy signal
        :param sample_rate: The sample rate of the input series
        :param seed: The random seed
        :param alpha: The multiplier
        :return: The input series with noisy values added
        """
        np.random.seed(seed)
        samples = alpha * np.random.normal(mean, std, size=len(series))
        for i in range(len(series)):
            samples[i] += np.sin(2 * np.pi * i * freq / sample_rate)
        np.add(series, samples, out=series)
        return series

    def gen_sig(self):
        """
        Generates an initial signal
        :return: Numpy array of signal values.
        """
        series = np.zeros(self.length, dtype=np.float64)
        for freq in self.freqs["values"]:
            for i in range(self.length):
                series[i] += np.sin(2 * np.pi * i * freq / self.srate)
        return series

    def run(self):
        """
        Called by DALiuGE to start signal generation. Conditionally adds noise if parameters are set
        :return: Writes signal to output ports.
        """
        outs = self.outputs
        if len(outs) < 1:
            raise Exception("At least one output required for %r" % self)
        self.series = self.gen_sig()
        if len(self.noise) > 0:
            if "alpha" in self.noise:
                self.noise["alpha"] = 1 / self.noise["alpha"]
            self.series = self.add_noise(
                self.series,
                self.noise["mean"],
                self.noise["std"],
                self.noise["freq"],
                self.srate,
                self.noise["seed"],
                self.noise.get("alpha", 0.1),
            )

        data = self.series.tobytes()
        for output in outs:
            output.len = len(data)
            output.write(data)

    def generate_recompute_data(self):
        # This will do for now
        return {
            "length": self.length,
            "sample_rate": self.srate,
            "frequencies": self.freqs,
            "status": self.status,
            "system": system_summary(),
        }


##
# @brief LP_WindowGenerator
# @details Generates a Hann window for low-pass filtering.
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_lowpass_components.LPWindowGenerator/String/readonly/
#     \~English Import direction for application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/length Signal length/ /Integer/readwrite/
#     \~English Length of the output signal
# @param[in] cparam/samplerate Sample rate/ /Integer/readwrite/
#     \~English The sample rate of the signal
# @param[in] cparam/cutoff Filter cutoff/ /Integer/readwrite/
#     \~English The frequency of the low-pass filter
# @param[out] port/window Window/float/
#     \~English Numpy array containing final signal (purely real (floats))
# @par EAGLE_END
class LPWindowGenerator(BarrierAppDROP):
    """
    Generates a Hann window for low-pass filtering.
    """

    component_meta = dlg_component(
        "LPWindowGen",
        "Low-pass filter example window generator",
        [None],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    # default values
    length = dlg_int_param("length", 256)
    cutoff = dlg_int_param("cutoff", 600)
    srate = dlg_int_param("samplerate", 5000)
    series = None

    def sinc(self, x_val: np.float64):
        """
        Computes the sin_c value for the input float
        :param x_val:
        """
        if np.isclose(x_val, 0.0):
            return 1.0
        return np.sin(np.pi * x_val) / (np.pi * x_val)

    def gen_win(self):
        """
        Generates the window values.
        :return: Numpy array of window series.
        """
        alpha = 2 * self.cutoff / self.srate
        win = np.zeros(self.length, dtype=np.float64)
        for i in range(int(self.length)):
            ham = 0.54 - 0.46 * np.cos(
                2 * np.pi * i / int(self.length)
            )  # Hamming coefficient
            hsupp = i - int(self.length) / 2
            win[i] = ham * alpha * self.sinc(alpha * hsupp)
        return win

    def run(self):
        """
        Called by DALiuGE to start drop execution
        :return:
        """
        outs = self.outputs
        if len(outs) < 1:
            raise Exception("At least one output required for %r" % self)
        self.series = self.gen_win()
        data = self.series.tobytes()
        for output in outs:
            output.len = len(data)
            output.write(data)

    def generate_recompute_data(self):
        output = dict()
        output["length"] = self.length
        output["cutoff"] = self.cutoff
        output["sample_rate"] = self.srate
        output["status"] = self.status
        output["system"] = system_summary()
        return output


##
# @brief LPAddNoise
# @details Component to add additional noise to a signal array.
# @par EAGLE_START
# @param category PythonApp
# @param[in] cparam/appclass appclass/dlg_lowpass_components.LPAddNoise/String/readonly/
#     \~English Import direction for application class
# @param[in] cparam/execution_time Execution Time/5/Float/readonly/False//False/
#     \~English Estimated execution time
# @param[in] cparam/num_cpus No. of CPUs/1/Integer/readonly/False//False/
#     \~English Number of cores used
# @param[in] cparam/noise Average noise/ /Float/readwrite/
#     \~English The average value of the injected noise signal
# @param[in] cparam/samplerate Sample rate/ /Integer/readwrite/
#     \~English The sample rate of the signal
# @param[in] cparam/stddiv Standard deviation/ /Float/readwrite/
#     \~English The standard deviation of the noise signal
# @param[in] cparam/frequency Noise frequency/ /Integer/readwrite/
#     \~English The frequency of the noise
# @param[in] cparam/randomseed Random seed/ /Integer/readwrite/
#     \~English Random seed of the noise generator
# @param[in] cparam/noisemultiplier Noise multiplier/ /Float/readwrite/
#     \~English A gain factor for the injected noise (alpha).
# @param[in] port/signal Signal/float/
#     \~English Numpy array containing incoming signal (string dump of floats)
# @param[out] port/signal Signal/float/
#     \~English Numpy array containing final signal (purely real (floats))
# @par EAGLE_END
class LPAddNoise(BarrierAppDROP):
    """
    Component to add additional noise to a signal array.
    """

    component_meta = dlg_component(
        "LPAddNoise",
        "Adds noise to a signal generated " "for the low-pass filter example",
        [dlg_batch_input("binary/*", [])],
        [dlg_batch_output("binary/*", [])],
        [dlg_streaming_input("binary/*")],
    )

    # default values
    mean = dlg_float_param("avg_noise", 0.0)
    std = dlg_float_param("std_deviation", 1.0)
    freq = dlg_int_param("frequency", 1200)
    srate = dlg_int_param("sample_rate", 5000)
    seed = dlg_int_param("random_seed", 42)
    alpha = dlg_float_param("noise_multiplier", 0.1)
    signal = np.empty([1])

    def add_noise(self):
        """
        Adds noise at a specified frequency.
        :return: Modified signal
        """
        np.random.seed(self.seed)
        samples = self.alpha * np.random.normal(
            self.mean, self.std, size=len(self.signal)
        )
        for i in range(len(self.signal)):
            samples[i] += np.sin(2 * np.pi * i * self.freq / self.srate)

        out_array = np.empty(self.signal.shape)
        np.add(self.signal, samples, out=out_array)
        self.signal = out_array
        return self.signal

    def get_inputs(self):
        """
        Reads input data into a numpy array.
        :return:
        """
        ins = self.inputs
        if len(ins) != 1:
            raise Exception("Precisely one input required for %r" % self)

        array = np.frombuffer(droputils.allDropContents(ins[0]))
        self.signal = np.frombuffer(array)

    def run(self):
        """
        Called by DALiuGE to start drop execution.
        :return:
        """
        outs = self.outputs
        if len(outs) < 1:
            raise Exception("At least one output required for %r" % self)
        self.get_inputs()
        sig = self.add_noise()
        data = sig.tobytes()
        for output in outs:
            output.len = len(data)
            output.write(data)

    def generate_recompute_data(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "sample_rate": self.srate,
            "seed": self.seed,
            "alpha": self.alpha,
            "system": system_summary(),
            "status": self.status,
        }
