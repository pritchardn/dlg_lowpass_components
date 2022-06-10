"""
dlg_lowpass_components generators
"""
import logging
import pickle

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
# @param[in] aparam/length Signal length/256/Integer/readwrite/
#     \~English Length of the output signal
# @param[in] aparam/samplerate Sample rate/5000/Integer/readwrite/
#     \~English The sample rate of the signal
# @param[in] aparam/frequencies Signal frequencies/{"values": [440, 800, 1000, 2000]}/Json/readwrite/
#     \~English A dictionary containing a single list of values - the frequencies incorporated in the original signal.
# @param[in] aparam/noise_params Noise parameters/{}/Json/readwrite/
#     \~English A dictionary containing several values defining the properties of an interleaved noise. noise, stddiv-deviation, frequency, random randomseed, noisemultiplier
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
    samplerate = dlg_int_param("samplerate", 5000)
    frequencies = dlg_dict_param("frequencies", {"values": [440, 800, 1000, 2000]})
    noise = dlg_dict_param(
        "noise", {}
    )  # {'noise': 0.0, 'stddiv': 1.0, 'frequency': 666, 'randomseed': 42})
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
        :param seed: The random randomseed
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
        for freq in self.frequencies["values"]:
            for i in range(self.length):
                series[i] += np.sin(2 * np.pi * i * freq / self.samplerate)
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
            if "noisemultiplier" in self.noise:
                self.noise["noisemultiplier"] = 1 / self.noise["noisemultiplier"]
            self.series = self.add_noise(
                self.series,
                self.noise["noise"],
                self.noise["stddiv"],
                self.noise["frequency"],
                self.samplerate,
                self.noise["randomseed"],
                self.noise.get("noisemultiplier", 0.1),
            )

        data = pickle.dumps(self.series)
        for output in outs:
            output.len = len(data)
            output.write(data)

    def generate_recompute_data(self):
        # This will do for now
        return {
            "length": self.length,
            "sample_rate": self.samplerate,
            "frequencies": self.frequencies,
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
# @param[in] aparam/length Signal length/256/Integer/readwrite/
#     \~English Length of the output signal
# @param[in] aparam/samplerate Sample rate/5000/Integer/readwrite/
#     \~English The sample rate of the signal
# @param[in] aparam/cutoff Filter cutoff/600/Integer/readwrite/
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
    samplerate = dlg_int_param("samplerate", 5000)
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
        alpha = 2 * self.cutoff / self.samplerate
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
        data = pickle.dumps(self.series)
        for output in outs:
            output.len = len(data)
            output.write(data)

    def generate_recompute_data(self):
        output = dict()
        output["length"] = self.length
        output["cutoff"] = self.cutoff
        output["sample_rate"] = self.samplerate
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
# @param[in] aparam/noise Average noise/0.0/Float/readwrite/
#     \~English The average value of the injected noise signal
# @param[in] aparam/samplerate Sample rate/5000/Integer/readwrite/
#     \~English The sample rate of the signal
# @param[in] aparam/stddiv Standard deviation/1.0/Float/readwrite/
#     \~English The standard deviation of the noise signal
# @param[in] aparam/frequency Noise frequency/1200/Integer/readwrite/
#     \~English The frequency of the noise
# @param[in] aparam/randomseed Random randomseed/42/Integer/readwrite/
#     \~English Random seed of the noise generator
# @param[in] aparam/noisemultiplier Noise multiplier/0.1/Float/readwrite/
#     \~English A gain factor for the injected noise (noisemultiplier).
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
    noise = dlg_float_param("noise", 0.0)
    stddiv = dlg_float_param("stddiv", 1.0)
    frequency = dlg_int_param("frequency", 1200)
    samplerate = dlg_int_param("samplerate", 5000)
    randomseed = dlg_int_param("randomseed", 42)
    noisemultiplier = dlg_float_param("noisemultiplier", 0.1)
    signal = np.empty([1])

    def add_noise(self):
        """
        Adds noise at a specified frequency.
        :return: Modified signal
        """
        np.random.seed(self.randomseed)
        samples = self.noisemultiplier * np.random.normal(
            self.noise, self.stddiv, size=len(self.signal)
        )
        for i in range(len(self.signal)):
            samples[i] += np.sin(2 * np.pi * i * self.frequency / self.samplerate)

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

        array = pickle.loads(droputils.allDropContents(ins[0]))
        self.signal = array

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
        data = pickle.dumps(sig)
        for output in outs:
            output.len = len(data)
            output.write(data)

    def generate_recompute_data(self):
        return {
            "noise": self.noise,
            "stddiv": self.stddiv,
            "sample_rate": self.samplerate,
            "randomseed": self.randomseed,
            "noisemultiplier": self.noisemultiplier,
            "system": system_summary(),
            "status": self.status,
        }
