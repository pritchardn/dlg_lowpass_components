import unittest
import time
from dlg_lowpass_components import LPSignalGenerator, LPWindowGenerator, LPAddNoise
from dlg.drop import InMemoryDROP
from dlg.droputils import allDropContents, DROPWaiterCtx

def _run_component(component):
    memory = InMemoryDROP("b", "b")
    component.addOutput(memory)
    component.run()
    memory.setCompleted()
    return allDropContents(memory)


class TestLPSignalGenerator(unittest.TestCase):

    def test_default(self):
        generator = LPSignalGenerator("a", "a")
        signal = _run_component(generator)
        self.assertIsNotNone(signal)
        self.assertEqual(generator.length * 8, len(signal))

    def test_changed_length(self):
        new_length = 1024
        generator = LPSignalGenerator("a", "a", length=new_length)
        signal = _run_component(generator)
        self.assertIsNotNone(signal)
        self.assertEqual(new_length, generator.length)
        self.assertEqual(new_length * 8, len(signal))

    def test_added_noise(self):
        noise_params = {'mean': 0.0, 'std': 1.0, 'freq': 666, 'seed': 42}
        noisy_generator = LPSignalGenerator("a", "a", noise=noise_params)
        vanilla_generator = LPSignalGenerator("A", "A")
        noisy_signal = _run_component(noisy_generator)
        vanilla_signal = _run_component(vanilla_generator)
        self.assertIsNotNone(noisy_signal)
        self.assertIsNotNone(vanilla_signal)
        self.assertNotEqual(noisy_signal, vanilla_signal)

    def test_changed_frequency(self):
        new_frequency = {'values': [440, 900, 1100, 1900, 2400]}
        generator = LPSignalGenerator("a", "a", freqs=new_frequency)
        vanilla_generator = LPSignalGenerator("A", "A")
        signal = _run_component(generator)
        vanilla_signal = _run_component(vanilla_generator)

        self.assertIsNotNone(signal)
        self.assertIsNotNone(vanilla_signal)
        self.assertNotEqual(signal, vanilla_signal)

class TestLPWindowGenerator(unittest.TestCase):

    def test_default(self):
        generator = LPWindowGenerator("a", "a")
        signal = _run_component(generator)
        self.assertIsNotNone(signal)
        self.assertEqual(generator.length * 8, len(signal))

    def test_change_length(self):
        new_length = 512
        generator = LPWindowGenerator("a", "a", length=new_length)
        signal = _run_component(generator)
        self.assertIsNotNone(signal)
        self.assertEqual(new_length, generator.length)
        self.assertEqual(new_length * 8, len(signal))

    def test_change_cutoff(self):
        new_cutoff = 660
        generator = LPWindowGenerator("a", "a", cutoff=new_cutoff)
        vanilla_generator = LPWindowGenerator("A", "A")
        signal = _run_component(generator)
        vanilla_signal = _run_component(vanilla_generator)
        self.assertIsNotNone(signal)
        self.assertIsNotNone(vanilla_signal)
        self.assertNotEqual(signal, vanilla_signal)

    def test_change_srate(self):
        new_srate = 4500
        generator = LPWindowGenerator("a", "a", srate=new_srate)
        vanilla_generator = LPWindowGenerator("A", "A")
        signal = _run_component(generator)
        vanilla_signal = _run_component(vanilla_generator)
        self.assertIsNotNone(signal)
        self.assertIsNotNone(vanilla_signal)
        self.assertNotEqual(signal, vanilla_signal)


class TestLPNoiseGenerator(unittest.TestCase):

    def test_compare_to_SignalGen(self):
        noise_params = {'mean': 0.0, 'std': 1.0, 'freq': 1200, 'seed': 42}
        noisy_generator = LPSignalGenerator("a", "a", noise=noise_params)
        noisy_signal = _run_component(noisy_generator)

        vanilla_generator = LPSignalGenerator("A", "A")
        interim_mem = InMemoryDROP("B", "B")
        noisy_adder = LPAddNoise("C", "C")
        final_mem = InMemoryDROP("D", "D")

        vanilla_generator.addOutput(interim_mem)
        noisy_adder.addInput(interim_mem)
        noisy_adder.addOutput(final_mem)

        with DROPWaiterCtx(self, final_mem, 10):
            vanilla_generator.run()
            interim_mem.setCompleted()

        final_signal = allDropContents(final_mem)

        self.assertIsNotNone(noisy_signal)
        self.assertIsNotNone(final_signal)
        self.assertEqual(noisy_signal, final_signal)
