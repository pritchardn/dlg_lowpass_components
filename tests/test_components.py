import unittest
import time
from dlg_lowpass_components import LPSignalGenerator, LPWindowGenerator, LPAddNoise
from dlg.drop import InMemoryDROP
from dlg.droputils import allDropContents, DROPWaiterCtx


class TestLPSignalGenerator(unittest.TestCase):

    def test_default(self):
        generator = LPSignalGenerator("a", "a")
        memory = InMemoryDROP("b", "b")
        generator.addOutput(memory)
        generator.run()
        memory.setCompleted()
        signal = allDropContents(memory)
        self.assertIsNotNone(signal)
        self.assertEqual(generator.length * 8, len(signal))

    def test_changed_length(self):
        new_length = 1024
        generator = LPSignalGenerator("a", "a", length=new_length)
        memory = InMemoryDROP("b", "b")
        generator.addOutput(memory)
        generator.run()
        memory.setCompleted()
        signal = allDropContents(memory)
        self.assertIsNotNone(signal)
        self.assertEqual(new_length, generator.length)
        self.assertEqual(new_length * 8, len(signal))

    def test_added_noise(self):
        noise_params = {'mean': 0.0, 'std': 1.0, 'freq': 666, 'seed': 42}
        noisy_generator = LPSignalGenerator("a", "a", noise=noise_params)
        vanilla_generator = LPSignalGenerator("A", "A")
        noisy_memory = InMemoryDROP("b", "b")
        vanilla_memory = InMemoryDROP("B", "B")
        noisy_generator.addOutput(noisy_memory)
        vanilla_generator.addOutput(vanilla_memory)
        noisy_generator.run()
        vanilla_generator.run()
        noisy_memory.setCompleted()
        vanilla_memory.setCompleted()
        noisy_signal = allDropContents(noisy_memory)
        vanilla_signal = allDropContents(vanilla_memory)
        self.assertIsNotNone(noisy_signal)
        self.assertIsNotNone(vanilla_signal)
        self.assertNotEqual(noisy_signal, vanilla_signal)

    def test_changed_frequency(self):
        new_frequency = {'values': [440, 900, 1100, 1900, 2400]}
        generator = LPSignalGenerator("a", "a", freqs=new_frequency)
        memory = InMemoryDROP("b", "b")
        vanilla_generator = LPSignalGenerator("A", "A")
        vanilla_memory = InMemoryDROP("B", "B")
        generator.addOutput(memory)
        vanilla_generator.addOutput(vanilla_memory)
        generator.run()
        vanilla_generator.run()
        memory.setCompleted()
        vanilla_memory.setCompleted()
        signal = allDropContents(memory)
        vanilla_signal = allDropContents(vanilla_memory)

        self.assertIsNotNone(signal)
        self.assertIsNotNone(vanilla_signal)
        self.assertNotEqual(signal, vanilla_signal)

class TestLPWindowGenerator(unittest.TestCase):

    def test_default(self):
        generator = LPWindowGenerator("a", "a")
        memory = InMemoryDROP("b", "b")
        generator.addOutput(memory)
        generator.run()
        memory.setCompleted()
        signal = allDropContents(memory)
        self.assertIsNotNone(signal)
        self.assertEqual(generator.length * 8, len(signal))

    def test_change_length(self):
        new_length = 512
        generator = LPWindowGenerator("a", "a", length=new_length)
        memory = InMemoryDROP("b", "b")
        generator.addOutput(memory)
        generator.run()
        memory.setCompleted()
        signal = allDropContents(memory)
        self.assertIsNotNone(signal)
        self.assertEqual(new_length, generator.length)
        self.assertEqual(new_length * 8, len(signal))

    def test_change_cutoff(self):
        new_cutoff = 660
        generator = LPWindowGenerator("a", "a", freqs=new_cutoff)
        memory = InMemoryDROP("b", "b")
        vanilla_generator = LPWindowGenerator("A", "A")
        vanilla_memory = InMemoryDROP("B", "B")
        generator.addOutput(memory)
        vanilla_generator.addOutput(vanilla_memory)
        generator.run()
        vanilla_generator.run()
        memory.setCompleted()
        vanilla_memory.setCompleted()
        signal = allDropContents(memory)
        vanilla_signal = allDropContents(vanilla_memory)

    def test_change_srate(self):
        new_srate = 4500
        generator = LPWindowGenerator("a", "a", srate=new_srate)
        memory = InMemoryDROP("b", "b")
        vanilla_generator = LPWindowGenerator("A", "A")
        vanilla_memory = InMemoryDROP("B", "B")
        generator.addOutput(memory)
        vanilla_generator.addOutput(vanilla_memory)
        generator.run()
        vanilla_generator.run()
        memory.setCompleted()
        vanilla_memory.setCompleted()
        signal = allDropContents(memory)
        vanilla_signal = allDropContents(vanilla_memory)


class TestLPNoiseGenerator(unittest.TestCase):

    def test_compare_to_SignalGen(self):
        noise_params = {'mean': 0.0, 'std': 1.0, 'freq': 1200, 'seed': 42}
        noisy_generator = LPSignalGenerator("a", "a", noise=noise_params)
        noisy_memory = InMemoryDROP("b", "b")
        noisy_generator.addOutput(noisy_memory)
        noisy_generator.run()
        noisy_memory.setCompleted()
        noisy_signal = allDropContents(noisy_memory)

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
