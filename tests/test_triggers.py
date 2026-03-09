import time
import unittest

from bci_essentials.triggers import (
    MarkerTypes,
    TriggerByte,
    TriggerDetector,
    DEFAULT_TRIGGER_MAP,
    SIMPLE_TRIGGER_MAP,
    make_p300_trigger_map,
    make_mi_trigger_map,
    make_ssvep_trigger_map,
    make_simple_trigger_map,
)
from bci_essentials.io.fake_stim_source import FakeStimEegSource
from bci_essentials.io.eeg_trigger_sources import EegStimTriggerMarkerSource


class TestMarkerTypes(unittest.TestCase):
    def test_trial_started_value(self):
        self.assertEqual(MarkerTypes.TRIAL_STARTED.value, "Trial Started")

    def test_trial_ends_value(self):
        self.assertEqual(MarkerTypes.TRIAL_ENDS.value, "Trial Ends")

    def test_train_classifier_value(self):
        self.assertEqual(MarkerTypes.TRAIN_CLASSIFIER.value, "Train Classifier")


class TestTriggerByte(unittest.TestCase):
    def test_status_byte_range(self):
        self.assertEqual(TriggerByte.TRIAL_STARTED, 240)
        self.assertEqual(TriggerByte.TRIAL_ENDS, 241)
        self.assertEqual(TriggerByte.TRAINING_COMPLETE, 242)
        self.assertEqual(TriggerByte.TRAIN_CLASSIFIER, 243)
        self.assertEqual(TriggerByte.UPDATE_CLASSIFIER, 244)
        self.assertEqual(TriggerByte.DONE_RS_COLLECTION, 245)

    def test_target_bytes(self):
        self.assertEqual(TriggerByte.TARGET, 1)
        self.assertEqual(TriggerByte.NON_TARGET, 2)


class TestDefaultTriggerMap(unittest.TestCase):
    def test_maps_status_bytes_to_marker_strings(self):
        self.assertEqual(DEFAULT_TRIGGER_MAP[240], "Trial Started")
        self.assertEqual(DEFAULT_TRIGGER_MAP[241], "Trial Ends")
        self.assertEqual(DEFAULT_TRIGGER_MAP[243], "Train Classifier")

    def test_has_six_entries(self):
        self.assertEqual(len(DEFAULT_TRIGGER_MAP), 6)


class TestSimpleTriggerMap(unittest.TestCase):
    def test_maps_target_and_non_target(self):
        self.assertEqual(SIMPLE_TRIGGER_MAP[1], "target")
        self.assertEqual(SIMPLE_TRIGGER_MAP[2], "non_target")


class TestMakeP300TriggerMap(unittest.TestCase):
    def test_includes_status_markers(self):
        tmap = make_p300_trigger_map(6)
        self.assertIn(240, tmap)
        self.assertEqual(tmap[240], "Trial Started")

    def test_stimulus_entries(self):
        tmap = make_p300_trigger_map(6)
        self.assertEqual(tmap[1], "p300,s,6,-1,1")
        self.assertEqual(tmap[6], "p300,s,6,-1,6")

    def test_entry_count(self):
        tmap = make_p300_trigger_map(6)
        # 6 status + 6 stimulus
        self.assertEqual(len(tmap), 12)

    def test_raises_on_zero_options(self):
        with self.assertRaises(ValueError):
            make_p300_trigger_map(0)

    def test_raises_on_too_many_options(self):
        with self.assertRaises(ValueError):
            make_p300_trigger_map(240)


class TestMakeMiTriggerMap(unittest.TestCase):
    def test_stimulus_entries(self):
        tmap = make_mi_trigger_map(3, 2.0)
        self.assertEqual(tmap[1], "mi,3,1,2.00")
        self.assertEqual(tmap[2], "mi,3,2,2.00")
        self.assertEqual(tmap[3], "mi,3,3,2.00")

    def test_includes_status_markers(self):
        tmap = make_mi_trigger_map(3, 2.0)
        self.assertIn(240, tmap)

    def test_raises_on_bad_inputs(self):
        with self.assertRaises(ValueError):
            make_mi_trigger_map(0, 2.0)
        with self.assertRaises(ValueError):
            make_mi_trigger_map(3, 0)


class TestMakeSsvepTriggerMap(unittest.TestCase):
    def test_stimulus_entries(self):
        tmap = make_ssvep_trigger_map([8.0, 10.0, 12.0], 4.0)
        self.assertEqual(tmap[1], "ssvep,3,1,4.00,8.0,10.0,12.0")

    def test_includes_status_markers(self):
        tmap = make_ssvep_trigger_map([8.0], 1.0)
        self.assertIn(240, tmap)

    def test_raises_on_empty_frequencies(self):
        with self.assertRaises(ValueError):
            make_ssvep_trigger_map([], 1.0)


class TestMakeSimpleTriggerMap(unittest.TestCase):
    def test_default_bytes(self):
        tmap = make_simple_trigger_map()
        self.assertEqual(tmap[1], "target")
        self.assertEqual(tmap[2], "non_target")

    def test_custom_bytes(self):
        tmap = make_simple_trigger_map(target_byte=10, non_target_byte=11)
        self.assertEqual(tmap[10], "target")
        self.assertEqual(tmap[11], "non_target")

    def test_raises_on_same_bytes(self):
        with self.assertRaises(ValueError):
            make_simple_trigger_map(target_byte=5, non_target_byte=5)


class TestTriggerDetectorRiseMode(unittest.TestCase):
    def setUp(self) -> None:
        self.tmap = {240: "Trial Started", 1: "stim_0", 2: "stim_1"}
        self.detector = TriggerDetector(self.tmap, detect_mode="rise")

    def test_baseline_zero_returns_none(self):
        self.assertIsNone(self.detector.detect(0))

    def test_rise_from_zero_triggers(self):
        self.detector.detect(0)
        self.assertEqual(self.detector.detect(240), "Trial Started")

    def test_held_value_does_not_retrigger(self):
        self.detector.detect(0)
        self.detector.detect(240)
        self.assertIsNone(self.detector.detect(240))

    def test_return_to_zero_and_new_trigger(self):
        self.detector.detect(0)
        self.detector.detect(240)
        self.detector.detect(0)
        self.assertEqual(self.detector.detect(1), "stim_0")

    def test_unmapped_value_returns_none(self):
        self.detector.detect(0)
        self.assertIsNone(self.detector.detect(99))

    def test_raises_on_invalid_mode(self):
        with self.assertRaises(ValueError):
            TriggerDetector(self.tmap, detect_mode="invalid")


class TestTriggerDetectorChangeMode(unittest.TestCase):
    def setUp(self) -> None:
        self.tmap = {240: "Trial Started", 1: "stim_0"}
        self.detector = TriggerDetector(self.tmap, detect_mode="change")

    def test_change_triggers_on_any_transition(self):
        self.detector.detect(0)
        self.assertEqual(self.detector.detect(240), "Trial Started")
        # 240 -> 1 triggers on change
        self.assertEqual(self.detector.detect(1), "stim_0")

    def test_change_to_zero_returns_none(self):
        self.detector.detect(0)
        self.detector.detect(240)
        self.assertIsNone(self.detector.detect(0))


class TestTriggerDetectorUnmapped(unittest.TestCase):
    def test_include_unmapped_produces_fallback_string(self):
        detector = TriggerDetector({}, detect_mode="rise", include_unmapped=True)
        detector.detect(0)
        self.assertEqual(detector.detect(99), "trigger_99")

    def test_exclude_unmapped_tracks_warned_set(self):
        detector = TriggerDetector({}, detect_mode="rise", include_unmapped=False)
        detector.detect(0)
        detector.detect(99)
        self.assertIn(99, detector.warned_unmapped)


class TestFakeStimEegSource(unittest.TestCase):
    def test_properties(self):
        source = FakeStimEegSource(n_channels=9, fsample=256.0)
        self.assertEqual(source.name, "FakeStimEegSource")
        self.assertEqual(source.fsample, 256.0)
        self.assertEqual(source.n_channels, 9)
        self.assertEqual(len(source.channel_types), 9)
        self.assertEqual(len(source.channel_units), 9)
        self.assertEqual(len(source.channel_labels), 9)

    def test_stim_channel_label(self):
        source = FakeStimEegSource(n_channels=9)
        self.assertEqual(source.channel_labels[-1], "STIM")
        self.assertEqual(source.channel_types[-1], "stim")

    def test_custom_channel_labels(self):
        labels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "TRG"]
        source = FakeStimEegSource(n_channels=9, channel_labels=labels)
        self.assertEqual(source.channel_labels, labels)

    def test_raises_on_wrong_label_count(self):
        with self.assertRaises(ValueError):
            FakeStimEegSource(n_channels=9, channel_labels=["a", "b"])

    def test_raises_on_both_trigger_modes(self):
        with self.assertRaises(ValueError):
            FakeStimEegSource(trigger_bytes=[1], trigger_schedule=[(0.0, 1)])

    def test_time_correction_is_zero(self):
        source = FakeStimEegSource()
        self.assertEqual(source.time_correction(), 0.0)

    def test_get_samples_returns_data(self):
        source = FakeStimEegSource(n_channels=4, fsample=256.0, duration=1.0)
        source.get_samples()  # starts the internal clock
        time.sleep(0.1)
        samples, timestamps = source.get_samples()
        self.assertGreater(len(timestamps), 0)
        self.assertEqual(len(samples[0]), 4)

    def test_get_samples_empty_after_duration(self):
        source = FakeStimEegSource(n_channels=4, fsample=100.0, duration=0.01)
        time.sleep(0.05)
        # drain all samples
        source.get_samples()
        samples, timestamps = source.get_samples()
        self.assertEqual(samples, [[]])
        self.assertEqual(len(timestamps), 0)


class TestEegStimTriggerMarkerSource(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = [
            (0.0, 240),  # Trial Started
            (0.3, 1),  # flash 0
            (0.6, 2),  # flash 1
            (1.5, 241),  # Trial Ends
        ]
        self.fake_eeg = FakeStimEegSource(
            trigger_schedule=self.schedule,
            n_channels=9,
            fsample=256.0,
            duration=2.0,
        )
        self.trigger_map = make_p300_trigger_map(n_options=6)
        self.source = EegStimTriggerMarkerSource(
            eeg_source=self.fake_eeg,
            stim_channel_name="STIM",
            trigger_map=self.trigger_map,
            detect_mode="rise",
        )

    def test_eeg_properties_delegate(self):
        self.assertEqual(self.source.name, "FakeStimEegSource")
        self.assertEqual(self.source.fsample, 256.0)
        self.assertEqual(self.source.n_channels, 9)
        self.assertEqual(len(self.source.channel_labels), 9)
        self.assertEqual(len(self.source.channel_types), 9)
        self.assertEqual(len(self.source.channel_units), 9)

    def test_time_correction(self):
        self.assertEqual(self.source.time_correction(), 0.0)

    def test_get_markers_empty_before_samples(self):
        markers, timestamps = self.source.get_markers()
        self.assertEqual(markers, [[]])
        self.assertEqual(len(timestamps), 0)

    def test_end_to_end_trigger_detection(self):
        detected = []
        for _ in range(200):
            samples, ts = self.source.get_samples()
            if not ts:
                if self.fake_eeg._samples_generated >= self.fake_eeg._total_samples:
                    break
                time.sleep(0.01)
                continue
            markers, mts = self.source.get_markers()
            for m, t in zip(markers, mts):
                detected.append(m[0])
            time.sleep(0.01)

        self.assertEqual(len(detected), 4)
        self.assertEqual(detected[0], "Trial Started")
        self.assertEqual(detected[1], "p300,s,6,-1,1")
        self.assertEqual(detected[2], "p300,s,6,-1,2")
        self.assertEqual(detected[3], "Trial Ends")

    def test_raises_on_bad_eeg_source(self):
        with self.assertRaises(TypeError):
            EegStimTriggerMarkerSource(
                eeg_source="not_an_eeg_source",
                stim_channel_name="STIM",
            )

    def test_raises_on_empty_stim_channel_name(self):
        with self.assertRaises(ValueError):
            EegStimTriggerMarkerSource(
                eeg_source=self.fake_eeg,
                stim_channel_name="",
            )

    def test_raises_on_bad_detect_mode(self):
        with self.assertRaises(ValueError):
            EegStimTriggerMarkerSource(
                eeg_source=self.fake_eeg,
                stim_channel_name="STIM",
                detect_mode="invalid",
            )

    def test_enabled_property(self):
        self.assertTrue(self.source.enabled)
        self.source.enabled = False
        self.assertFalse(self.source.enabled)

    def test_disabled_returns_no_markers(self):
        self.source.enabled = False
        time.sleep(0.1)
        self.source.get_samples()
        markers, timestamps = self.source.get_markers()
        self.assertEqual(markers, [[]])
        self.assertEqual(len(timestamps), 0)


if __name__ == "__main__":
    unittest.main()
