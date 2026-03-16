import unittest

from bci_essentials.io.eeg_trigger_sources import (
    DEFAULT_TRIGGER_MAP,
    EegStimTriggerMarkerSource,
    make_p300_trigger_map,
    make_mi_trigger_map,
    make_ssvep_trigger_map,
    make_simple_trigger_map,
    TARGET_BYTE,
    NON_TARGET_BYTE,
)
from bci_essentials.io.sources import EegSource, MarkerSource


class FakeEegSource(EegSource):
    def __init__(
        self,
        samples=None,
        timestamps=None,
        channel_labels=None,
        fsample=256.0,
    ):
        self._channel_labels = channel_labels or ["Ch0", "Ch1", "Ch2", "STIM"]
        self._n_channels = len(self._channel_labels)
        self._batches = []
        if samples is not None:
            self._batches.append((samples, timestamps or []))
        self._batch_idx = 0
        self._fsample = fsample

    def add_batch(self, samples, timestamps):
        self._batches.append((samples, timestamps))

    @property
    def name(self) -> str:
        return "FakeEegSource"

    @property
    def fsample(self) -> float:
        return self._fsample

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def channel_types(self) -> list[str]:
        return ["eeg"] * self._n_channels

    @property
    def channel_units(self) -> list[str]:
        return ["microvolts"] * self._n_channels

    @property
    def channel_labels(self) -> list[str]:
        return list(self._channel_labels)

    def get_samples(self) -> tuple[list[list], list]:
        if self._batch_idx < len(self._batches):
            result = self._batches[self._batch_idx]
            self._batch_idx += 1
            return result
        return [[]], []

    def time_correction(self) -> float:
        return 0.0


class TestDefaultTriggerMap(unittest.TestCase):
    def test_has_six_status_entries(self):
        self.assertEqual(len(DEFAULT_TRIGGER_MAP), 6)

    def test_known_byte_values(self):
        self.assertEqual(DEFAULT_TRIGGER_MAP[240], "Trial Started")
        self.assertEqual(DEFAULT_TRIGGER_MAP[241], "Trial Ends")
        self.assertEqual(DEFAULT_TRIGGER_MAP[242], "Training Complete")
        self.assertEqual(DEFAULT_TRIGGER_MAP[243], "Train Classifier")
        self.assertEqual(DEFAULT_TRIGGER_MAP[244], "Update Classifier")
        self.assertEqual(DEFAULT_TRIGGER_MAP[245], "Done with all RS collection")


class TestMakeSimpleTriggerMap(unittest.TestCase):
    def test_target_and_non_target_present(self):
        m = make_simple_trigger_map(include_status=False)
        self.assertEqual(m[TARGET_BYTE], "target")
        self.assertEqual(m[NON_TARGET_BYTE], "non_target")

    def test_includes_status_by_default(self):
        m = make_simple_trigger_map()
        self.assertEqual(m[TARGET_BYTE], "target")
        self.assertEqual(m[240], "Trial Started")

    def test_custom_bytes(self):
        m = make_simple_trigger_map(
            target_byte=10, non_target_byte=20, include_status=False
        )
        self.assertEqual(m[10], "target")
        self.assertEqual(m[20], "non_target")

    def test_raises_on_duplicate_bytes(self):
        with self.assertRaises(ValueError):
            make_simple_trigger_map(target_byte=5, non_target_byte=5)

    def test_custom_entries_merged(self):
        m = make_simple_trigger_map(custom={99: "special"}, include_status=False)
        self.assertEqual(m[99], "special")

    def test_includes_all_status_events(self):
        m = make_simple_trigger_map(include_status=True)
        self.assertEqual(m[240], "Trial Started")
        self.assertEqual(m[241], "Trial Ends")
        self.assertEqual(m[242], "Training Complete")
        self.assertEqual(m[243], "Train Classifier")
        self.assertEqual(m[244], "Update Classifier")
        self.assertEqual(m[245], "Done with all RS collection")
        self.assertEqual(m[TARGET_BYTE], "target")
        self.assertEqual(m[NON_TARGET_BYTE], "non_target")
        self.assertEqual(len(m), 8)


class TestMakeP300TriggerMap(unittest.TestCase):
    def test_entry_count(self):
        m = make_p300_trigger_map(n_options=4)
        self.assertEqual(len(m), 6 + 4)

    def test_stimulus_marker_format(self):
        m = make_p300_trigger_map(n_options=3)
        self.assertEqual(m[1], "p300,s,3,-1,1")
        self.assertEqual(m[2], "p300,s,3,-1,2")
        self.assertEqual(m[3], "p300,s,3,-1,3")

    def test_stimulus_bytes_are_one_indexed(self):
        m = make_p300_trigger_map(n_options=2)
        self.assertIn(1, m)
        self.assertIn(2, m)
        self.assertNotIn(0, m)

    def test_status_codes_preserved(self):
        m = make_p300_trigger_map(n_options=1)
        for k, v in DEFAULT_TRIGGER_MAP.items():
            self.assertEqual(m[k], v)

    def test_raises_on_zero_options(self):
        with self.assertRaises(ValueError):
            make_p300_trigger_map(n_options=0)

    def test_raises_on_too_many_options(self):
        with self.assertRaises(ValueError):
            make_p300_trigger_map(n_options=240)


class TestMakeMiTriggerMap(unittest.TestCase):
    def test_entry_count(self):
        m = make_mi_trigger_map(n_classes=2, epoch_length=2.0)
        self.assertEqual(len(m), 6 + 2)

    def test_marker_contains_paradigm_prefix(self):
        m = make_mi_trigger_map(n_classes=3, epoch_length=1.5)
        self.assertIn("mi,", m[1])

    def test_raises_on_invalid_classes(self):
        with self.assertRaises(ValueError):
            make_mi_trigger_map(n_classes=-1, epoch_length=1.0)


class TestMakeSsvepTriggerMap(unittest.TestCase):
    def test_entry_count(self):
        m = make_ssvep_trigger_map(frequencies=[8.0, 10.0, 12.0], epoch_length=4.0)
        self.assertEqual(len(m), 6 + 3)

    def test_marker_contains_paradigm_prefix(self):
        m = make_ssvep_trigger_map(frequencies=[8.0, 12.0], epoch_length=3.0)
        self.assertIn("ssvep,", m[1])

    def test_raises_on_empty_frequencies(self):
        with self.assertRaises(ValueError):
            make_ssvep_trigger_map(frequencies=[], epoch_length=1.0)


class TestConstructionValidation(unittest.TestCase):
    def test_requires_stim_channel_name(self):
        with self.assertRaises(TypeError):
            EegStimTriggerMarkerSource(eeg_source=FakeEegSource())

    def test_rejects_empty_channel_name(self):
        with self.assertRaises(ValueError):
            EegStimTriggerMarkerSource(eeg_source=FakeEegSource(), stim_channel_name="")

    def test_rejects_non_string_channel_name(self):
        with self.assertRaises(ValueError):
            EegStimTriggerMarkerSource(eeg_source=FakeEegSource(), stim_channel_name=3)

    def test_rejects_invalid_detect_mode(self):
        with self.assertRaises(ValueError):
            EegStimTriggerMarkerSource(
                eeg_source=FakeEegSource(),
                stim_channel_name="STIM",
                detect_mode="invalid",
            )

    def test_rejects_non_eeg_source(self):
        with self.assertRaises(TypeError):
            EegStimTriggerMarkerSource(
                eeg_source="not_a_source", stim_channel_name="STIM"
            )

    def test_rejects_out_of_range_trigger_map_key(self):
        with self.assertRaises(ValueError):
            EegStimTriggerMarkerSource(
                eeg_source=FakeEegSource(),
                stim_channel_name="STIM",
                trigger_map={256: "too big"},
            )

    def test_is_both_eeg_and_marker_source(self):
        src = EegStimTriggerMarkerSource(
            eeg_source=FakeEegSource(), stim_channel_name="STIM"
        )
        self.assertIsInstance(src, EegSource)
        self.assertIsInstance(src, MarkerSource)

    def test_enabled_true_by_default(self):
        src = EegStimTriggerMarkerSource(
            eeg_source=FakeEegSource(), stim_channel_name="STIM"
        )
        self.assertTrue(src.enabled)

    def test_enabled_false_constructor(self):
        src = EegStimTriggerMarkerSource(
            eeg_source=FakeEegSource(), stim_channel_name="STIM", enabled=False
        )
        self.assertFalse(src.enabled)


class TestEegSourcePassthrough(unittest.TestCase):
    def setUp(self) -> None:
        self.eeg = FakeEegSource(channel_labels=["Fp1", "Fp2", "STIM"], fsample=512.0)
        self.src = EegStimTriggerMarkerSource(
            eeg_source=self.eeg, stim_channel_name="STIM"
        )

    def test_name(self):
        self.assertEqual(self.src.name, "FakeEegSource")

    def test_fsample(self):
        self.assertEqual(self.src.fsample, 512.0)

    def test_n_channels(self):
        self.assertEqual(self.src.n_channels, 3)

    def test_channel_labels(self):
        self.assertEqual(self.src.channel_labels, ["Fp1", "Fp2", "STIM"])

    def test_time_correction(self):
        self.assertEqual(self.src.time_correction(), 0.0)


class TestRiseModeDetection(unittest.TestCase):
    def test_single_trigger_on_zero_to_nonzero(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
            timestamps=[1.0, 2.0, 3.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map={1: "Trial Started"},
        )
        src.get_samples()
        markers, ts = src.get_markers()
        self.assertEqual(markers, [["Trial Started"]])
        self.assertAlmostEqual(ts[0], 2.0)

    def test_no_repeat_while_sustained_high(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            timestamps=[1.0, 2.0, 3.0],
        )
        src = EegStimTriggerMarkerSource(eeg_source=eeg, stim_channel_name="STIM")
        src.get_samples()
        _, ts = src.get_markers()
        self.assertEqual(ts, [])

    def test_multiple_rises_detected(self):
        eeg = FakeEegSource(
            samples=[
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 2],
                [0, 0, 0, 0],
            ],
            timestamps=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map={1: "Trial Started", 2: "Trial Ends"},
        )
        src.get_samples()
        markers, _ = src.get_markers()
        self.assertEqual(markers, [["Trial Started"], ["Trial Ends"]])

    def test_timestamp_matches_trigger_sample(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 240]],
            timestamps=[10.0, 10.004, 10.008],
        )
        src = EegStimTriggerMarkerSource(eeg_source=eeg, stim_channel_name="STIM")
        src.get_samples()
        _, ts = src.get_markers()
        self.assertAlmostEqual(ts[0], 10.008)


class TestChangeModeDetection(unittest.TestCase):
    def test_detects_any_nonzero_change(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]],
            timestamps=[1.0, 2.0, 3.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map={1: "Trial Started", 2: "Trial Ends"},
            detect_mode="change",
        )
        src.get_samples()
        markers, _ = src.get_markers()
        self.assertEqual(markers, [["Trial Started"], ["Trial Ends"]])

    def test_zero_change_not_emitted(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 0]],
            timestamps=[1.0, 2.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            detect_mode="change",
        )
        src.get_samples()
        _, ts = src.get_markers()
        self.assertEqual(ts, [])


class TestUnmappedTriggers(unittest.TestCase):
    def test_unmapped_ignored_by_default(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 99]],
            timestamps=[1.0, 2.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map={1: "Trial Started"},
        )
        src.get_samples()
        markers, ts = src.get_markers()
        self.assertEqual(ts, [])

    def test_unmapped_included_when_flag_set(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 99]],
            timestamps=[1.0, 2.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map={1: "Trial Started"},
            include_unmapped=True,
        )
        src.get_samples()
        markers, _ = src.get_markers()
        self.assertEqual(markers, [["trigger_99"]])


class TestChannelNameLookup(unittest.TestCase):
    def test_unknown_name_returns_no_markers(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [1, 0, 0, 1]],
            timestamps=[1.0, 2.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="DOES_NOT_EXIST",
            trigger_map={1: "Trial Started"},
        )
        src.get_samples()
        markers, timestamps = src.get_markers()
        self.assertEqual(markers, [[]])
        self.assertEqual(timestamps, [])

    def test_correct_name_detects_triggers(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 1]],
            timestamps=[1.0, 2.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map={1: "Trial Started"},
        )
        src.get_samples()
        markers, _ = src.get_markers()
        self.assertEqual(markers, [["Trial Started"]])

    def test_name_setter_clears_cache(self):
        eeg = FakeEegSource(
            channel_labels=["Ch0", "OLD", "NEW"],
            samples=[[0, 0, 0], [0, 1, 0]],
            timestamps=[1.0, 2.0],
        )
        eeg.add_batch(
            samples=[[0, 0, 0], [0, 0, 1]],
            timestamps=[3.0, 4.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="OLD",
            trigger_map={1: "Trial Started"},
        )
        src.get_samples()
        m1, _ = src.get_markers()
        self.assertEqual(m1, [["Trial Started"]])

        src.stim_channel_name = "NEW"
        src.get_samples()
        m2, _ = src.get_markers()
        self.assertEqual(m2, [["Trial Started"]])


class TestEnabledToggle(unittest.TestCase):
    def test_disabled_skips_trigger_detection(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 1]],
            timestamps=[1.0, 2.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg, stim_channel_name="STIM", enabled=False
        )
        src.get_samples()
        _, ts = src.get_markers()
        self.assertEqual(ts, [])

    def test_eeg_data_passes_through_when_disabled(self):
        original = [[1.0, 2.0, 3.0, 4.0]]
        eeg = FakeEegSource(samples=original, timestamps=[1.0])
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg, stim_channel_name="STIM", enabled=False
        )
        samples, ts = src.get_samples()
        self.assertEqual(samples, original)

    def test_reenable_resumes_detection(self):
        eeg = FakeEegSource(
            channel_labels=["Ch0", "Ch1", "Ch2", "STIM"],
            samples=[[0, 0, 0, 0], [0, 0, 0, 1]],
            timestamps=[1.0, 2.0],
        )
        eeg.add_batch(
            samples=[[0, 0, 0, 0], [0, 0, 0, 2]],
            timestamps=[3.0, 4.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map={1: "Trial Started", 2: "Trial Ends"},
            enabled=False,
        )
        src.get_samples()
        _, ts1 = src.get_markers()
        self.assertEqual(ts1, [])

        src.enabled = True
        src.get_samples()
        markers, ts2 = src.get_markers()
        self.assertEqual(len(ts2), 1)
        self.assertEqual(markers[0], ["Trial Ends"])


class TestEdgeCases(unittest.TestCase):
    def test_empty_samples_no_error(self):
        eeg = FakeEegSource(samples=[], timestamps=[])
        src = EegStimTriggerMarkerSource(eeg_source=eeg, stim_channel_name="STIM")
        samples, ts = src.get_samples()
        self.assertEqual(samples, [])
        markers, mts = src.get_markers()
        self.assertEqual(mts, [])

    def test_get_markers_drains_queue(self):
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 240]],
            timestamps=[1.0, 2.0],
        )
        src = EegStimTriggerMarkerSource(eeg_source=eeg, stim_channel_name="STIM")
        src.get_samples()
        _, ts1 = src.get_markers()
        self.assertEqual(len(ts1), 1)
        _, ts2 = src.get_markers()
        self.assertEqual(ts2, [])

    def test_get_markers_empty_before_any_samples(self):
        src = EegStimTriggerMarkerSource(
            eeg_source=FakeEegSource(), stim_channel_name="STIM"
        )
        markers, ts = src.get_markers()
        self.assertEqual(markers, [[]])
        self.assertEqual(ts, [])

    def test_p300_map_integration(self):
        tmap = make_p300_trigger_map(n_options=4)
        eeg = FakeEegSource(
            samples=[[0, 0, 0, 0], [0, 0, 0, 1]],
            timestamps=[1.0, 2.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map=tmap,
        )
        src.get_samples()
        markers, _ = src.get_markers()
        self.assertEqual(markers, [["p300,s,4,-1,1"]])

    def test_simple_map_with_status_integration(self):
        # status events (240+) + target/non-target on low bytes
        tmap = make_simple_trigger_map(include_status=True)
        eeg = FakeEegSource(
            samples=[
                [0, 0, 0, 0],
                [0, 0, 0, 240],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 241],
            ],
            timestamps=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        src = EegStimTriggerMarkerSource(
            eeg_source=eeg,
            stim_channel_name="STIM",
            trigger_map=tmap,
        )
        src.get_samples()
        markers, ts = src.get_markers()
        self.assertEqual(
            markers,
            [["Trial Started"], ["target"], ["non_target"], ["Trial Ends"]],
        )
        self.assertEqual(ts, [2.0, 4.0, 6.0, 8.0])

    def test_underlying_source_error_handled(self):
        eeg = FakeEegSource()
        eeg.get_samples = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        src = EegStimTriggerMarkerSource(eeg_source=eeg, stim_channel_name="STIM")
        samples, ts = src.get_samples()
        self.assertEqual(samples, [[]])
        self.assertEqual(ts, [])


if __name__ == "__main__":
    unittest.main()
