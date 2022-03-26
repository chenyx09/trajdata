import unittest

from avdata import AgentType, UnifiedDataset


class TestDatasetSizes(unittest.TestCase):
    def test_two_datasets(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"], centric="agent"
        )

        self.assertEqual(len(dataset), 1_924_196)

    def test_splits(self):
        dataset = UnifiedDataset(desired_data=["nusc_mini-mini_train"], centric="agent")

        self.assertEqual(len(dataset), 10_598)

        dataset = UnifiedDataset(desired_data=["nusc_mini-mini_val"], centric="agent")

        self.assertEqual(len(dataset), 4_478)

    def test_geography(self):
        dataset = UnifiedDataset(desired_data=["singapore"], centric="agent")

        self.assertEqual(len(dataset), 8_965)

        dataset = UnifiedDataset(desired_data=["boston"], centric="agent")

        self.assertEqual(len(dataset), 6_111)

        dataset = UnifiedDataset(desired_data=["palo_alto"], centric="agent")

        self.assertEqual(len(dataset), 1_909_120)

        dataset = UnifiedDataset(desired_data=["boston", "palo_alto"], centric="agent")

        self.assertEqual(len(dataset), 1_915_231)

    def test_exclusion(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            no_types=[AgentType.UNKNOWN],
        )

        self.assertEqual(len(dataset), 610_074)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            no_types=[AgentType.UNKNOWN, AgentType.BICYCLE],
        )

        self.assertEqual(len(dataset), 603_089)

    def test_inclusion(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            only_types=[AgentType.VEHICLE],
        )

        self.assertEqual(len(dataset), 554_880)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            only_types=[AgentType.VEHICLE, AgentType.UNKNOWN],
        )

        self.assertEqual(len(dataset), 1_869_002)

    def test_history_future(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            history_sec=(0.1, 2.0),
            future_sec=(0.1, 2.0),
        )

        self.assertEqual(len(dataset), 1_685_896)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            history_sec=(0.5, 2.0),
            future_sec=(0.5, 3.0),
        )

        self.assertEqual(len(dataset), 1_155_704)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            history_sec=(0.5, 1.0),
            future_sec=(0.5, 0.7),
        )

        self.assertEqual(len(dataset), 1_155_704)


if __name__ == "__main__":
    unittest.main()
