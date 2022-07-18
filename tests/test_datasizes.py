import random
import unittest

from trajdata import AgentType, UnifiedDataset


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

    def test_prediction_inclusion(self):
        unfiltered_dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
        )

        filtered_dataset = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            only_predict=[AgentType.VEHICLE],
        )

        self.assertGreaterEqual(len(unfiltered_dataset), len(filtered_dataset))

        for _ in range(100):
            sample_idx = random.randint(0, len(filtered_dataset) - 1)
            self.assertEqual(filtered_dataset[sample_idx].agent_type, AgentType.VEHICLE)

        filtered_dataset2 = UnifiedDataset(
            desired_data=["nusc_mini", "lyft_sample"],
            centric="agent",
            only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        )

        for _ in range(100):
            sample_idx = random.randint(0, len(filtered_dataset2) - 1)
            self.assertIn(
                filtered_dataset2[sample_idx].agent_type, filtered_dataset2.only_predict
            )

        self.assertGreaterEqual(len(filtered_dataset2), len(filtered_dataset))

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

    def test_interpolation(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_train"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_types=[AgentType.VEHICLE],
            incl_robot_future=False,
            incl_map=False,
            map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=0,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self.assertEqual(len(dataset), 11_046)


if __name__ == "__main__":
    unittest.main()
