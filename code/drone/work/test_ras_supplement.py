import unittest
import torch
import ras_supplement as R


class ProtocolTest(unittest.TestCase):
    def test_only_obstacle_penalty_changes(self):
        s = torch.zeros(3, 13)
        s[:, :3] = torch.tensor([[0., 0., .55], [0., 0., 1.], [0., 0., 2.]])
        u = torch.tensor([[10., .1, .2, .3]]).expand(3, 4)
        goal = torch.tensor([1., 0., .7])
        base = R.ORIGINAL_COST(s, u, goal)
        barrier = 50 * torch.nn.functional.softplus(-10 * (R.Dr.clearance(s[:, :3]) - .1)) / 10
        self.assertIs(R.penalty_cost(1), R.ORIGINAL_COST)
        for scale in (.5, 2.):
            torch.testing.assert_close(R.penalty_cost(scale)(s, u, goal), base + (scale - 1) * barrier)

    def test_reach_then_crash_is_not_safe(self):
        goal = torch.tensor([0., 0., .7])
        s = torch.zeros(3, 13)
        s[:, 2] = torch.tensor([1.5, .7, .4])
        row = R.outcome(s, goal)
        self.assertTrue(row["reach"])
        self.assertTrue(row["crash"])
        self.assertFalse(row["safe"])

    def test_unreached_safe_flight_is_not_completion(self):
        s = torch.zeros(3, 13)
        s[:, 2] = 2.
        row = R.outcome(s, torch.tensor([0., 0., .7]))
        self.assertFalse(row["crash"])
        self.assertFalse(row["safe"])


if __name__ == "__main__":
    unittest.main()
