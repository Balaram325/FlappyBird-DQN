import unittest
from compare_runs import compare


class ComparisonTests(unittest.TestCase):
    def test_measured_reward_difference(self):
        dqn = {"episodes":100,"seed":123,"mean_reward":-0.27,
               "median_reward":-0.30,"std_reward":0.69,"max_reward":1.50}
        random = {"episodes":100,"seed":123,"mean_reward":-7.64,
                  "median_reward":-7.50,"std_reward":0.87,"max_reward":-4.50}
        result = compare(dqn, random)
        self.assertAlmostEqual(result["mean_reward_difference"],7.37)

    def test_mismatch_is_rejected(self):
        a={"episodes":100,"seed":123}
        b={"episodes":99,"seed":123}
        with self.assertRaises(ValueError):
            compare(a,b)


if __name__ == "__main__":
    unittest.main()
