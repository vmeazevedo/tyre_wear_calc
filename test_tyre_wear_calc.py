import numpy as np
import unittest
from tyre_wear_calc import train_model, predict_laps

class TestRandomForestModel(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[30.5, 25.0], [29.8, 28.0], [31.2, 30.0]])
        self.y_train = np.array([20, 25, 18])
        self.user_input = [30.3, 28.5]
        self.model = train_model(self.X_train, self.y_train)

    def test_model_training(self):
        self.assertIsNotNone(self.model)

    def test_predict_laps(self):
        predicted_laps = predict_laps(self.model, self.user_input)
        self.assertIsInstance(predicted_laps, float)
        self.assertGreaterEqual(predicted_laps, 0)

if __name__ == '__main__':
    unittest.main()
