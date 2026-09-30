"""Small end-to-end checks for the warehouse and model contracts."""

import unittest

import numpy as np

from src.analytics import executive_kpis
from src.classification_engine import classify_order, train_decision_tree
from src.regression_engine import predict_basket_price, train_regressors
from src.warehouse_engine import build_star_schema, joined_sales, roll_up, slice_dice


class PlatformSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sales = joined_sales()

    def test_warehouse_totals_and_filters(self):
        schema = build_star_schema()
        self.assertEqual(len(self.sales), 3000)
        self.assertEqual(len(schema["Fact_Sales"]), len(self.sales))
        self.assertTrue(np.allclose(self.sales["total_price"], self.sales["quantity"] * self.sales["unit_price"]))
        monthly = roll_up(self.sales, "Month", ["pizza_category"])
        self.assertAlmostEqual(monthly["revenue"].sum(), executive_kpis(self.sales)["Gross Revenue"])
        rush = slice_dice(self.sales, weekend_only=True, peak_only=True)
        self.assertTrue(rush["is_weekend"].all())
        self.assertTrue(rush["peak_period"].isin(["Lunch", "Dinner"]).all())

    def test_model_inference(self):
        regression = train_regressors(self.sales)
        self.assertEqual(len(regression.metrics), 3)
        basket = predict_basket_price(regression.models["Random Forest Regressor"], 2, 12, 2, 1, "M", "Classic", 19)
        self.assertAlmostEqual(basket["arithmetic_total"], 28)
        self.assertGreater(basket["model_prediction"], 0)
        tree = train_decision_tree(self.sales)
        order = classify_order(tree.model, 2, 15, 19, "M", "Classic")
        self.assertIn(order["tier"], {"Budget", "Standard", "Premium"})
        self.assertAlmostEqual(sum(order["probabilities"].values()), 1)


if __name__ == "__main__":
    unittest.main()
