import random

from locust import HttpUser, between, task


class ApiUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self):
        self.last_id = None

    @task(9)
    def predict(self):
        payload = {
            "qty_lag_1": random.uniform(0, 500),
            "qty_lag_2": random.uniform(0, 500),
            "qty_lag_4": random.uniform(0, 500),
            "qty_roll_mean": random.uniform(0, 500),
            "price_lag_1": random.uniform(0.5, 10),
            "week_of_year": random.randint(1, 52),
            "month": random.randint(1, 12),
        }
        with self.client.post("/predict", json=payload, catch_response=True) as r:
            if r.status_code == 200:
                self.last_id = r.json()["prediction_id"]
            else:
                r.failure(f"HTTP {r.status_code}")

    @task(1)
    def feedback_or_health(self):
        if self.last_id and random.random() < 0.5:
            self.client.post("/feedback", json={"prediction_id": self.last_id,
                                                "actual_quantity": random.uniform(0, 500)})
        else:
            self.client.get("/health")
