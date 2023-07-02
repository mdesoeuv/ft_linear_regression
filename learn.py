from pathlib import Path
import csv
import copy
import argparse


class RegressionError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class UniVariableLinearRegression:
    """Linear Regression with one variable (feature)."""

    @staticmethod
    def normalize(x: float, x_min=0, x_max=1):
        return (x - x_min) / (x_max - x_min)

    def normalize_dataset(self):
        self.normalized_dataset = copy.deepcopy(self.dataset)
        for entry in self.normalized_dataset:
            entry[self.feature] = self.normalize(
                float(entry[self.feature]), self.feature_min, self.feature_max
            )

    @staticmethod
    def predict(feature: float, theta0: float, theta1: float):
        return theta0 + (feature * theta1)

    def calculate_cost(self):
        cost = 0
        for feature, target in zip(self.norm_features, self.y):
            diff = self.predict(feature, self.theta0, self.theta1) - target
            cost += (diff * diff) / (2 * self.dataset_size)
        return cost

    def calculate_rmse(self):
        return self.calculate_cost()**0.5

    def derivative_sum(self):
        sum_dt0 = 0
        sum_dt1 = 0
        for feature, target in zip(self.norm_features, self.y):
            sum_dt0 += (
                self.predict(feature, self.theta0, self.theta1) - target
            ) / self.dataset_size
            sum_dt1 += (
                feature
                * (self.predict(feature, self.theta0, self.theta1) - target)
                / self.dataset_size
            )
        return (self.learning_rate * sum_dt0, self.learning_rate * sum_dt1)

    def r_square(self):
        SSres = 0
        SStot = 0
        target_avg = sum(self.y) / len(self.y)
        for feature, target in zip(self.norm_features, self.y):
            error = target - self.predict(feature, self.theta0, self.theta1)
            SSres += error**2
            SStot += (target - target_avg) ** 2
        return 1 - (SSres / SStot)

    def gradient_descent(self):
        min_reached = False
        while not min_reached and self.iter < self.max_iterations:
            self.iter += 1
            cost = self.calculate_cost()
            self.iters_costs.append(
                {"iterations": self.iter, "costs": cost, "rmse": cost**0.5}
            )
            tmp_theta0, tmp_theta1 = self.derivative_sum()
            if round(self.theta0 - tmp_theta0, 2) == round(self.theta0, 2) \
                and round(
                self.theta1 - tmp_theta1, 2
            ) == round(self.theta1, 2):
                print(f"Minimum reached in {self.iter} iterations :")
                print(f"learning_rate={self.learning_rate}")
                print(f"theta0={self.theta0 - tmp_theta0}")
                print(f"theta1={self.theta1 - tmp_theta1}\n")
                min_reached = True
            self.theta0 = self.theta0 - tmp_theta0
            self.theta1 = self.theta1 - tmp_theta1

    def print_analytics(self):
        print("Analytics :")
        print(f"Learning Rate = {self.learning_rate}")
        print(f"Iterations = {self.iter}")
        print(f"R2 = {self.r_square()}")
        print(f"RMSE = {self.calculate_rmse()}")
        print(f"Theta0 = {self.theta0}")
        print(f"Theta1 = {self.theta1}\n")

    def __init__(
        self,
        dataset: list,
        feature: str,
        target: str,
        theta0=0,
        theta1=0,
        learning_rate=0.5,
        max_iterations=10000,
    ):
        self.dataset = dataset
        self.feature = feature
        self.target = target
        self.dataset_size = len(dataset)
        if not self.dataset_size:
            raise RegressionError("Invalid dataset size.")
        try:
            self.x = [float(entry[feature]) for entry in self.dataset]
            self.y = [float(entry[target]) for entry in self.dataset]
        except KeyError as e:
            raise RegressionError("Invalid dataset.") from e
        self.theta0 = theta0
        self.theta1 = theta1
        self.feature_min = min(self.x)
        self.feature_max = max(self.x)
        self.norm_features = [
            self.normalize(feature, self.feature_min, self.feature_max)
            for feature in self.x
        ]
        self.normalize_dataset()
        self.learning_rate = learning_rate
        self.iter = 0
        self.iters_costs = []
        self.max_iterations = max_iterations


def read_csv(filepath: str):
    if not Path(filepath).exists() or Path(filepath).is_dir():
        print("File does not exist or is a directory")
        exit(1)
    try:
        dataset = []
        with open(filepath, mode="r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset.append(row)
        return dataset
    except Exception as err:
        print(f"CSV file error : {err}")
        exit(1)


def write_to_csv(filepath: str, data: list, fields: list):
    if Path(filepath).is_dir():
        print("File is a directory")
        exit(1)

    try:
        with open(filepath, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    except Exception as err:
        print(f"CSV file error : {err}")
        exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="learn", usage="learn -f [dataset csv file]"
        )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        action="store",
        required=True,
        help="path of the csv file containing the training dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    dataset = read_csv(args.file)

    try:
        regression = UniVariableLinearRegression(
            dataset=dataset,
            feature="km",
            target="price",
            learning_rate=0.5,
            theta0=0,
            theta1=0,
        )
    except RegressionError as err:
        print(err)
        exit(1)

    regression.gradient_descent()
    regression.print_analytics()

    write_to_csv(
        "costs_iterations.csv",
        regression.iters_costs,
        fields=["iterations", "costs", "rmse"]
    )
    write_to_csv(
        "regression_parameters.csv",
        [
            {
                "theta0": regression.theta0,
                "theta1": regression.theta1,
                "x_min": regression.feature_min,
                "x_max": regression.feature_max,
                "learning_rate": regression.learning_rate,
            }
        ],
        fields=["theta0", "theta1", "x_min", "x_max", "learning_rate"],
    )
