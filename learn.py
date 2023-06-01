from pathlib import Path
import csv
import sys
import copy
import math
from dataclasses import dataclass


LEARNING_RATE = 0.1
MAX_ITERATIONS = 10000


class RegressionError(Exception):
	def __init__(self, msg):
		super().__init__(msg)


@dataclass
class RegressionParameters:
	theta0: float = 0
	theta1: float = 0
	feature_min: float = 0
	feature_max: float = 1


class UniVariableLinearRegression(RegressionParameters):

	@staticmethod	
	def normalize(x: float, x_min = 0, x_max = 1):
		return (x - x_min) / (x_max - x_min)

	def normalize_dataset(self):
		self.normalized_dataset = copy.deepcopy(self.dataset)
		for entry in self.normalized_dataset:
			entry[self.feature] = self.normalize(
				float(entry[self.feature]),
				self.feature_min,
				self.feature_max
				)
	
	def calculate_cost(self):
		cost = 0
		for feature, target in zip(self.norm_features, self.y):
			diff = predict(feature, self.theta0, self.theta1) - target
			cost += (diff * diff) / (2 * self.dataset_size)
		return cost 
	
	def calculate_rmse(self):
		return math.sqrt(self.calculate_cost())

	def derivative_sum(self):
		sum_dt0 = 0
		sum_dt1 = 0
		for feature, target in zip(self.norm_features, self.y):
			sum_dt0 += (predict(feature, self.theta0, self.theta1) - target) / self.dataset_size
			sum_dt1 += feature * (predict(feature, self.theta0, self.theta1) - target) / self.dataset_size
		return (self.learning_rate * sum_dt0, self.learning_rate * sum_dt1)

	def r_square(self):
		SSres = 0
		SStot = 0
		target_avg = sum(self.y) / len(self.y)
		for feature, target in zip(self.norm_features, self.y):
			error = target - predict(feature, self.theta0, self.theta1)
			SSres += error**2
			SStot += (target - target_avg)**2
		return 1 - (SSres / SStot)
	
	def gradient_descent(self):			
		min_reached = False
		while not min_reached and regression.iter < MAX_ITERATIONS:
			regression.iter += 1
			cost = regression.calculate_cost()
			self.iters_costs.append(
				{"iterations": regression.iter, "costs": cost, "rmse": math.sqrt(cost)}
				)
			tmp_theta0, tmp_theta1 = regression.derivative_sum()
			if round(regression.theta0 - tmp_theta0, 2) == round(regression.theta0, 2) and round(regression.theta1 - tmp_theta1, 2) == round(regression.theta1, 2):
				print(f"Minimum reached in {regression.iter} iterations : theta0={regression.theta0 - tmp_theta0}, theta1={regression.theta1 - tmp_theta1}")
				min_reached = True
			regression.theta0 = regression.theta0 - tmp_theta0
			regression.theta1 = regression.theta1 - tmp_theta1

	def __init__(self, dataset: list, feature: str, target: str,
	      theta0 = 0, theta1 = 0,learning_rate = 0):
		self.dataset = dataset
		self.feature = feature
		self.target = target
		self.dataset_size = len(dataset)
		self.iters_costs = []
		if not self.dataset_size:
			raise RegressionError("Invalid dataset size.")
		try:
			self.x = [float(entry[feature]) for entry in self.dataset]
			self.y = [float(entry[target]) for entry in self.dataset]
		except KeyError as e:
			raise RegressionError("Invalid dataset.") from e
		
		super().__init__(
			theta0=theta0,
			theta1=theta1,
			feature_min=min(self.x),
			feature_max=max(self.x)
		)
		self.norm_features = [self.normalize(feature, self.feature_min, self.feature_max) for feature in self.x]
		self.normalize_dataset()
		self.learning_rate = learning_rate
		self.iter = 0


def read_csv(filepath: str):
	if not Path(filepath).exists() or Path(filepath).is_dir():
		print("File does not exist or is a directory")
		exit(1)

	dataset = []
	with open(filepath, mode='r', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			dataset.append(row)
	return dataset


def write_to_csv(filepath: str, data: list, fields: list):

	if Path(filepath).is_dir():
		print("File is a directory")
		exit(1)

	with open(filepath, mode='w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fields)
		writer.writeheader()
		writer.writerows(data)


def predict(feature: float, theta0: float, theta1: float):
	return theta0 + (feature * theta1)


if __name__ == "__main__":
	
	if len(sys.argv) < 2:
		print("usage: learn [csvfile]")
		exit(1)
	
	dataset = read_csv(sys.argv[1])

	try:
		regression = UniVariableLinearRegression(
			dataset=dataset,
			feature="km",
			target="price",
			learning_rate=LEARNING_RATE
		)
	except RegressionError as e:
		print(e)
		exit(1)

	regression.gradient_descent()

	write_to_csv("costs_iterations.csv", regression.iters_costs, ["iterations", "costs", "rmse"])
	write_to_csv(
		"regression_parameters.csv",
		[{
			"theta0": regression.theta0,
			"theta1": regression.theta1,
			"x_min": regression.feature_min,
			"x_max": regression.feature_max
			}],
		fields=["theta0", "theta1", "x_min", "x_max"]
		)

