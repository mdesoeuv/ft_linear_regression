from pathlib import Path
import csv
import sys
import copy
import math
from dataclasses import dataclass


LEARNING_RATE_1 = 0.5
LEARNING_RATE_2 = 0.1
LEARNING_RATE_3 = 0.01
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
		while not min_reached and self.iter < MAX_ITERATIONS:
			self.iter += 1
			cost = self.calculate_cost()
			self.iters_costs.append(
				{"iterations": self.iter, "costs": cost, "rmse": math.sqrt(cost)}
				)
			tmp_theta0, tmp_theta1 = self.derivative_sum()
			if round(self.theta0 - tmp_theta0, 2) == round(self.theta0, 2) and round(self.theta1 - tmp_theta1, 2) == round(self.theta1, 2):
				print(f"Minimum reached in {self.iter} iterations : theta0={self.theta0 - tmp_theta0}, theta1={self.theta1 - tmp_theta1}")
				min_reached = True
			self.theta0 = self.theta0 - tmp_theta0
			self.theta1 = self.theta1 - tmp_theta1

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
		regression1 = UniVariableLinearRegression(
			dataset=dataset,
			feature="km",
			target="price",
			learning_rate=LEARNING_RATE_1
		)
		regression2 = UniVariableLinearRegression(
			dataset=dataset,
			feature="km",
			target="price",
			learning_rate=LEARNING_RATE_2
		)
		regression3 = UniVariableLinearRegression(
			dataset=dataset,
			feature="km",
			target="price",
			learning_rate=LEARNING_RATE_3
		)
	except RegressionError as e:
		print(e)
		exit(1)

	regression1.gradient_descent()
	regression2.gradient_descent()
	regression3.gradient_descent()

	write_to_csv("costs_iterations1.csv", regression1.iters_costs, ["iterations", "costs", "rmse"])
	write_to_csv("costs_iterations2.csv", regression2.iters_costs, ["iterations", "costs", "rmse"])
	write_to_csv("costs_iterations3.csv", regression3.iters_costs, ["iterations", "costs", "rmse"])
	write_to_csv(
		"regression_parameters.csv",
		[{
			"theta0": regression1.theta0,
			"theta1": regression1.theta1,
			"x_min": regression1.feature_min,
			"x_max": regression1.feature_max
			}],
		fields=["theta0", "theta1", "x_min", "x_max"]
		)

