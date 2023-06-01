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


	@staticmethod
	def normalize_dataset(dataset: list, feature: str, feature_min: float, feature_max: float):

		cpy_data = copy.deepcopy(dataset)
		for entry in cpy_data:
			entry[feature] = UniVariableLinearRegression.normalize(float(entry[feature]), feature_min, feature_max)
		return cpy_data
	
	def calculate_cost(self):
		
		cost = 0

		for feature, target in zip(self.norm_features, self.y):
			diff = predict(feature, self.theta0, self.theta1) - target
			cost += (diff * diff) / (2 * self.dataset_size)
		return cost 


	def __init__(self, dataset: list, feature: str, target: str):
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
		self.feature_min = min(self.x)
		self.feature_max = max(self.x)
		self.norm_features = [self.normalize(feature, self.feature_min, self.feature_max) for feature in self.x]
		self.normalized_dataset = self.normalize_dataset(self.dataset, self.feature, self.feature_min, self.feature_max)


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





def derivative_sum(dataset: list, theta0: float, theta1: float):
	sum_dt0 = 0
	sum_dt1 = 0
	m = len(dataset)
	for entry in dataset:
		mileage = float(entry['km'])
		price = float(entry['price'])
		sum_dt0 += (predict(mileage, theta0, theta1) - price) / m
		sum_dt1 += mileage * (predict(mileage, theta0, theta1) - price) / m

	return (sum_dt0, sum_dt1)


def r_square(normalized_dataset: list, theta0: float, theta1: float):
	mileages = [float(entry['km']) for entry in normalized_dataset]
	prices = [float(entry['price']) for entry in normalized_dataset]
	SSres = 0
	SStot = 0
	price_avg = sum(prices) / len(prices)
	for mileage, price in zip(mileages, prices):
		error = price - predict(mileage, theta0, theta1)
		SSres += error**2
		SStot += (price - price_avg)**2
	return 1 - (SSres / SStot)




if __name__ == "__main__":
	
	if len(sys.argv) < 2:
		print("usage: learn [csvfile]")
		exit(1)
	
	dataset = read_csv(sys.argv[1])

	try:
		reg = UniVariableLinearRegression(
			dataset=dataset,
			feature="km",
			target="price"
		)
	except RegressionError as e:
		print(e)
		exit(1)
	
	min_reached = False
	iter_count = 0
	iters_costs = []

	while not min_reached and iter_count < MAX_ITERATIONS:
		iter_count += 1
		cost = reg.calculate_cost()
		iters_costs.append({"iterations": iter_count, "costs": cost, "rmse": math.sqrt(cost)})
		derivate_theta0, derivate_theta1 = derivative_sum(reg.normalized_dataset, reg.theta0, reg.theta1)
		tmp_theta0 = (LEARNING_RATE * derivate_theta0)
		tmp_theta1 = (LEARNING_RATE * derivate_theta1)
		if round(reg.theta0 - tmp_theta0, 2) == round(reg.theta0, 2) and round(reg.theta1 - tmp_theta1, 2) == round(reg.theta1, 2):
			print(f"Minimum reached in {iter_count} iterations : theta0={reg.theta0 - tmp_theta0}, theta1={reg.theta1 - tmp_theta1}")
			min_reached = True
		reg.theta0 = reg.theta0 - tmp_theta0
		reg.theta1 = reg.theta1 - tmp_theta1

	write_to_csv("costs_iterations.csv", iters_costs, ["iterations", "costs", "rmse"])
	write_to_csv(
		"regression_results.csv",
		[{
			"theta0": reg.theta0,
			"theta1": reg.theta1,
			"x_min": reg.feature_min,
			"x_max": reg.feature_max
			}],
		fields=["theta0", "theta1", "x_min", "x_max"]
		)

