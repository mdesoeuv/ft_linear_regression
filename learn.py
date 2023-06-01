from pathlib import Path
import csv
import sys
import copy
import math


LEARNING_RATE = 0.1
MAX_ITERATIONS = 10000


class UniVariableLinearRegression:

	@staticmethod	
	def normalize(x: float, x_min = 0, x_max = 1):

		return (x - x_min) / (x_max - x_min)


	@staticmethod
	def normalize_dataset(dataset: list, feature: str, feature_min: float, feature_max: float):

		cpy_data = copy.deepcopy(dataset)
		for entry in cpy_data:
			entry[feature] = UniVariableLinearRegression.normalize(float(entry[feature]), feature_min, feature_max)
		return cpy_data
	

	def __init__(self, dataset: list, feature: str, target: str):
		self.dataset = dataset
		self.feature = feature
		self.target = target
		self.dataset_size = len(dataset)
		self.x = [float(entry[feature] for entry in self.dataset)]
		self.y = [float(entry[target] for entry in self.dataset)]
		self.x_min = min(self.x)
		self.x_max = max(self.y)
		self.normalized_dataset = self.normalize_dataset(self.dataset, self.x_min, self.x_max)
		self.theta0 = 0
		self.theta1 = 1


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




def estimate_price(mileage: float, theta0: float, theta1: float):
	return theta0 + (mileage * theta1)


def calculate_cost(dataset: list, theta0: float, theta1: float):
	cost = 0
	m = len(dataset)
	if not m:
		print("Invalid dataset.")
		exit(1)
	
	for entry in dataset:
		try:
			mileage = float(entry['km'])
			price = float(entry['price'])
		except KeyError:
			print("Invalid dataset.")
			exit(1)
		diff = estimate_price(mileage, theta0, theta1) - price
		cost += (diff * diff) / (2 * m)
	return cost 


def derivative_sum(dataset: list, theta0: float, theta1: float):
	sum_dt0 = 0
	sum_dt1 = 0
	m = len(dataset)
	for entry in dataset:
		mileage = float(entry['km'])
		price = float(entry['price'])
		sum_dt0 += (estimate_price(mileage, theta0, theta1) - price) / m
		sum_dt1 += mileage * (estimate_price(mileage, theta0, theta1) - price) / m

	return (sum_dt0, sum_dt1)


def r_square(normalized_dataset: list, theta0: float, theta1: float):
	mileages = [float(entry['km']) for entry in normalized_dataset]
	prices = [float(entry['price']) for entry in normalized_dataset]
	SSres = 0
	SStot = 0
	price_avg = sum(prices) / len(prices)
	for mileage, price in zip(mileages, prices):
		error = price - estimate_price(mileage, theta0, theta1)
		SSres += error**2
		SStot += (price - price_avg)**2
	return 1 - (SSres / SStot)




if __name__ == "__main__":
	
	if len(sys.argv) < 2:
		print("usage: learn [csvfile]")
		exit(1)
	
	dataset = read_csv(sys.argv[1])
	try:
		mileages = [float(entry['km']) for entry in dataset]
	except KeyError:
		print("Invalid dataset.")
		exit(1)
	
	mileage_min = min(mileages)
	mileage_max = max(mileages)

	norm_dataset = UniVariableLinearRegression.normalize_dataset(dataset, "km", mileage_min, mileage_max)
	
	theta0 = 0
	theta1 = 0

	min_reached = False
	iter_count = 0
	iters_costs = []

	while not min_reached and iter_count < MAX_ITERATIONS:
		iter_count += 1
		cost = calculate_cost(norm_dataset, theta0, theta1)
		iters_costs.append({"iterations": iter_count, "costs": cost, "rmse": math.sqrt(cost)})
		derivate_theta0, derivate_theta1 = derivative_sum(norm_dataset, theta0, theta1)
		tmp_theta0 = (LEARNING_RATE * derivate_theta0)
		tmp_theta1 = (LEARNING_RATE * derivate_theta1)
		if round(theta0 - tmp_theta0, 2) == round(theta0, 2) and round(theta1 - tmp_theta1, 2) == round(theta1, 2):
			print(f"Minimum reached in {iter_count} iterations : theta0={theta0 - tmp_theta0}, theta1={theta1 - tmp_theta1}")
			min_reached = True
		theta0 = theta0 - tmp_theta0
		theta1 = theta1 - tmp_theta1

	write_to_csv("costs_iterations.csv", iters_costs, ["iterations", "costs", "rmse"])
	write_to_csv(
		"regression_results.csv",
		[{
			"theta0": theta0,
			"theta1": theta1,
			"x_min": mileage_min,
			"x_max": mileage_max
			}],
		fields=["theta0", "theta1", "x_min", "x_max"]
		)

