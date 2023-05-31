from pathlib import Path
import csv
import sys
import matplotlib.pyplot as plt


LEARNING_RATE = 0.1
MAX_ITERATIONS = 10000



def read_data(filepath: str):
	if not Path(filepath).exists() or Path(filepath).is_dir():
		print("File does not exist or is a directory")
		exit(1)

	dataset = []
	with open(filepath, mode='r', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			dataset.append(row)
	return dataset


def normalize_dataset(dataset: list):
	mileages = [float(entry['km']) for entry in dataset]
	# prices = [float(entry['price']) for entry in dataset]

	mileage_min = min(mileages)
	mileage_max = max(mileages)
	mileage_avg = sum(mileages) / len(mileages)
	print(f"mileage_min: {mileage_min}, mileage_max: {mileage_max}, mileage_avg: {mileage_avg}")

	# price_min = min(prices)
	# price_max = max(prices)

	for entry in dataset:
		entry['km'] = (float(entry['km'])) / (mileage_max)
		# entry['price'] = (float(entry['price']) - price_min) / (price_max - price_min)

	print(dataset)
	return dataset


def estimate_price(mileage: float, theta0: float, theta1: float):
	return theta0 + (mileage * theta1)


def calculate_cost(dataset: list, theta0: float, theta1: float):
	cost = 0
	for entry in dataset:
		mileage = int(entry['km'])
		price = int(entry['price'])
		diff = estimate_price(mileage, theta0, theta1) - price
		cost += (diff * diff)
	return cost / (2 * len(dataset))

def derivative_sum(dataset: list, theta0: float, theta1: float):
	sum_dt0 = 0
	sum_dt1 = 0
	for entry in dataset:
		mileage = int(entry['km'])
		price = int(entry['price'])
		# print(f"mileage: {mileage}, price: {price}")
		sum_dt0 += estimate_price(mileage, theta0, theta1) - price
		sum_dt1 += mileage * (estimate_price(mileage, theta0, theta1) - price)

	sum_dt0 = sum_dt0 / len(dataset)
	sum_dt1 = sum_dt1 / len(dataset)
	return (sum_dt0, sum_dt1)


if __name__ == "__main__":
	
	if len(sys.argv) < 2:
		print("usage: learn [csvfile]")
		exit(1)
	
	dataset = read_data(sys.argv[1])
	norm_dataset = normalize_dataset(dataset)
	m = len(norm_dataset)
	
	# random.seed(datetime.datetime.now().timestamp())
	theta0 = 0
	theta1 = 0
	print(f"starting with theta0: {theta0}, theta1: {theta1}")

	min_reached = False
	iter_count = 0
	costs = []
	iters = []
	while iter_count < MAX_ITERATIONS:
		iter_count += 1
		iters.append(iter_count)
		cost = calculate_cost(norm_dataset, theta0, theta1)
		print(f"cost : {cost}")
		costs.append(cost)
		derivate_theta0, derivate_theta1 = derivative_sum(norm_dataset, theta0, theta1)
		print(f"dt0: {derivate_theta0}, dt1: {derivate_theta1}")
		# print(derivate_theta0, derivate_theta1)
		tmp_theta0 = (LEARNING_RATE * derivate_theta0)
		tmp_theta1 = (LEARNING_RATE * derivate_theta1)
		# if round(tmp_theta0, 5) == round(theta0, 5) and round(tmp_theta1, 5) == round(theta1, 5):
		# 	print(f"Minimum reached in {iter_count} iterations : theta0={theta0}, theta1={theta1}")
		# 	min_reached = True
		theta0 = theta0 - tmp_theta0
		theta1 = theta1 - tmp_theta1
		print(f"theta0: {theta0}, theta1: {theta1}")

	fig, ax = plt.subplots()
	ax.scatter(iters, costs)
	ax.set_xlabel('iterations')
	ax.set_ylabel('cost')	
	plt.show()