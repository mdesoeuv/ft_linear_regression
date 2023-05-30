from pathlib import Path
import csv
import sys

LEARNING_RATE = 0.01


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


def estimate_price(mileage: float, theta0: float, theta1: float):
	return theta0 + (mileage * theta1)


def derivate_sum(dataset: list, theta0: float, theta1: float):
	sum_theta0 = 0
	sum_theta1 = 0
	for entry in dataset:
		mileage = float(entry['km'])
		price = float(entry['price'])
		# print(mileage, price)
		sum_theta0 += estimate_price(mileage, theta0, theta1) - price
		sum_theta1 += mileage * estimate_price(mileage, theta0, theta1) - price

	print(sum_theta0, sum_theta1)
	sum_theta0 = sum_theta0 / len(dataset)
	sum_theta1 = sum_theta1 / len(dataset)
	return (sum_theta0, sum_theta1)


if __name__ == "__main__":
	
	if len(sys.argv) < 2:
		print("Please input the csv filepath")
		exit(1)
	
	dataset = read_data(sys.argv[1])
	m = len(dataset)
	
	theta0 = 0
	theta1 = 0

	min_reached = False
	for _ in range(100):
		derivate_theta0, derivate_theta1 = derivate_sum(dataset, theta0, theta1)
		# print(derivate_theta0, derivate_theta1)
		theta0 = theta0 - (LEARNING_RATE * derivate_theta0)
		theta1 = theta1 - (LEARNING_RATE * derivate_theta1)
		print(theta0, theta1)
		if 0 <= derivate_theta0 < 1 and 0 <= derivate_theta1 < 1:
			print(f"Minimum reached : theta0={theta0}, theta1={theta1}")
			min_reached = True

		
			