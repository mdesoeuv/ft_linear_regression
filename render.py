import sys
import csv
import matplotlib.pyplot as plt
from learn import read_csv, estimate_price, normalize


def plot_iterations_costs(filepath: str):

	data = read_csv(filepath)
	
	try:
		iters = [float(entry['iterations']) for entry in data]
		costs = [float(entry['costs']) for entry in data]
	except KeyError:
		print("Invalid dataset.")
		exit(1)

	fig, ax = plt.subplots()
	fig.suptitle("Cost function evolution with iterations")
	ax.scatter(iters, costs)
	ax.set_xlabel('iterations')
	ax.set_ylabel('cost')	


def plot_linear_regression(dataset_path: str, results_path: str):
	dataset = read_csv(dataset_path)
	results = read_csv(results_path)[-1]
	print(results)

	try:
		theta0 = float(results["theta0"])
		theta1 = float(results["theta1"])
		x_min = float(results["x_min"])
		x_max = float(results["x_max"])
		
		x = [int(entry['km']) for entry in dataset]
		y = [int(entry['price']) for entry in dataset]
	except KeyError:
		print("Invalid dataset.")
		exit(1)

	fig, ax = plt.subplots()
	fig.suptitle("Linear regression representation")
	ax.scatter(x, y)
	ax.set_xlabel('kilometers')
	ax.set_ylabel('price')

	# Calculating two points to draw the regression line
	x1 = 200000
	x1_norm = normalize(x1, x_min, x_max)
	y1 = estimate_price(x1_norm, theta0, theta1)
	x2 = 42000
	x2_norm = normalize(x2, x_min, x_max)
	y2 = estimate_price(x2_norm, theta0, theta1)
	print([x1, y1], [x2, y2])
	plt.plot([x1, x2], [y1, y2], label='linear_regression', color='red')



if __name__ == "__main__":
    
	if len(sys.argv) < 2:
		print("usage: learn [csvfile]")
		exit(1)
	
	plot_iterations_costs("costs_iterations.csv")
	plot_linear_regression(sys.argv[1], "regression_results.csv")

	plt.show()
