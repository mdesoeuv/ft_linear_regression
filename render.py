import sys
import csv
import matplotlib.pyplot as plt
from learn import read_data, estimate_price, normalize


def plot_iterations_costs(filepath: str):

	data = read_data(filepath)
	iters = [float(entry['iterations']) for entry in data]
	costs = [float(entry['costs']) for entry in data]
	fig, ax = plt.subplots()
	fig.suptitle("Cost function evolution with iterations")
	ax.scatter(iters, costs)
	ax.set_xlabel('iterations')
	ax.set_ylabel('cost')	


def plot_linear_regression(dataset_path: str, results_path: str):
	dataset = read_data(dataset_path)
	results = read_data(results_path)[-1]
	print(results)
	theta0 = float(results["theta0"])
	theta1 = float(results["theta1"])
	x_min = float(results["x_min"])
	x_max = float(results["x_max"])
	
	x = [int(entry['km']) for entry in dataset]
	y = [int(entry['price']) for entry in dataset]

	fig, ax = plt.subplots()
	fig.suptitle("Linear regression representation")
	ax.scatter(x, y)
	ax.set_xlabel('kilometers')
	ax.set_ylabel('price')

	x1 = 200000
	x1_norm = normalize(x1, x_min, x_max)
	y1 = estimate_price(x1_norm, theta0, theta1)
	x2 = 50000
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
	# theta0 = 8008.411898164373
	# theta1 = -4656.517711291304
	# xmin = 22899.0
	# xmax = 240000.0

