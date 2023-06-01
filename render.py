import sys
import matplotlib.pyplot as plt
from learn import read_csv, estimate_price, normalize, normalize_dataset ,r_square

#TODO: Analytics Object

def plot_iterations_costs(data: str):
	
	try:
		iters = [float(entry['iterations']) for entry in data]
		costs = [float(entry['costs']) for entry in data]
		rmse = [float(entry['rmse']) for entry in data]
	except KeyError:
		print("Invalid dataset.")
		exit(1)

	fig, ax = plt.subplots()
	fig.suptitle("Cost function evolution with iterations")
	ax.scatter(iters, costs)
	ax.set_xlabel('iterations')
	ax.set_ylabel('cost')
	try:
		fig.savefig("iterations_costs.png")	
	except Exception:
		print("Png export failed.")
	print(f"RMSE = {rmse[-1]}")


def plot_linear_regression(dataset: str, results: str):

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
	print("Regression Line drawn with points :")
	print([x1, y1], [x2, y2], end="\n\n")
	plt.plot([x1, x2], [y1, y2], label='linear_regression', color='red')
	try:
		fig.savefig("linear_regression.png")	
	except Exception:
		print("Png export failed.")



if __name__ == "__main__":
    
	if len(sys.argv) < 2:
		print("usage: render [csvfile]")
		exit(1)
	
	dataset = read_csv(sys.argv[1])
	iters_costs_data = read_csv("costs_iterations.csv")
	results_data = read_csv("regression_results.csv")[-1]
	print(results_data)
	#TODO: object RegressionParameters
	theta0 = float(results_data["theta0"])
	theta1 = float(results_data["theta1"])
	x_min = float(results_data["x_min"])
	x_max = float(results_data["x_max"])
		

	norm_dataset = normalize_dataset(dataset, x_min, x_max)


	print("Analytics :")
	rsquare = r_square(norm_dataset, theta0, theta1)
	print(f"R2 = {rsquare}")

	plot_iterations_costs(iters_costs_data)
	plot_linear_regression(dataset, results_data)

	plt.show()
