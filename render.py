import sys
import matplotlib.pyplot as plt
from learn import read_csv, predict, UniVariableLinearRegression, RegressionParameters, RegressionError


def plot_iterations_costs(datas: list):
	
	fig, ax = plt.subplots()
	fig.suptitle("Cost function evolution with iterations")
	ax.set_xlabel('iterations')
	ax.set_ylabel('cost')
	
	all_costs = []
	all_iters = []
	for data in datas:
		try:
			iters = [float(entry['iterations']) for entry in data]
			costs = [float(entry['costs']) for entry in data]
			all_iters.append(iters)
			all_costs.append(costs)
			# rmse = [float(entry['rmse']) for entry in data]
		except KeyError:
			print("Invalid dataset.")
			exit(1)
	size = len(all_iters[0])
	ax.scatter(all_iters[0], all_costs[0][:size], s=10, c="blue", label="Learning Rate : 0.5")
	ax.scatter(all_iters[0], all_costs[1][:size], s=10, c="red", label="Learning Rate : 0.1")
	ax.scatter(all_iters[0], all_costs[2][:size], s=10, c="green", label="Learning Rate : 0.01")
	ax.legend()
	try:
		fig.savefig("iterations_costs.png")	
	except Exception:
		print("Png export failed.")
	

def plot_linear_regression(data: UniVariableLinearRegression):
	fig, ax = plt.subplots()
	fig.suptitle("Linear regression representation")
	ax.scatter(data.x, data.y)
	ax.set_xlabel(data.feature)
	ax.set_ylabel(data.target)

	# Calculating two points to draw the regression line
	x1 = 200000
	x1_norm = UniVariableLinearRegression.normalize(x1, data.feature_min, data.feature_max)
	y1 = predict(x1_norm, data.theta0, data.theta1)
	x2 = 42000
	x2_norm = UniVariableLinearRegression.normalize(x2, data.feature_min, data.feature_max)
	y2 = predict(x2_norm, data.theta0, data.theta1)
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
	data = UniVariableLinearRegression(
		dataset=dataset,
		feature="km",
		target="price"
	)
	iters_costs_data1 = read_csv("costs_iterations1.csv")
	iters_costs_data2 = read_csv("costs_iterations2.csv")
	iters_costs_data3 = read_csv("costs_iterations3.csv")
	results_data = read_csv("regression_parameters.csv")[-1]

	try:
		data.feature_min = float(results_data["x_min"])
		data.feature_max = float(results_data["x_max"])
		data.theta0 = float(results_data["theta0"])
		data.theta1 = float(results_data["theta1"])
	except KeyError:
		print("Invalid results csv file.")
		exit(1)
	
	print("Analytics :")
	print(f"R2 = {data.r_square()}")
	print(f"RMSE = {data.calculate_rmse()}")

	plot_iterations_costs([iters_costs_data1, iters_costs_data2, iters_costs_data3])
	plot_linear_regression(data)

	plt.show()
