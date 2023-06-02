import sys, signal
import matplotlib.pyplot as plt
from learn import read_csv, predict, UniVariableLinearRegression, RegressionError


def plot_iterations_costs(datas: list, learning_rates: list):
	
	fig, ax = plt.subplots()
	fig.suptitle("Cost function evolution with iterations")
	ax.set_xlabel('iterations')
	ax.set_ylabel('cost')
	
	for data, learning_rate in zip(datas, learning_rates):
		try:
			iters = [float(entry['iterations']) for entry in data]
			costs = [float(entry['costs']) for entry in data]
			ax.scatter(iters, costs, s=5, label=f"Learning Rate : {learning_rate}")
		except KeyError:
			print("Invalid dataset.")
			exit(1)
	ax.legend()
	ax.set(xlim=(0,400))
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
	
	signal.signal(signal.SIGINT, signal.SIG_DFL)	

	dataset = read_csv(sys.argv[1])

	# Plotting original regression
	regressions = []
	try:
		data = UniVariableLinearRegression(
			dataset=dataset,
			feature="km",
			target="price"
		)
	except RegressionError:
		print("Invalid data.")
		exit(1)

	results_data = read_csv("regression_parameters.csv")[-1]

	try:
		data.feature_min = float(results_data["x_min"])
		data.feature_max = float(results_data["x_max"])
		data.theta0 = float(results_data["theta0"])
		data.theta1 = float(results_data["theta1"])
		data.learning_rate = float(results_data["learning_rate"])
	except KeyError:
		print("Invalid results csv file.")
		exit(1)
	data.iters_costs = read_csv("costs_iterations.csv")
	regressions.append(data)
	plot_linear_regression(data)
	
	# Plotting more regressions to emphasis the learning rate
	for learning_rate in [0.55, 0.7, 1, 0.05]:
		try:
			regression = UniVariableLinearRegression(
				dataset=dataset,
				feature="km",
				target="price",
				learning_rate=learning_rate
			)
		except RegressionError:
			print("Invalid data.")
			exit(1)

		regression.gradient_descent()
		regression.print_analytics()
		regressions.append(regression)

	plot_iterations_costs(
		[regression.iters_costs for regression in regressions],
		[regression.learning_rate for regression in regressions]
		)
	
	plt.show()