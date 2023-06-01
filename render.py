import sys
import matplotlib.pyplot as plt
from learn import read_csv, predict,r_square, UniVariableLinearRegression, RegressionParameters, RegressionError
from dataclasses import dataclass

#TODO: Analytics Object



@dataclass
class Analytics:
	rsquare: float
	rmse: float
	
	def __repr__(self):
		return f"Analytics :\nR2 = {self.rsquare}\nRMSE = {self.rmse}\n\n"


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


def plot_linear_regression(dataset: str, params: RegressionParameters):

	try:
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
	x1_norm = UniVariableLinearRegression.normalize(x1, params.x_min, params.x_max)
	y1 = predict(x1_norm, params.theta0, params.theta1)
	x2 = 42000
	x2_norm = UniVariableLinearRegression.normalize(x2, params.x_min, params.x_max)
	y2 = predict(x2_norm, params.theta0, params.theta1)
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
	params = RegressionParameters(
		theta0=float(results_data.get("theta0", 0)),
		theta1=float(results_data.get("theta1", 0)),
		x_min=float(results_data.get("x_min", 0)),
		x_max=float(results_data.get("x_max", 1))
	)

	norm_dataset = UniVariableLinearRegression.normalize_dataset(dataset, "km", params.x_min, params.x_max)

	print("Analytics :")
	rsquare = r_square(norm_dataset, params.theta0, params.theta1)
	print(f"R2 = {rsquare}")

	plot_iterations_costs(iters_costs_data)
	plot_linear_regression(dataset, params)

	plt.show()
