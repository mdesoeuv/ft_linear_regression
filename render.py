import sys
import csv
import matplotlib.pyplot as plt
from learn import read_data, estimate_price

def normalize(x: float, x_min: float, x_max: float, x_avg):

	return (x ) / (x_max )


if __name__ == "__main__":
    
	if len(sys.argv) < 2:
		print("usage: learn [csvfile]")
		exit(1)
	
	dataset = read_data(sys.argv[1])
	x = [int(entry['km']) for entry in dataset]
	y = [int(entry['price']) for entry in dataset]

	fig, ax = plt.subplots()
	fig.suptitle("Linear regression representation")
	ax.scatter(x, y)
	ax.set_xlabel('kilometers')
	ax.set_ylabel('price')

	theta0 = 8499.599649933132
	theta1 = -5147.751262008374
	xmin = 22899.0
	xmax = 240000.0
	xavg = 101066.25

	x1 = 200000
	x1_norm = normalize(x1, xmin, xmax, xavg)
	y1 = estimate_price(x1_norm, theta0, theta1)
	x2 = 50000
	x2_norm = normalize(x2, xmin, xmax, xavg)
	y2 = estimate_price(x2_norm, theta0, theta1)
	print([x1, y1], [x2, y2])
	plt.plot([x1, x2], [y1, y2], label='linear_regression')

	plt.show()
