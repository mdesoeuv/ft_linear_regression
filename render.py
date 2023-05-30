import sys
import csv
import matplotlib.pyplot as plt
from learn import read_data, estimate_price

if __name__ == "__main__":
    
	if len(sys.argv) < 2:
		print("usage: learn [csvfile]")
		exit(1)
	
	dataset = read_data(sys.argv[1])
	x = [int(entry['km']) for entry in dataset]
	y = [int(entry['price']) for entry in dataset]

	fig, ax = plt.subplots()
	ax.scatter(x, y)
	ax.set_xlabel('kilometers')
	ax.set_ylabel('price')

	theta0 = 0.04344178708402814
	theta1 = -0.042637780971932715
	x1 = 200000
	y1 = estimate_price(x1, theta0, theta1)
	x2 = 50000
	y2 = estimate_price(x2, theta0, theta1)
	print([x1, y1], [x2, y2])
	plt.plot([x1, y1], [x2, y2], label='linear_regression')

	plt.show()
