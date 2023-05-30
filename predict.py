import argparse

def parse_arguments():

	parser = argparse.ArgumentParser(prog="predict", usage="predict -t0 [theta0] -t1 [theta1]")
	parser.add_argument(
		"-t0", "--theta0", type=float,
		action='store', required=True,
		help="theta0 coefficient of the linear function : price = t0 + (t1 x mileage)"
		)
	parser.add_argument("-t1", "--theta1", type=float,
						action='store', required=True,
						help="theta1 coefficient of the linear function : price = t0 + (t1 * mileage)"
						)
	return parser.parse_args()


if __name__ == '__main__':

	args = parse_arguments()

	theta0 = args.theta0
	theta1 = args.theta1

	prompt = "Enter mileage in kilometers : "
	while True:
		mileage = input(prompt)
		if mileage.lower() == "exit":
			exit(0)
		try:
			mileage = float(mileage)
			print(f"calculated price is : {theta0 + (mileage * theta1)}")
		except ValueError:
			print("Please input a valid mileage or 'exit' to leave the program")
