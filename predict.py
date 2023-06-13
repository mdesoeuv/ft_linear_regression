import argparse
from learn import read_csv, UniVariableLinearRegression


def parse_arguments():

    parser = argparse.ArgumentParser(
        prog="predict", usage="predict -f [training csv]"
        )
    parser.add_argument(
        "-f", "--file", type=str,
        action='store', required=False,
        help="path of the csv file containing the training results"
        )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    filepath = args.file

    theta0 = 0
    theta1 = 0
    x_min = 0
    x_max = 1

    if filepath:
        results = read_csv(filepath)[-1]
        try:
            theta0 = float(results["theta0"])
            theta1 = float(results["theta1"])
            x_min = float(results["x_min"])
            x_max = float(results["x_max"])
        except KeyError:
            print("Invalid results csv file.")
            exit(1)

    prompt = "Enter mileage in kilometers : "
    while True:
        mileage = input(prompt)
        if mileage.lower() == "exit":
            exit(0)
        try:
            mileage = UniVariableLinearRegression.normalize(
                float(mileage), x_min, x_max
                )
            print(f"calculated price is : {theta0 + (mileage * theta1)}")
        except ValueError:
            print(
                "Please input a valid mileage or 'exit' to leave the program"
                )
