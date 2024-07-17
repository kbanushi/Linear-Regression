import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def addPolynomialFeatures(X, degree):
    X_poly = np.ones((X.shape[0], 1))
    for i in range(1, degree + 1):
        X_poly = np.concatenate((X_poly, X ** i), axis=1)
    return X_poly


def learning(x, y):
    xTranspose = x.T
    parameters = la.inv(xTranspose.dot(x)).dot(xTranspose).dot(y)

    return parameters


def polynomial_function(parameters, x):
    y = parameters[0]
    for i in range(1, len(parameters)):
        y += parameters[i] * (x ** i)
    return y


def meanSquaredError(true, prediction):
    return np.sqrt(np.mean((true - prediction) ** 2))


def partOne():
    x = np.random.uniform(0, 1, 20)
    n = np.random.normal(0, 0.3, 20)
    y = np.sin(2 * np.pi * x) + n

    x = x.reshape(-1, 1)
    xPoly = addPolynomialFeatures(x, degree=3)

    parameters = learning(xPoly, y)
    print("Parameters:", parameters)

    xTest = np.array([[0], [0.25], [0.5], [0.75], [1]])

    xTestPoly = addPolynomialFeatures(xTest, degree=3)

    prediction = xTestPoly.dot(parameters)

    print("Predictions:", prediction)

    time = np.arange(0, 1, 0.01)
    function = polynomial_function(parameters, time)

    plt.plot(time, np.sin(2 * np.pi * time), 'ro', label='Sin Graph')
    plt.plot(x, y, 'bo', label='Training Data')
    plt.plot(xTest, prediction, 'yo', label='Test Prediction')
    for i in range(len(xTest)):
        plt.annotate(str(prediction[i]), xy=(xTest[i], prediction[i]))
    plt.plot(time, function, 'go', label='Polynomial After Training')

    plt.legend()

    plt.show()


def partTwo():
    x = np.random.uniform(0, 1, 100)
    n = np.random.normal(0, 0.3, 100)
    y = np.sin(2 * np.pi * x) + n

    x = x.reshape(-1, 1)
    polynomial = addPolynomialFeatures(x, degree=3)

    prediction_errors = []
    numFolds = 5
    foldSize = len(polynomial) // numFolds

    for foldStart in range(0, len(polynomial), foldSize):
        foldEnd = foldStart + foldSize

        fold_index, fold_value = x[foldStart:foldEnd], y[foldStart:foldEnd]
        other_index, other_value = np.concatenate((x[:foldStart], x[foldEnd:])), np.concatenate((y[:foldStart], y[foldEnd:]))

        fold_index = fold_index.reshape(-1, 1)
        fold_index = addPolynomialFeatures(fold_index, degree=3)

        other_index = other_index.reshape(-1, 1)
        other_index = addPolynomialFeatures(other_index, degree=3)

        fold_parameters = learning(fold_index, fold_value)

        prediction = other_index.dot(fold_parameters)

        prediction_error = meanSquaredError(other_value, prediction)
        prediction_errors.append(prediction_error)
        print("Learned Parameters:", fold_parameters)
        print("Fold Prediction Error:", prediction_error, end="\n\n")

    plt.show()

    print("Average Prediction Error:", np.mean(prediction_errors))



def partThree():
    x = np.random.uniform(0, 1, 100)
    n = np.random.normal(0, 0.3, 100)
    y = np.sin(2 * np.pi * x) + n

    figure, axis = plt.subplots(3, 2)

    x = x.reshape(-1, 1)
    predictionAverages = np.array([])
    time = np.arange(0, 1, 0.01)

    axis[0, 0].plot(x, y, 'bo')
    axis[0, 0].set_title("Training Data Plot")

    polynomials = []

    for i in range(1, 11, 2):
        predictionErrors = []
        first = True

        # Manually split data into training and validation sets
        numSamples = x.shape[0]
        numFolds = 5
        foldSize = numSamples // numFolds

        for foldStart in range(0, numSamples, foldSize):
            foldEnd = foldStart + foldSize

            foldIndex, foldValue = x[foldStart:foldEnd], y[foldStart:foldEnd]
            otherIndex, otherValue = np.concatenate((x[:foldStart], x[foldEnd:])), np.concatenate((y[:foldStart], y[foldEnd:]))

            foldIndex = addPolynomialFeatures(foldIndex, degree=i)
            otherIndex = addPolynomialFeatures(otherIndex, degree=i)

            foldParameters = learning(foldIndex, foldValue)

            prediction = otherIndex.dot(foldParameters)

            predictionError = meanSquaredError(otherValue, prediction)
            predictionErrors.append(predictionError)

            if first:
                foldPolynomialFunction = polynomial_function(foldParameters, time)
                polynomials.append(foldPolynomialFunction)
                first = False

        mean = np.mean(predictionErrors)
        predictionAverages = np.append(predictionAverages, mean)
        print("Average Prediction Error For Degree:", i, mean, end="\n\n")

    min_degree_index = np.argmin(predictionAverages)
    min_degree = 2 * min_degree_index + 1  # Convert index to actual degree
    print("Degree with the lowest average prediction error:", min_degree)

    axis[0,1].plot(polynomials[0])
    axis[0,1].set_title("Order 1")

    axis[1, 0].plot(polynomials[1])
    axis[1, 0].set_title("Order 3")

    axis[1, 1].plot(polynomials[2])
    axis[1, 1].set_title("Order 5")

    axis[2, 0].plot(polynomials[3])
    axis[2, 0].set_title("Order 7")

    axis[2, 1].plot(polynomials[4])
    axis[2, 1].set_title("Order 9")

    plt.show()


def main():
    print("Part One Results:")
    partOne()

    print("\nPart Two Results:")
    partTwo()

    print("\nPart Three Results:")
    partThree()


main()
