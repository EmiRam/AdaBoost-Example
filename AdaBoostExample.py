# AdaBoost Example 

# This is an implementation of the AdaBoost alogorithm with 
# console printed messages to walk through the process.
# The first part of this file defines a class for a "weak learner" a.k.a. a 
# "decision stump". Then some functions are defined.
# The code that is executed when this file is run follows.
# It generates some random data and then trains the AdaBoost model and tests it.

import numpy as np

# CLASS DEFINITION

# class to represent each weak learner/decision stump
class WeakLearner:
    def __init__(self):
        self.orientation = '>'
        self.attributeIndex = None
        self.threshold = None
        self.alpha = None

    # Returns classifications/predictions for a matrix of data
    def classify(self, data):
        numberOfSamples = data.shape[0]
        attributeValues = data[:, self.attributeIndex]
        classifications = np.ones(numberOfSamples)
        if self.orientation == '>':
            classifications[attributeValues < self.threshold] = -1
        else:
            classifications[attributeValues > self.threshold] = -1

        return classifications

    # Returns a classification/prediction for a single data point
    def classifySingleDataPoint(self, dataPoint):
        attributeValue = dataPoint[self.attributeIndex]
        classification = 1

        if self.orientation == '>' and attributeValue < self.threshold :
            classification = -1
        elif self.orientation == '<' and attributeValue > self.threshold:
            classification = -1

        return classification

# FUNCTION DEFINITIONS

# Performs the classification/prediction of a data point using the final model from the weak learners
def classifyDataPointAdaboost(weakLearners, dataPoint):
    sum = 0
    print()
    for weakLearner in weakLearners:
        print(str(round(weakLearner.alpha, 4)) + " * " + str(weakLearner.classifySingleDataPoint(dataPoint)))
        sum += weakLearner.alpha * weakLearner.classifySingleDataPoint(dataPoint)

    print("Sum = " + str(round(sum, 4)))
    return np.sign(sum)

def trainAdaboost(data, labels, numberOfWeakLearners, thresholdPlusOrMinus):
    weakLearners = []

    numberOfSamples = data.shape[0]

    # weights start off equaling 1/number of data points
    weights = np.full(numberOfSamples, (1 / numberOfSamples))

    

    for i in range(numberOfWeakLearners):
            print("Iteration #:" + str(i+1))
            weakLearner = WeakLearner()
            minimizedError = 1000000

            print("Weights for this iteration:")
            print(np.around(weights, 4))

            for v in range(data.shape[1]):
                print("Attribute #:" + str(v+1))
                attributeValues = data[:, v]
                # set thresholds equal to all unique values in the training set
                thresholds = np.unique(attributeValues)
                # shift the thresholds slightly so that they don't lie directly on the data points
                thresholds = thresholds + thresholdPlusOrMinus
                # add another threshold at the very far "left" of the data
                thresholds = np.insert(thresholds, 0, (thresholds[0] - (2 * thresholdPlusOrMinus)), axis=0)

                for threshold in thresholds:
                    # predict for value > threshold = 1
                    orientation = '>'
                    predictions = np.ones(numberOfSamples)
                    predictions[attributeValues < threshold] = -1

                    # Error = sum of weights of data points wrongly predicted
                    errors = weights[labels != predictions]
                    error = sum(errors)

                    print("Error for attribute #" + str(v+1) + " with a threshold of value > " + str(round(threshold, 4)) + " is " + str(round(error, 4)))

                    if error < minimizedError:
                        weakLearner.orientation = orientation
                        weakLearner.threshold = threshold
                        weakLearner.attributeIndex = v
                        minimizedError = error

                    # reverse the orientation and do it again
                    # predict for value < threshold = 1
                    orientation = '<'
                    predictions = np.ones(numberOfSamples)
                    predictions[attributeValues > threshold] = -1

                    # Error = sum of weights of data points wrongly predicted
                    errors = weights[labels != predictions]
                    error = sum(errors)

                    print("Error for attribute #" + str(v+1) + " with a threshold of value < " + str(round(threshold, 4)) + " is " + str(round(error, 4)))

                    if error < minimizedError:
                        weakLearner.orientation = orientation
                        weakLearner.threshold = threshold
                        weakLearner.attributeIndex = v
                        minimizedError = error

            print("Therefore, the lowest error for this iteration = " + str(round(minimizedError, 4)))
            print("This error was obtained from the indicator function that uses attribute #" + str(weakLearner.attributeIndex + 1) + " where the value is " + weakLearner.orientation + " " + str(round(weakLearner.threshold, 4)) )
            
            # add this tiny number in the event that we are dividing by zero
            avoidDivideByZero = 0.0000000001

            sumOfWeights = np.sum(weights)

            epsilon = minimizedError / sumOfWeights
            print("Epsilon = " + str(round(epsilon, 4)))
            weakLearner.alpha = 0.5 * np.log((1.0 - epsilon + avoidDivideByZero) / (epsilon + avoidDivideByZero))
            print("Alpha = " + str(round(weakLearner.alpha, 4)))
            # calculate predictions and update weights
            predictions = weakLearner.classify(data)
            print("Data Point Predictions for this iteration:")
            print(predictions)

            print("Correct weight update factor = " + str(round(np.exp(-weakLearner.alpha) / sumOfWeights, 4 )))
            print("Incorrect weight update factor = " + str(round(np.exp(weakLearner.alpha) / sumOfWeights, 4 )))

            # Multiply the predictions and the label values to decide whether or not the prediction was correct
            # this decides if the weight should be increased or decreased
            weights *= np.exp(-weakLearner.alpha * labels * predictions)
            
            weights /= sumOfWeights

            weakLearners.append(weakLearner)


    return weakLearners

# START OF EXECUTION

# DATASET GENERATION
sizeOfClass = 10

class1Means = [1,3,5]
class1Cov = [[15,0,0],[0,15,0],[0,0,10]]

class2Means = [-3,0,-2]
class2Cov = [[15,0,0],[0,15,0],[0,0,10]]

class1NumericalData = np.random.multivariate_normal(class1Means, class1Cov, sizeOfClass)
class2NumericalData = np.random.multivariate_normal(class2Means, class2Cov, sizeOfClass)
trainingDataSet = np.concatenate((class1NumericalData[0:5], class2NumericalData[0:5]), axis=0)
trainingDataSet = np.around(trainingDataSet, 2)

testingDataSet = np.concatenate((class1NumericalData[5:7], class2NumericalData[5:7]), axis=0)
testingDataSet = np.around(testingDataSet, 2)

print("The training dataset, first 5 rows are from class 1, last 5 are from class 2:")
print(trainingDataSet)

print("The testing dataset, first 2 rows are from class 1, last 2 are from class 2:")
print(testingDataSet)

# Set up labels, 1 is for class 1, -1 is for class 2
trainingLabels = np.array([1,1,1,1,1,-1,-1,-1,-1,-1])
testingLabels = np.array([1,1,-1,-1])


# run AdaBoost with 3 weak learners, 
# with a threshold plus or minus of 0.005 because the data has been rounded to 2 decimal places,
# so this means the thresholds tested will be guaranteed to be between data points
weakLearners = trainAdaboost(trainingDataSet, trainingLabels, 3, 0.005)

print("\nTest classification results:")
for i in range(len(testingDataSet)):
    classification = classifyDataPointAdaboost(weakLearners, testingDataSet[i])

    if classification == 1:
        print("Test Example #" + str(i+1) + " is predicted to belong to Class 1.")
        if testingLabels[i] == 1:
            print("The classification was correct.")
        else:
            print("The classification was incorrect.")
    else:
        print("Test Example #" + str(i+1) + " is predicted to belong to Class 2.")
        if testingLabels[i] == -1:
            print("The classification was correct.")
        else:
            print("The classification was incorrect.")

print("done")

# SAMPLE RUN OUTPUT
#The training dataset, first 5 rows are from class 1, last 5 are from class 2:
#[[ 1.01  5.84 -1.54]
# [ 2.37 -3.45  3.73]
# [ 0.17  3.46  1.91]
# [ 5.02 -6.48  5.55]
# [ 8.52  3.24 -1.25]
# [ 0.67  1.79 -2.7 ]
# [-0.28 -5.82 -2.11]
# [-1.3  -2.43  2.22]
# [ 1.53  1.28 -2.4 ]
# [-2.   -1.17 -4.49]]
#The testing dataset, first 2 rows are from class 1, last 2 are from class 2:
#[[ 2.67  0.3   2.2 ]
# [ 2.81  6.42  7.99]
# [-2.26  1.95  0.64]
# [-7.61 -1.91 -0.74]]
#Iteration #:1
#Weights for this iteration:
#[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
#Attribute #:1
#Error for attribute #1 with a threshold of value > -2.005 is 0.5
#Error for attribute #1 with a threshold of value < -2.005 is 0.5
#Error for attribute #1 with a threshold of value > -1.995 is 0.4
#Error for attribute #1 with a threshold of value < -1.995 is 0.6
#Error for attribute #1 with a threshold of value > -1.295 is 0.3
#Error for attribute #1 with a threshold of value < -1.295 is 0.7
#Error for attribute #1 with a threshold of value > -0.275 is 0.2
#Error for attribute #1 with a threshold of value < -0.275 is 0.8
#Error for attribute #1 with a threshold of value > 0.175 is 0.3
#Error for attribute #1 with a threshold of value < 0.175 is 0.7
#Error for attribute #1 with a threshold of value > 0.675 is 0.2
#Error for attribute #1 with a threshold of value < 0.675 is 0.8
#Error for attribute #1 with a threshold of value > 1.015 is 0.3
#Error for attribute #1 with a threshold of value < 1.015 is 0.7
#Error for attribute #1 with a threshold of value > 1.535 is 0.2
#Error for attribute #1 with a threshold of value < 1.535 is 0.8
#Error for attribute #1 with a threshold of value > 2.375 is 0.3
#Error for attribute #1 with a threshold of value < 2.375 is 0.7
#Error for attribute #1 with a threshold of value > 5.025 is 0.4
#Error for attribute #1 with a threshold of value < 5.025 is 0.6
#Error for attribute #1 with a threshold of value > 8.525 is 0.5
#Error for attribute #1 with a threshold of value < 8.525 is 0.5
#Attribute #:2
#Error for attribute #2 with a threshold of value > -6.485 is 0.5
#Error for attribute #2 with a threshold of value < -6.485 is 0.5
#Error for attribute #2 with a threshold of value > -6.475 is 0.6
#Error for attribute #2 with a threshold of value < -6.475 is 0.4
#Error for attribute #2 with a threshold of value > -5.815 is 0.5
#Error for attribute #2 with a threshold of value < -5.815 is 0.5
#Error for attribute #2 with a threshold of value > -3.445 is 0.6
#Error for attribute #2 with a threshold of value < -3.445 is 0.4
#Error for attribute #2 with a threshold of value > -2.425 is 0.5
#Error for attribute #2 with a threshold of value < -2.425 is 0.5
#Error for attribute #2 with a threshold of value > -1.165 is 0.4
#Error for attribute #2 with a threshold of value < -1.165 is 0.6
#Error for attribute #2 with a threshold of value > 1.285 is 0.3
#Error for attribute #2 with a threshold of value < 1.285 is 0.7
#Error for attribute #2 with a threshold of value > 1.795 is 0.2
#Error for attribute #2 with a threshold of value < 1.795 is 0.8
#Error for attribute #2 with a threshold of value > 3.245 is 0.3
#Error for attribute #2 with a threshold of value < 3.245 is 0.7
#Error for attribute #2 with a threshold of value > 3.465 is 0.4
#Error for attribute #2 with a threshold of value < 3.465 is 0.6
#Error for attribute #2 with a threshold of value > 5.845 is 0.5
#Error for attribute #2 with a threshold of value < 5.845 is 0.5
#Attribute #:3
#Error for attribute #3 with a threshold of value > -4.495 is 0.5
#Error for attribute #3 with a threshold of value < -4.495 is 0.5
#Error for attribute #3 with a threshold of value > -4.485 is 0.4
#Error for attribute #3 with a threshold of value < -4.485 is 0.6
#Error for attribute #3 with a threshold of value > -2.695 is 0.3
#Error for attribute #3 with a threshold of value < -2.695 is 0.7
#Error for attribute #3 with a threshold of value > -2.395 is 0.2
#Error for attribute #3 with a threshold of value < -2.395 is 0.8
#Error for attribute #3 with a threshold of value > -2.105 is 0.1
#Error for attribute #3 with a threshold of value < -2.105 is 0.9
#Error for attribute #3 with a threshold of value > -1.535 is 0.2
#Error for attribute #3 with a threshold of value < -1.535 is 0.8
#Error for attribute #3 with a threshold of value > -1.245 is 0.3
#Error for attribute #3 with a threshold of value < -1.245 is 0.7
#Error for attribute #3 with a threshold of value > 1.915 is 0.4
#Error for attribute #3 with a threshold of value < 1.915 is 0.6
#Error for attribute #3 with a threshold of value > 2.225 is 0.3
#Error for attribute #3 with a threshold of value < 2.225 is 0.7
#Error for attribute #3 with a threshold of value > 3.735 is 0.4
#Error for attribute #3 with a threshold of value < 3.735 is 0.6
#Error for attribute #3 with a threshold of value > 5.555 is 0.5
#Error for attribute #3 with a threshold of value < 5.555 is 0.5
#Therefore, the lowest error for this iteration = 0.1
#This error was obtained from the indicator function that uses attribute #3 where the value is > -2.105
#Epsilon = 0.1
#Alpha = 1.0986
#Data Point Predictions for this iteration:
#[ 1.  1.  1.  1.  1. -1. -1.  1. -1. -1.]
#Correct weight update factor = 0.3333
#Incorrect weight update factor = 3.0
#Iteration #:2
#Weights for this iteration:
#[0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.3    0.0333 0.0333]
#Attribute #:1
#Error for attribute #1 with a threshold of value > -2.005 is 0.4333
#Error for attribute #1 with a threshold of value < -2.005 is 0.1667
#Error for attribute #1 with a threshold of value > -1.995 is 0.4
#Error for attribute #1 with a threshold of value < -1.995 is 0.2
#Error for attribute #1 with a threshold of value > -1.295 is 0.1
#Error for attribute #1 with a threshold of value < -1.295 is 0.5
#Error for attribute #1 with a threshold of value > -0.275 is 0.0667
#Error for attribute #1 with a threshold of value < -0.275 is 0.5333
#Error for attribute #1 with a threshold of value > 0.175 is 0.1
#Error for attribute #1 with a threshold of value < 0.175 is 0.5
#Error for attribute #1 with a threshold of value > 0.675 is 0.0667
#Error for attribute #1 with a threshold of value < 0.675 is 0.5333
#Error for attribute #1 with a threshold of value > 1.015 is 0.1
#Error for attribute #1 with a threshold of value < 1.015 is 0.5
#Error for attribute #1 with a threshold of value > 1.535 is 0.0667
#Error for attribute #1 with a threshold of value < 1.535 is 0.5333
#Error for attribute #1 with a threshold of value > 2.375 is 0.1
#Error for attribute #1 with a threshold of value < 2.375 is 0.5
#Error for attribute #1 with a threshold of value > 5.025 is 0.1333
#Error for attribute #1 with a threshold of value < 5.025 is 0.4667
#Error for attribute #1 with a threshold of value > 8.525 is 0.1667
#Error for attribute #1 with a threshold of value < 8.525 is 0.4333
#Attribute #:2
#Error for attribute #2 with a threshold of value > -6.485 is 0.4333
#Error for attribute #2 with a threshold of value < -6.485 is 0.1667
#Error for attribute #2 with a threshold of value > -6.475 is 0.4667
#Error for attribute #2 with a threshold of value < -6.475 is 0.1333
#Error for attribute #2 with a threshold of value > -5.815 is 0.4333
#Error for attribute #2 with a threshold of value < -5.815 is 0.1667
#Error for attribute #2 with a threshold of value > -3.445 is 0.4667
#Error for attribute #2 with a threshold of value < -3.445 is 0.1333
#Error for attribute #2 with a threshold of value > -2.425 is 0.1667
#Error for attribute #2 with a threshold of value < -2.425 is 0.4333
#Error for attribute #2 with a threshold of value > -1.165 is 0.1333
#Error for attribute #2 with a threshold of value < -1.165 is 0.4667
#Error for attribute #2 with a threshold of value > 1.285 is 0.1
#Error for attribute #2 with a threshold of value < 1.285 is 0.5
#Error for attribute #2 with a threshold of value > 1.795 is 0.0667
#Error for attribute #2 with a threshold of value < 1.795 is 0.5333
#Error for attribute #2 with a threshold of value > 3.245 is 0.1
#Error for attribute #2 with a threshold of value < 3.245 is 0.5
#Error for attribute #2 with a threshold of value > 3.465 is 0.1333
#Error for attribute #2 with a threshold of value < 3.465 is 0.4667
#Error for attribute #2 with a threshold of value > 5.845 is 0.1667
#Error for attribute #2 with a threshold of value < 5.845 is 0.4333
#Attribute #:3
#Error for attribute #3 with a threshold of value > -4.495 is 0.4333
#Error for attribute #3 with a threshold of value < -4.495 is 0.1667
#Error for attribute #3 with a threshold of value > -4.485 is 0.4
#Error for attribute #3 with a threshold of value < -4.485 is 0.2
#Error for attribute #3 with a threshold of value > -2.695 is 0.3667
#Error for attribute #3 with a threshold of value < -2.695 is 0.2333
#Error for attribute #3 with a threshold of value > -2.395 is 0.3333
#Error for attribute #3 with a threshold of value < -2.395 is 0.2667
#Error for attribute #3 with a threshold of value > -2.105 is 0.3
#Error for attribute #3 with a threshold of value < -2.105 is 0.3
#Error for attribute #3 with a threshold of value > -1.535 is 0.3333
#Error for attribute #3 with a threshold of value < -1.535 is 0.2667
#Error for attribute #3 with a threshold of value > -1.245 is 0.3667
#Error for attribute #3 with a threshold of value < -1.245 is 0.2333
#Error for attribute #3 with a threshold of value > 1.915 is 0.4
#Error for attribute #3 with a threshold of value < 1.915 is 0.2
#Error for attribute #3 with a threshold of value > 2.225 is 0.1
#Error for attribute #3 with a threshold of value < 2.225 is 0.5
#Error for attribute #3 with a threshold of value > 3.735 is 0.1333
#Error for attribute #3 with a threshold of value < 3.735 is 0.4667
#Error for attribute #3 with a threshold of value > 5.555 is 0.1667
#Error for attribute #3 with a threshold of value < 5.555 is 0.4333
#Therefore, the lowest error for this iteration = 0.0667
#This error was obtained from the indicator function that uses attribute #1 where the value is > -0.275
#Epsilon = 0.1111
#Alpha = 1.0397
#Data Point Predictions for this iteration:
#[ 1.  1.  1.  1.  1.  1. -1. -1.  1. -1.]
#Correct weight update factor = 0.5893
#Incorrect weight update factor = 4.714
#Iteration #:3
#Weights for this iteration:
#[0.0196 0.0196 0.0196 0.0196 0.0196 0.1571 0.0196 0.1768 0.1571 0.0196]
#Attribute #:1
#Error for attribute #1 with a threshold of value > -2.005 is 0.5303
#Error for attribute #1 with a threshold of value < -2.005 is 0.0982
#Error for attribute #1 with a threshold of value > -1.995 is 0.5107
#Error for attribute #1 with a threshold of value < -1.995 is 0.1179
#Error for attribute #1 with a threshold of value > -1.295 is 0.3339
#Error for attribute #1 with a threshold of value < -1.295 is 0.2946
#Error for attribute #1 with a threshold of value > -0.275 is 0.3143
#Error for attribute #1 with a threshold of value < -0.275 is 0.3143
#Error for attribute #1 with a threshold of value > 0.175 is 0.3339
#Error for attribute #1 with a threshold of value < 0.175 is 0.2946
#Error for attribute #1 with a threshold of value > 0.675 is 0.1768
#Error for attribute #1 with a threshold of value < 0.675 is 0.4518
#Error for attribute #1 with a threshold of value > 1.015 is 0.1964
#Error for attribute #1 with a threshold of value < 1.015 is 0.4321
#Error for attribute #1 with a threshold of value > 1.535 is 0.0393
#Error for attribute #1 with a threshold of value < 1.535 is 0.5893
#Error for attribute #1 with a threshold of value > 2.375 is 0.0589
#Error for attribute #1 with a threshold of value < 2.375 is 0.5696
#Error for attribute #1 with a threshold of value > 5.025 is 0.0786
#Error for attribute #1 with a threshold of value < 5.025 is 0.55
#Error for attribute #1 with a threshold of value > 8.525 is 0.0982
#Error for attribute #1 with a threshold of value < 8.525 is 0.5303
#Attribute #:2
#Error for attribute #2 with a threshold of value > -6.485 is 0.5303
#Error for attribute #2 with a threshold of value < -6.485 is 0.0982
#Error for attribute #2 with a threshold of value > -6.475 is 0.55
#Error for attribute #2 with a threshold of value < -6.475 is 0.0786
#Error for attribute #2 with a threshold of value > -5.815 is 0.5303
#Error for attribute #2 with a threshold of value < -5.815 is 0.0982
#Error for attribute #2 with a threshold of value > -3.445 is 0.55
#Error for attribute #2 with a threshold of value < -3.445 is 0.0786
#Error for attribute #2 with a threshold of value > -2.425 is 0.3732
#Error for attribute #2 with a threshold of value < -2.425 is 0.2553
#Error for attribute #2 with a threshold of value > -1.165 is 0.3536
#Error for attribute #2 with a threshold of value < -1.165 is 0.275
#Error for attribute #2 with a threshold of value > 1.285 is 0.1964
#Error for attribute #2 with a threshold of value < 1.285 is 0.4321
#Error for attribute #2 with a threshold of value > 1.795 is 0.0393
#Error for attribute #2 with a threshold of value < 1.795 is 0.5893
#Error for attribute #2 with a threshold of value > 3.245 is 0.0589
#Error for attribute #2 with a threshold of value < 3.245 is 0.5696
#Error for attribute #2 with a threshold of value > 3.465 is 0.0786
#Error for attribute #2 with a threshold of value < 3.465 is 0.55
#Error for attribute #2 with a threshold of value > 5.845 is 0.0982
#Error for attribute #2 with a threshold of value < 5.845 is 0.5303
#Attribute #:3
#Error for attribute #3 with a threshold of value > -4.495 is 0.5303
#Error for attribute #3 with a threshold of value < -4.495 is 0.0982
#Error for attribute #3 with a threshold of value > -4.485 is 0.5107
#Error for attribute #3 with a threshold of value < -4.485 is 0.1179
#Error for attribute #3 with a threshold of value > -2.695 is 0.3536
#Error for attribute #3 with a threshold of value < -2.695 is 0.275
#Error for attribute #3 with a threshold of value > -2.395 is 0.1964
#Error for attribute #3 with a threshold of value < -2.395 is 0.4321
#Error for attribute #3 with a threshold of value > -2.105 is 0.1768
#Error for attribute #3 with a threshold of value < -2.105 is 0.4518
#Error for attribute #3 with a threshold of value > -1.535 is 0.1964
#Error for attribute #3 with a threshold of value < -1.535 is 0.4321
#Error for attribute #3 with a threshold of value > -1.245 is 0.2161
#Error for attribute #3 with a threshold of value < -1.245 is 0.4125
#Error for attribute #3 with a threshold of value > 1.915 is 0.2357
#Error for attribute #3 with a threshold of value < 1.915 is 0.3928
#Error for attribute #3 with a threshold of value > 2.225 is 0.0589
#Error for attribute #3 with a threshold of value < 2.225 is 0.5696
#Error for attribute #3 with a threshold of value > 3.735 is 0.0786
#Error for attribute #3 with a threshold of value < 3.735 is 0.55
#Error for attribute #3 with a threshold of value > 5.555 is 0.0982
#Error for attribute #3 with a threshold of value < 5.555 is 0.5303
#Therefore, the lowest error for this iteration = 0.0393
#This error was obtained from the indicator function that uses attribute #1 where the value is > 1.535
#Epsilon = 0.0625
#Alpha = 1.354
#Data Point Predictions for this iteration:
#[-1.  1. -1.  1.  1. -1. -1. -1. -1. -1.]
#Correct weight update factor = 0.4108
#Incorrect weight update factor = 6.1619

#Test classification results:

#1.0986 * 1
#1.0397 * 1
#1.354 * 1
#Sum = 3.4924
#Test Example #1 is predicted to belong to Class 1.
#The classification was correct.

#1.0986 * 1
#1.0397 * 1
#1.354 * 1
#Sum = 3.4924
#Test Example #2 is predicted to belong to Class 1.
#The classification was correct.

#1.0986 * 1
#1.0397 * -1
#1.354 * -1
#Sum = -1.2951
#Test Example #3 is predicted to belong to Class 2.
#The classification was correct.

#1.0986 * 1
#1.0397 * -1
#1.354 * -1
#Sum = -1.2951
#Test Example #4 is predicted to belong to Class 2.
#The classification was correct.
#done
