/*	CS6200  Che Shian Hung  2/18/2018
	Programming Assignment 1
	Purpose: This program uses perceptron algorithm to train the classifiers for three classes.
			 Each sample is represented as a three dimensional vector, and there are 30 sample data
			 randomly generated for each class everytime the program runs. The first half of data
			 is used for training, and the second half of data is used for testing the classifiers
			 after training. There are three fixed sphere centers that are used to generate the data, and 
			 the distances between them have to be greater than 15. The data will then be randomly
			 generated inside each sphere with radius 2. In the beginning, the user can decide to
			 train the classifiers with separable or non-separable data. If training with 
			 non-separable data, the last sample for both training and testing for each class will
			 be swap to the next class. For instance, the samples for class 1 will now have the samples
			 for class 3.
	Architecture: There are mainly three steps in the program: data generation, perceptron algorithm,
			 and testing. Each step has been encapsulated in few functions. Also, for testing purpose,
			 each function has been designed for reusability. For instance, we can output any information
			 for a specific class.
	Data Structure: The data information like sample vectors, weight vectors, and result are all stored
			 in statically allocated arrays. The arrays are all multidimensional for easy programming 
			 purpose. All data information are set as global variables so that all the functions are
			 able to modify the information without passing any arrays into the function. In the program,
			 I defined as many as constant as opssible at the top of the program, so that we can easily
			 modify the constant for different testing purposes such as adding more sample data. However,
			 CLASSNUM and DIMENSION can only be changed after modifying the program 
*/

#define _USE_MATH_DEFINES

// Import libraries and constants
#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<cmath>
#include<string>
#define RADIUS 2
#define TESTSIZE 15
#define CLASSNUM 3
#define DIMENSION 3
#define SAMPLESIZE 30
#define WEIRDSYMBOL 0.8		// The greek symbol in gradient decent algorithm (zeta...???)
#define MODIFYINDEX 15		// The index of sample data that is chose to be non-separable while training
#define MAXITERATION 3000

using namespace std;

// Declare global constant variable
const double sphereCenter[CLASSNUM][DIMENSION] = { {0, 20, 0}, {7.5, 5, 0}, {-7.5, 5, 0} };	// Hard coded sphere centers

// Declare global variables
double classSamples[CLASSNUM][SAMPLESIZE][DIMENSION + 1];									// Sample for all three classes, including training data and testing data
double classWeights[CLASSNUM][DIMENSION + 1];												// Weights for all three classes, can be trained and untrained
int resultClass[CLASSNUM][SAMPLESIZE - TESTSIZE];											// Captures the testing result after testing with trained classifiers
bool modifyData = false;																	// A switch capture by the user to modify few data

void generateAllSamples();																	// Generate samples randomly for all classes
void generateRandomSamples(double samples[SAMPLESIZE][DIMENSION + 1], int classNum);		// Generate samples randomly for a specific class
void displayAllSamples();																	// Display samples for all classes
void displaySamples(int classNum);															// Display samples for a specific class
void displayAllWeights();																	// Display weights for all classes
void displayWeights(int classNum);															// Display weights for a specific class
void perceptron();																			// Apply perceptron algorithm and update weights
void testWeights();																			// Test the classifiers with data and update test result
void initializeWeights();																	// Initialize weights' values for all classifiers 
void displayTestResult();																	// Display test result for each class
void modifyTrainedData(int modifyIndex);													// Swap a specific sample between classes
double vectorMultiplication(double a[DIMENSION + 1], double b[DIMENSION + 1]);				// Perform vector multiplication between two vectors

int main() {
	// Capture input from user to determine if the data is linearly separable or not
	string input = "";
	do {
		if (input != "") printf("Invalid input. Please try again.\n");
		printf("Trained with non-separable data set? (y/n) ");
		cin >> input;
	} while (input != "y" && input != "n" && input != "Y" && input != "N");
	if (input == "y" || input == "Y") modifyData = true;

	srand(time(NULL));

	// Generate sample data and display for each class
	generateAllSamples();
	displayAllSamples();

	// If need to modify data, modify it and display again
	if (modifyData) {
		modifyTrainedData(MODIFYINDEX);
		modifyTrainedData(SAMPLESIZE);
		displayAllSamples();
	}

	// Perform perceptron algorithm and display the result weights for each classifier
	perceptron();
	displayAllWeights();

	// Test the tarined classifiers with second half of data and display testing result
	testWeights();
	displayTestResult();

	system("pause");
	return 0;
}

void generateAllSamples() {
	for (int i = 0; i < 3; i++) {
		generateRandomSamples(classSamples[i], i + 1);
	}
}

void generateRandomSamples(double sample[SAMPLESIZE][DIMENSION + 1], int classNum) {
	for (int j = 0; j < SAMPLESIZE; j++) {
		double theta = rand() % 6282 / double(1000);
		double phi = rand() % 3141 / double(1000) - 1.5705;

		sample[j][0] = 1;
		sample[j][1] = sphereCenter[classNum - 1][0] + RADIUS * cos(theta) * cos(phi);
		sample[j][2] = sphereCenter[classNum - 1][1] + RADIUS * sin(phi);
		sample[j][3] = sphereCenter[classNum - 1][2] + RADIUS * sin(theta) * cos(phi);
	}
}

void displayAllSamples() {
	printf("display all samples:\n");
	for (int i = 1; i < DIMENSION + 1; i++) displaySamples(i);
	printf("\n\n");
}

void displaySamples(int classNum) {
	printf("display class %d samples:\n", classNum);
	for (int i = 0; i < SAMPLESIZE; i++) {
		if ((i == MODIFYINDEX - 1 && modifyData) || (i == SAMPLESIZE - 1 && modifyData)) printf("****");
		for (int j = 0; j < DIMENSION + 1; j++) {
			if (j != 3)
				printf("%f, ", classSamples[classNum - 1][i][j]);
			else
				printf("%f\n", classSamples[classNum - 1][i][j]);
		}
	}
	printf("---------------------------------\n\n");
}

void displayAllWeights() {
	printf("display all weights:\n");
	for (int i = 1; i < CLASSNUM + 1; i++) displayWeights(i);
	printf("\n\n");
}

void displayWeights(int classNum) {
	printf("display class %d weights:\n", classNum);
	for (int i = 0; i < DIMENSION + 1; i++) {
		if (i != 3)
			printf("%f, ", classWeights[classNum - 1][i]);
		else
		printf("%f\n", classWeights[classNum - 1][i]);
	}
	printf("\n");
}

void perceptron() {
	bool allGreaterThanZero;
	initializeWeights();

	// For each classifier
	for (int i = 0; i < CLASSNUM; i++) {
		int iterationCount = 0;
		do {	
			allGreaterThanZero = true;
			iterationCount++;
			// Test with data for each class
			for (int j = 0; j < CLASSNUM; j++) {
				// Test with all the data used to train for each class
				for (int k = 0; k < TESTSIZE; k++) {
					double val = vectorMultiplication(classWeights[i], classSamples[j][k]);
					if (i != j) val = val * -1;
					if (val <= 0) {
						for (int l = 0; l < DIMENSION + 1; l++) {
							if (i != j)
								classWeights[i][l] -= classSamples[j][k][l] * WEIRDSYMBOL;
							else
								classWeights[i][l] += classSamples[j][k][l] * WEIRDSYMBOL;
						}
						allGreaterThanZero = false;
					}
				}
			}
		} while (!allGreaterThanZero && iterationCount < MAXITERATION);
		printf("%d iterations for class %d trained data.\n", iterationCount, i + 1);
	}
	printf("\n\n");
}

void testWeights() {
	for (int i = 0; i < 3; i++) {
		for (int j = TESTSIZE; j < SAMPLESIZE; j++) {
			int classNum = 1;
			double testValue[CLASSNUM];
			// Store the values after testing with each classifier, the values represent the distance between the point and each decision boundary
			for (int k = 0; k < CLASSNUM; k++) {
				testValue[k] = vectorMultiplication(classWeights[k], classSamples[i][j]);
			}
			// Find the maximum value, which is the maximum distance among three classifieres
			double max = testValue[0];
			for (int k = 1; k < CLASSNUM; k++) {
				if (max < testValue[k]) {
					max = testValue[k];
					classNum = k + 1;
				}
			}
			// Store the result class, which is the one with the maximum distance
			resultClass[i][j - TESTSIZE] = classNum;
		}
	}
}

void initializeWeights() {
	for (int i = 0; i < CLASSNUM; i++) {
		for (int j = 0; j < DIMENSION + 1; j++) {
			classWeights[i][j] = 0;
		}
	}
}

void displayTestResult() {
	printf("display test result:\n");
	for (int i = 0; i < CLASSNUM; i++) {
		printf("result for class %d: \n", i + 1);
		for (int j = 0; j < SAMPLESIZE - TESTSIZE; j++) {
			printf("%d\n", resultClass[i][j]);
		}
		printf("\n");
	}
	printf("--------------------------------\n\n");
}

void modifyTrainedData(int modifyIndex) {
	modifyIndex--;
	double class12Sample[2][DIMENSION] = { {classSamples[0][modifyIndex][1], classSamples[0][modifyIndex][2], classSamples[0][modifyIndex][3]},{ classSamples[1][modifyIndex][1], classSamples[1][modifyIndex][2], classSamples[1][modifyIndex][3] } };
	for (int i = 0; i < CLASSNUM; i++) {
		for (int j = 1; j < DIMENSION + 1; j++) {
			if (i == 0) classSamples[i][modifyIndex][j] = classSamples[2][modifyIndex][j];
			else classSamples[i][modifyIndex][j] = class12Sample[i - 1][j - 1];
		}
	}
}

double vectorMultiplication(double a[DIMENSION + 1], double b[DIMENSION + 1]) {
	double sum = 0;
	for (int i = 0; i < DIMENSION + 1; i++) {
		sum += a[i] * b[i];
	}
	return sum;
}