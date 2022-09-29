/*
* *********************************************************
* Program shows OpenCV's background subtraction through:
* 1) frame difference
* 2) adaptive background (weighted average)
* 3) Mixture Of Gaussians (MOG2)
* *********************************************************
*/

//C++
#include <iostream>
#include <sstream>
#include <limits>
#include <vector>
#include <array>
//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Global variables
const int CAMERA_ID = 0;
const int NUM_FRAMES_DIFFERENCE = 20;
const float LEARNING_RATE_ALPHA = 0.05f;
const float LEARNING_RATE_MOG = 0.05f;



int UserInputInRange(int,int);
void FrameDifference();
void AdaptiveBackground();
void MixtureOfGaussians();



int main()
{
	std::cout <<	 "Background subtractor program awakens ..."	<< std::endl;

	// App loop
	while (true) {
		std::cout << "Available background subtraction algorithms:" << std::endl;
		std::cout << "1) frame difference"							<< std::endl;
		std::cout << "2) adaptive background through alpha value"	<< std::endl;
		std::cout << "3) Mixture of Gaussians (MOG2) method"		<< std::endl;

		int userChoice = UserInputInRange(1,3);

		if (userChoice == 1) FrameDifference();
		if (userChoice == 2) AdaptiveBackground();
		if (userChoice == 3) MixtureOfGaussians();
		if (userChoice == 4) break;
	}

	return EXIT_SUCCESS;
}

/*
************************************************************
* User Interface
* **********************************************************
*/

int UserInputInRange(int min, int max) {
	int userChoice = 0;
	do {
		printf("Please choose number between %d and %d\n", min, max);
		std::cin >> userChoice;
	} while (userChoice < min || userChoice > max);
	return userChoice;
}


/*
************************************************************
* Frame Difference
* **********************************************************
*/

void FrameDifference() {
	Mat frame, frameGray;
	Mat difference, tresholdedDiff;
	const int arrayTolerance = 80;
	const int arrayDim = arrayTolerance + NUM_FRAMES_DIFFERENCE;
	std::array<Mat, arrayDim> buffer;
	int keyboardInput = 0;
	
	// Spawn GUI Windows
	cv::namedWindow("Frame");
	cv::namedWindow("Old Frame");
	cv::namedWindow("Difference");
	cv::namedWindow("Motion");
	cv::moveWindow("Frame", 100, 100);
	cv::moveWindow("Old Frame", 600, 100);
	cv::moveWindow("Difference", 100, 600);
	cv::moveWindow("Motion", 600, 600);

	// Open Webcam
	VideoCapture capture(0); 
	if (!capture.isOpened()) {
		std::cerr << "[FAIL] Cannot open webcam." << std::endl;
		exit(EXIT_FAILURE);
	}

	unsigned int currFrame = 0;
	unsigned int oldFrame = 0;

	// Loop, exit if user press 'esc' or 'q' 
	while ((char)keyboardInput != 'q' && (char)keyboardInput != 27) {

		capture >> frame; // store frames
		
		cv::cvtColor(frame, frameGray, cv::COLOR_RGB2GRAY);

		buffer.at(currFrame) = frameGray;
		
		std::cout << "Current frame -> " << currFrame << std::endl;
		
		cv::imshow("Frame", buffer.at(currFrame));

		// handle skipping first frames
		if (currFrame >= NUM_FRAMES_DIFFERENCE) { 
			
			// handle buffer as circular
			if (currFrame >= arrayDim - 1) currFrame = 0;
			if (oldFrame >= arrayDim - 1) oldFrame = 0;

			std::cout << "Old frame     -> " << oldFrame << std::endl;
			cv::imshow("Old Frame", buffer.at(oldFrame));

			cv::absdiff(frameGray, buffer.at(oldFrame), difference); 
			//cv::subtract(buffer.at(currFrame), buffer.at(oldFrame), difference); 

			cv::imshow("Difference", difference);

			cv::threshold(difference, tresholdedDiff, 50, 255, cv::THRESH_BINARY);

			//results display
			cv::imshow("Motion", tresholdedDiff);

			oldFrame++;

		}

		currFrame++;

		//get input from keyboard
		keyboardInput = waitKey(30);


	}

	
	capture.release(); //delete capture object
	cv::destroyAllWindows(); //destroy GUI windows
}


/*
************************************************************
* Adaptive Background
* **********************************************************
*/

static int ctr = 1;
void bg_train(Mat frame, Mat* background) {
	if (ctr == 1) {
		printf("initial background storage..\n");
		frame.copyTo(*background);
	}
	ctr++;
}


void bg_update(Mat frame, Mat* background) {
	*background = LEARNING_RATE_ALPHA * frame + *background * (1.0f - LEARNING_RATE_ALPHA);
}

void AdaptiveBackground() {
	Mat frame; //current frame
	Mat frameGray, bg;
	Mat motionMask, motionThres;

	//create GUI windows
	namedWindow("Frame");
	namedWindow("Motion Mask");
	namedWindow("Background");

	// Open Webcam
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		std::cerr << "[FAIL] Cannot open webcam." << std::endl;
		exit(EXIT_FAILURE);
	}

	int keyboardInput = 0;

	// Main loop, exit if user press 'esc' or 'q' 
	while ((char)keyboardInput != 'q' && (char)keyboardInput != 27) {

		capture >> frame;

		//color conversion to gray
		cvtColor(frame, frameGray, COLOR_RGB2GRAY);
		//store first frame as background
		bg_train(frameGray, &bg);
		//bg subtraction
		absdiff(bg, frameGray, motionMask);
		//mask thresholding
		threshold(motionMask, motionThres, 50, 255, THRESH_BINARY);
		//set the current frame as background for the next frame
		bg_update(frameGray, &bg);
		//get input from keyboard
		keyboardInput = waitKey(30);

		imshow("Frame", frame);
		imshow("Motion Mask", motionMask);
		imshow("Background", bg);
	}

	//delete capture object
	capture.release();

	//destroy GUI windows
	destroyAllWindows();
}


/*
************************************************************
* Mixture Of Gaussians (MOG2)
* **********************************************************
*/

void MixtureOfGaussians() {
	Mat frame; //current frame
	Mat fgMaskMOG2; //fg mask generated by MOG2 method
	Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
	//create GUI windows
	namedWindow("Frame");
	namedWindow("Foreground Mask");
	//create Background Subtractor objects
	pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
	// Open Webcam
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		std::cerr << "[FAIL] Cannot open webcam." << std::endl;
		exit(EXIT_FAILURE);
	}
	int keyboardInput = 0;
	// Main loop, exit if user press 'esc' or 'q' 
	while ((char)keyboardInput != 'q' && (char)keyboardInput != 27) {
		capture >> frame;
		//update the background model
		pMOG2->apply(frame, fgMaskMOG2, LEARNING_RATE_MOG);
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("Foreground Mask", fgMaskMOG2);
		//get input from keyboard
		keyboardInput = waitKey(30);
	}
	//delete capture object
	capture.release();
	//destroy GUI windows
	destroyAllWindows();
}



