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
//opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Global variables
const int CAMERA_ID = 0;
const float LEARNING_RATE_MOG = 0.05f;
const float LEARNING_RATE_ALPHA = 0.05f;


int UserInputInRange(int,int);
void FrameDifference();
void AdaptiveBackground();
void MixtureOfGaussians();



int main()
{
	std::cout <<	 "Background subtractor program awakens ..."	<< std::endl;

	// App main loop
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


int UserInputInRange(int min, int max) {
	int userChoice = 0;
	do {
		printf("Please choose number between %d and %d\n", min, max);
		std::cin >> userChoice;
	} while (userChoice < min || userChoice > max);
	return userChoice;
}


//creates needed GUI windows
void SpawnNeededWindows() { 
	namedWindow("Frame");
	namedWindow("Foreground Mask");
	namedWindow("Background");
}


void FrameDifference() {
	Mat frame; //current frame
	Mat frameGray, bg;
	Mat D1, b1, motionMask;
	Mat* Pic = new Mat[1000];
	int keyboardInput = 0;
	
	SpawnNeededWindows();

	VideoCapture capture(0); // Open Webcam
	if (!capture.isOpened()) {
		std::cerr << "[FAIL] Cannot open webcam." << std::endl;
		exit(EXIT_FAILURE);
	}

	const unsigned int frameGap = 3;

	unsigned int currFrame = frameGap;
	unsigned int oldFrame = 0;

	// Loop, exit if user press 'esc' or 'q' 
	while ((char)keyboardInput != 'q' && (char)keyboardInput != 27) {

		capture >> frame; // store frames

		// handle skipping first frames & int overflow
		if (currFrame < frameGap) { continue;}
		if (currFrame == UINT_MAX) {

		}

		// handle int overflow
		//if currFrame ==



		cvtColor(frame, frameGray, cv::COLOR_RGB2GRAY);

		frameGray.copyTo(Pic[currFrame]);
		currFrame++;
		oldFrame++;

		if (currFrame == 0) {
			currFrame = 3;
		}

		if (currFrame > oldFrame) //need at least 3 frame (for 3-frame difference)
		{

			absdiff(Pic[currFrame], Pic[oldFrame], D1);

			//mask thresholding
			threshold(D1, b1, 50, 255, THRESH_BINARY);

			motionMask = b1;

			//results display
			cv::imshow("Frame", Pic[currFrame]);
			cv::imshow("Foreground Mask", motionMask);

			waitKey(30);
		}

		//get input from keyboard
		keyboardInput = waitKey(30);
	}

	
	capture.release(); //delete capture object
	cv::destroyAllWindows(); //destroy GUI windows
}

static int ctr = 1;
void bg_train(Mat frame, Mat* background) {
	if (ctr == 1) {
		printf("initial background storage..\n");
		frame.copyTo(*background);
	}
	ctr++;
}


void bg_update(Mat frame, Mat* background) {
	float alfa = 0.05;
	*background = alfa * frame + *background * (1.0 - alfa);
}

void AdaptiveBackground() {
	Mat frame; //current frame
	Mat frameGray, bg;
	Mat motionMask, motionThres;
	//Mat* Pic = new Mat[1000];

	SpawnNeededWindows();

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
		imshow("Foreground Mask", motionMask);
		imshow("Background", bg);
	}

	//delete capture object
	capture.release();

	//destroy GUI windows
	destroyAllWindows();
}


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



