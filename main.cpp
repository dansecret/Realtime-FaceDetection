/*
        Author :
        Hamdandih 
        2022
*/
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/////////// Face Detection////////////////
void main() {

	VideoCapture video(0);
	CascadeClassifier facedetect;
	Mat img, imgHSV,imgGray,imgBlur,imgCanny,mask;
	int hmin = 0, smin = 0, vmin = 0;
	int hmax = 179, smax = 255, vmax = 255;
	facedetect.load("Resources/haarcascade_frontalface_default.xml");

	if (facedetect.empty()) {
		cout << "XML file not load" << endl;
	}
	/// <summary>
	/// Trackbar initialization
	/// </summary>
	namedWindow("Trackbars", (640, 200));
	createTrackbar("Hue Min", "Trackbars", &hmin, 255);
	createTrackbar("Hue Max", "Trackbars", &hmax, 255);
	createTrackbar("sat Min", "Trackbars", &smin, 255);
	createTrackbar("sat max", "Trackbars", &smax, 255);
	createTrackbar("val Min", "Trackbars", &vmin, 255);
	createTrackbar("val Max", "Trackbars", &vmax, 255);
	while (true) {
		video.read(img);
		vector<Rect> faces;
		facedetect.detectMultiScale(img, faces, 1.3, 5);
		cvtColor(img, imgHSV, COLOR_BGR2HSV);
		cvtColor(img, imgGray, COLOR_BGR2GRAY);
		GaussianBlur(img, imgBlur, Size(3, 3), 3, 0);
		Canny(imgBlur, imgCanny, 25, 75);

		cout << faces.size() << endl;

		for (int i = 0; i < faces.size(); i++) {
			rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
			rectangle(img, Point(0,0), Point(250,60), Scalar(255, 0, 255), FILLED);
			putText(img, to_string(faces.size())+" Face Found", Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 1);
		}

		/// <summary>
		/// Trackbar
		/// </summary>
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);
		inRange(imgHSV, lower, upper, mask);

		/// <summary>
		/// Close trackbar
		/// </summary>

		imshow("Frame", img);
		imshow("HSV Video",imgHSV);
		imshow("HSV Gray", imgGray);
		imshow("HSV Blur", imgBlur);
		imshow("HSV Canny", imgCanny);
		imshow("Image Mask ", mask);
		waitKey(1);
	}
}