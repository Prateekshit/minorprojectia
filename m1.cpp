#include <iostream>
#include <cmath>
#include <array>
#include <string>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

#define IMAX 2147483641
#define RESIZING_FACTOR 2
#define X_SEAMS (int)img.cols*(RESIZING_FACTOR - 1)
#define Y_SEAMS (int)img.rows*(RESIZING_FACTOR - 1)

struct pixel {
	int x; int y;
};

struct seam {
	vector<pixel> pixels;
	int totalEnergy;
};

seam 	bestSeamX (Mat energy);
int		bestSeamX_helper (Mat &M, Mat &J, Mat &energy, int i, int j);
seam 	bestSeamY (Mat energy);
int		bestSeamY_helper (Mat &M, Mat &J, Mat &energy, int i, int j);
int 	minNeighbor (int x1, int x2, int x3, int i, int j, Mat &J);
void 	removeSeamY (Mat &img, seam &s);
void 	removeSeamX (Mat &img, seam &s);
Mat 	removeSeamYColor (Mat &img, seam &s);

int main (int argc, char **argv) {
	int i, j, n;
	Mat img, gaussImg, sobelx, sobely, energy, colorImg;
	if (argc < 2)
		{cout<<"Usage: ./q1 Image_file\n"; return -1;}
	
	img = imread(argv[1]);
	if (!img.data) {
		cout<<"Image not read!"<<endl;
		return -1;
	}
	cout << "image read successfully" << endl;
	colorImg = img;
	imshow("Original Image", colorImg);
	cvtColor(img, img, CV_RGB2GRAY);
	resize(img, img, Size(), RESIZING_FACTOR, RESIZING_FACTOR, INTER_CUBIC);
	resize(colorImg, colorImg, Size(), RESIZING_FACTOR, RESIZING_FACTOR, INTER_CUBIC);
	
	imshow("Resized", colorImg);
	imwrite("resized.jpg", colorImg);
	// // Mat dummy = (Mat_<double>(5, 5) << 5.5, 16, 22.5, 13, 16, 6.5, 8, 11.5, 20, 6.5, 2.5, 9.5, 7, 7.5, 4, 5.5, 3.5, 5.5, 10, 4.5, 6.5, 3.5, 9.5, 9, 5.5);
	
	Mat planes[3];
	cout << "processing";
	for (int i=0; i<Y_SEAMS; i++) {
		cout << ".";
		GaussianBlur(img, gaussImg, Size(3,3), 1);
		Sobel(gaussImg, sobelx, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(sobelx, sobelx);
		Sobel(gaussImg, sobely, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(sobely, sobely);
		addWeighted(sobelx, 0.5, sobely, 0.5, 0, energy);
		energy.convertTo(energy, CV_64FC1);
		struct seam SY = bestSeamY(energy);
		removeSeamY(img, SY);
		img = Mat(img, Range::all(), Range(0,img.cols-1));

		split(colorImg, planes);
		removeSeamY(planes[0], SY);
		removeSeamY(planes[1], SY);
		removeSeamY(planes[2], SY);
		planes[0] = Mat(planes[0], Range::all(), Range(0, planes[0].cols-1));
		planes[1] = Mat(planes[1], Range::all(), Range(0, planes[1].cols-1));
		planes[2] = Mat(planes[2], Range::all(), Range(0, planes[2].cols-1));
		merge(planes, 3, colorImg);
	}
	cout << endl;
	for (int i=0; i<X_SEAMS; i++) {
		cout << ".";
		GaussianBlur(img, gaussImg, Size(3,3), 1);
		Sobel(gaussImg, sobelx, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(sobelx, sobelx);
		Sobel(gaussImg, sobely, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(sobely, sobely);
		addWeighted(sobelx, 0.5, sobely, 0.5, 0, energy);
		energy.convertTo(energy, CV_64FC1);
		struct seam SX = bestSeamX(energy);
		removeSeamX(img, SX);
		img = Mat(img, Range(0, img.rows-1), Range::all());

		split(colorImg, planes);
		removeSeamX(planes[0], SX);
		removeSeamX(planes[1], SX);
		removeSeamX(planes[2], SX);
		planes[0] = Mat(planes[0], Range(0, planes[0].rows-1), Range::all());
		planes[1] = Mat(planes[1], Range(0, planes[1].rows-1), Range::all());
		planes[2] = Mat(planes[2], Range(0, planes[2].rows-1), Range::all());
		merge(planes, 3, colorImg);
	}
	imwrite("output.jpg", colorImg);
	imshow("Output", colorImg);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

seam bestSeamY (Mat energy) {
	Mat M = Mat::ones(energy.size(), CV_64F);
	Mat J = Mat::ones(energy.size(), CV_64F);
	M = -1*M;	J = 2*J;
	int minimum = bestSeamY_helper(M, J, energy, energy.rows-1, 0), index, temp;
	for (int i=1; i<energy.cols; i++) {
		temp = bestSeamY_helper(M, J, energy, energy.rows-1, i);
		if (temp < minimum) {
			minimum = temp;
			index = i;
		}
	}
	struct seam S;
	int colIndex = index;
	// cout<<"Col Index "<<colIndex<<endl;
	struct pixel p = {(energy.rows-1), colIndex};
	S.pixels.push_back(p);	S.totalEnergy = energy.at<double>(energy.rows-1, colIndex);
	for (int i=energy.rows-1; i>0; i--) {
		int prev = colIndex;
		colIndex += (int)J.at<double>(i, prev);
		// cout<<"J "<<(int)J.at<double>(i, prev)<<endl;
		struct pixel p = {(int)(i-1), (int)colIndex};
		S.pixels.push_back(p);
		S.totalEnergy += energy.at<double>(i-1, colIndex);
	}
	return S;
}

int	bestSeamY_helper (Mat &M, Mat &J, Mat &energy, int i, int j) {
	if (M.at<double>(i,j) != -1) {
		return M.at<double>(i,j);
	}
	if (i==0){
		M.at<double>(i,j) = energy.at<double>(i,j);
		return energy.at<double>(i,j);
	}
	double min1 = IMAX, min2=IMAX, min3=IMAX;
	if (j-1 >= 0) {
		min1 = bestSeamY_helper(M, J, energy, i-1, j-1);
	}
	if (j+1 < energy.cols) {
		min3 = bestSeamY_helper(M, J, energy, i-1, j+1);
	}
	min2 = bestSeamY_helper(M, J, energy, i-1, j);

	M.at<double>(i,j) = energy.at<double>(i,j) + minNeighbor(min1, min2, min3, i, j, J);
	return M.at<double>(i,j);
}

seam bestSeamX (Mat energy) {
	Mat M = Mat::ones(energy.size(), CV_64F);
	Mat J = Mat::ones(energy.size(), CV_64F);
	M = -1*M;	J = 2*J;
	int minimum = bestSeamX_helper(M, J, energy, 0, energy.cols-1), index, temp;
	for (int i=1; i<energy.rows; i++) {
		temp = bestSeamX_helper(M, J, energy, i, energy.cols-1);
		if (temp < minimum) {
			minimum = temp;
			index = i;
		}
	}
	struct seam S;
	int rowIndex = index;
	struct pixel p = {rowIndex, (energy.rows-1)};
	S.pixels.push_back(p);	S.totalEnergy = energy.at<double>(rowIndex, energy.rows-1);
	for (int i=energy.cols-1; i>0; i--) {
		int prev = rowIndex;
		rowIndex += (int)J.at<double>(prev, i);
		struct pixel p = {rowIndex, i-1};
		S.pixels.push_back(p);
		S.totalEnergy += energy.at<double>(rowIndex, i-1);
	}
	return S;
}

int	bestSeamX_helper (Mat &M, Mat &J, Mat &energy, int i, int j) {
	if (M.at<double>(i,j) != -1) {
		return M.at<double>(i,j);
	}
	if (j==0){
		M.at<double>(i,j) = energy.at<double>(i,j);
		return energy.at<double>(i,j);
	}
	double min1 = IMAX, min2=IMAX, min3=IMAX;
	if (i-1 >= 0) {
		min1 = bestSeamX_helper(M, J, energy, i-1, j-1);
	}
	if (i+1 < energy.cols) {
		min3 = bestSeamX_helper(M, J, energy, i+1, j-1);
	}
	min2 = bestSeamX_helper(M, J, energy, i, j-1);

	M.at<double>(i,j) = energy.at<double>(i,j) + minNeighbor(min1, min2, min3, i, j, J);
	return M.at<double>(i,j);
}

int minNeighbor (int x1, int x2, int x3, int i, int j, Mat &J) {
	int minVal = x1, 	pos = -1;
	if (x2 < minVal) {
		minVal = x2;	pos = 0;
	}
	if (x3 < minVal) {
		minVal = x3;	pos = 1;
	}
	J.at<double>(i,j) = pos;
	return minVal;
}

void removeSeamY (Mat &img, seam &s) {
	
	int initx, inity;
	for (int i=0; i<s.pixels.size(); i++) {
		initx = s.pixels[i].x;	inity = s.pixels[i].y;
		for (int j=inity+1; j<img.cols; j++) {
			img.at<uchar>(initx, j-1) = img.at<uchar>(initx, j);
		}
	}
}

void removeSeamX (Mat &img, seam &s) {
	int initx, inity;
	for (int i=0; i<s.pixels.size(); i++) {
		initx = s.pixels[i].x;	inity = s.pixels[i].y;
		for (int j=initx+1; j<img.rows; j++) {
			img.at<uchar>(j-1, inity) = img.at<uchar>(j, inity);
		}
	}
}