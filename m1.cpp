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

#define IMAX std::numeric_limits<double>::max()

struct pixel {
	unsigned short x; unsigned short y; int value;
};

struct seam {
	vector<pixel> pixels;
	int totalEnergy;
};

seam 	bestSeamX (Mat energy);
int		bestSeamX_helper (Mat &M, Mat &energy, int i, int j);
seam 	bestSeamY (Mat energy);
int		bestSeamY_helper (Mat &M, Mat &energy, int i, int j);
int 	minValue (int x1, int x2, int x3);

int main (int argc, char **argv) {
	int i, j, n;
	Mat img, gaussImg, sobelx, sobely, energy;
	if (argc < 2)
		{cout<<"Usage: ./q1 Image_file\n"; return -1;}
	
	// img = imread(argv[1]);
	// cvtColor(img, img, CV_RGB2GRAY);
	// imshow("Original Image", img);
	// GaussianBlur(img, gaussImg, Size(3,3), 1);
	// imshow("Gaussian New", gaussImg);
	// waitKey(0);
	// destroyWindow("Original Image");
	// destroyWindow("Gaussian New");

	// Sobel(gaussImg, sobelx, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	// convertScaleAbs(sobelx, sobelx);
	// Sobel(gaussImg, sobely, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	// convertScaleAbs(sobely, sobely);
	// imshow("Sobel x", sobelx);
	// imshow("Sobel y", sobely);
	// waitKey(0);
	// destroyWindow("Sobel x");
	// destroyWindow("Sobel y");

	// addWeighted(sobelx, 0.5, sobely, 0.5, 0, energy);
	// imshow("Energy", energy);
	// waitKey(0);
	// destroyWindow("Energy");
	Mat dummy = (Mat_<double>(5, 5) << 5.5, 16, 22.5, 13, 16, 6.5, 8, 11.5, 20, 6.5, 2.5, 9.5, 7, 7.5, 4, 5.5, 3.5, 5.5, 10, 4.5, 6.5, 3.5, 9.5, 9, 5.5);
	bestSeamY(dummy);
	return 0;

}

seam bestSeamY (Mat energy) {
	Mat M = Mat::ones(energy.size(), CV_64F);
	for (int i=0; i<energy.rows; i++) {
		for (int j=0; j<energy.rows; j++) {
			M.at<double>(i,j) = -1;
		}
	}
	int minimum = IMAX, index, temp;
	for (int i=0; i<energy.cols; i++) {
		temp = bestSeamY_helper(M, energy, energy.rows-1, i);
		if (temp < minimum) {
			minimum = temp;
			index = i;
		}
	}
	cout<<M;
	cout<<"\nMin "<<minimum<<"; minIndex "<<index<<endl;
	struct seam S;
	return S;
}

int	bestSeamY_helper (Mat &M, Mat &energy, int i, int j) {
	if (M.at<double>(i,j) != -1) {
		// cout<<"wtf?\n";
		return M.at<double>(i,j);
	}
	if (i==0){
		M.at<double>(i,j) = energy.at<double>(i,j);
		return energy.at<double>(i,j);
	}
	double min1 = IMAX, min2=IMAX, min3=IMAX;
	if (j-1 >= 0) {
		min1 = bestSeamY_helper(M, energy, i-1, j-1);
	}
	if (j+1 < energy.cols) {
		min3 = bestSeamY_helper(M, energy, i-1, j+1);
	}
	min2 = bestSeamY_helper(M, energy, i-1, j);

	M.at<double>(i,j) = energy.at<double>(i,j) + minValue(min1, min2, min3);
	// cout<<"M("<<i<<", "<<j<<") "<<M.at<double>(i,j)<<endl;
	cout<<"energy "<<energy.at<double>(i,j)<<"; minValue "<<minValue(min1, min2, min3)<<endl;
	return M.at<double>(i,j);
}

int minValue (int x1, int x2, int x3) {
	int minVal = x1;
	if (minVal < x2) {
		minVal = x2;
	}
	if (minVal < x3) {
		minVal = x3;
	}
	return minVal;
}