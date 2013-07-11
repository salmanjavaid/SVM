#ifndef _SVM_1_H
#define _SVM_1_H


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <fstream>




using namespace cv;
using namespace std;


class SVM_1 {

private: 
	
	
	

	

public:

	/* Different Kernel types */

	enum Kernel {
	Gaussian, 
	RBF,
	Linear,
	Wavelet
	};

	/* This struct is used for the Model yielded by SVM Train */
	typedef struct {
	
		Mat_<double> X, y, alphas, weights;
		double b;
		Kernel Type;
	} Model;

	

	Model M; /* A variable of type Model which will be used in Prediction */
	
	void Train_(const Mat_<double>&, const Mat_<double>&, const double&, Kernel, const double&, const int&);  /*Training function by Platt's SMO*/
	void Classify_(const Model&, Mat_<double>&, Kernel, Mat_<double>&); /*Classification*/
	void Accuracy_(Mat&, Mat&); /*This measures accuracy of SVM*/
	void Kernel_Compute_(const Mat_<double> &, Mat_<double>&, Kernel); /*Computes the kernel*/
	void Replace(const Mat_<double>&, Mat_<double>&); /*Replaces 0s' with -1 in Training and Test examples*/


};




#endif