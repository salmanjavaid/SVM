#include "SVM_1.hpp"


int main()
{
	CvMLData Data;
	
	/* Load data */
	Data.read_csv("D:\\CData.csv");
	

	Mat DataMat = Data.get_values();
	Mat_<double> X = DataMat.colRange(0,2); /* The first two columns are examples */
	Mat_<double> y = DataMat.col(2);	/* the last column are zeros */


	SVM_1 SVM_Obj;
	
	SVM_Obj.Train_(X, y, 200.00, SVM_1::Linear, 0.001, 20); //Train data

	
	return 0;

}