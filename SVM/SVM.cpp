#ifndef _SVM_1_H
#include "SVM_1.hpp"
#endif


/*Compute the Kernel, right now only Linear Kernel is present*/
void SVM_1::Kernel_Compute_(const Mat_<double> &src, Mat_<double> &dst, Kernel Type)
{
	if (Type == Linear)
	{
		Mat_<double> _temp_;              
		transpose(src, _temp_);
		dst = src * _temp_;
	}
}

/* function to replace targets values with zeros to -1 */

void SVM_1::Replace(const Mat_<double> &src, Mat_<double> &dst)
{
	for(int _i_ = 0; _i_ < src.rows; _i_++){
		if (src(_i_, 0) == 0.00){  
			dst(_i_, 0) = -1.00;
		}
		else{
			dst(_i_, 0) = src(_i_, 0);	
		}
	}
}


/* this is a simpler version of Platt's SMO algorithm as presented by Ng's class notes */

void SVM_1::Train_(const Mat_<double>& X, const Mat_<double>& Y, const double& C, Kernel Type, const double& TOI, const int& max_passes)
{

	
	int _Rows_ = X.rows;
	int _Cols_ = X.cols;


	Mat_<double> _alphas_(_Rows_, 1, 0.00);
	Mat_<double> _E_(_Rows_, 1, 0.00);
	Mat_<double> _K_(_Rows_, _Cols_, 0.00);
	int _passes_ = 0;
	double _eta_ = 0, _L_ = 0, _H_ = 0, _b_ = 0, _b_1_ = 0, _b_2_ = 0, u = 0;
	double _alpha_i_old_ = 0, _alpha_j_old_ = 0;
	
	 
	Kernel_Compute_(X, _K_, Type);
	Mat_<double> y(Y.rows, Y.cols, CV_32F);
	Replace(Y, y);

	while(_passes_ < max_passes)
	{
		int _number_of_changed_alphas_ = 0;

		for (int _i_ = 0; _i_ < _Rows_; _i_++)
		{
  			Mat_<double> _temp_;
			multiply(_alphas_, y, _temp_);

			

			multiply(_K_.col(_i_), _temp_, _temp_);
			
			Scalar k = sum(_temp_);
			u = k[0];
			_E_.row(_i_).col(0) = _b_ + u - y(_i_, 0);


			

			Mat_<double> _temp_1_ = y.row(_i_) * _E_.row(_i_);
			double t1 = _temp_1_(0, 0);
			double t2 = -1*TOI;
			_temp_1_ = y.row(_i_) * _E_.row(_i_);
			double t3 = _temp_1_(0, 0);
			double t4 = _alphas_(_i_, 0);

		
			
			if ( ( t1 < t2 )  && ( t3 < C ) ||  ( ( t1 > TOI ) && ( t4 > 0 ) ) )
			{
				int _j_ = 0;

				_j_ = rand() % _Rows_ ;
				while (_j_ == _i_) {
	                _j_ = rand() % _Rows_;
				}
			
				multiply(_alphas_, y, _temp_);
				multiply(_K_.col(_j_), _temp_, _temp_);

			
				Scalar k = sum(_temp_);
				u = k[0];
				_E_.row(_j_) = _b_ + u - y.row(_j_);

				_alpha_i_old_ = _alphas_(_i_, 0);
				_alpha_j_old_ = _alphas_(_j_, 0);

				//cout<<_E_(_j_, 0)<<endl;
				if ( y(_i_, 0) == y(_j_, 0))
				{
					_L_ = std::max(0.00, _alphas_(_i_, 0) +  _alphas_(_j_, 0) - C);
					_H_ = std::min(C, _alphas_(_i_, 0) +  _alphas_(_j_, 0));
				}
				else
				{
					_L_ = std::max(0.00, _alphas_(_i_, 0) - _alphas_(_j_, 0));
					_H_ = std::min(C, C + _alphas_(_i_, 0) - _alphas_(_j_, 0));
				}

				if (_L_ == _H_)
				{
					continue;
				}

				_eta_ = 2 * _K_(_i_, _j_) - _K_(_i_, _i_) - _K_(_j_, _j_);

				if (_eta_ >= 0)
				{
					continue;
				}

				_alphas_(_j_, 0) = _alphas_(_j_, 0) - (  y(_j_, 0) * ( _E_(_i_, 0) - _E_(_j_, 0) ) / _eta_ );

				//cout<<_alphas_(_j_, 0)<<endl;

				_alphas_(_j_, 0) = std::min(_H_, _alphas_(_j_, 0));
				_alphas_(_j_, 0) = std::max(_L_, _alphas_(_j_, 0));

				

				if (std::abs( _alphas_(_j_, 0) - _alpha_j_old_ ) < TOI)
				{
					_alphas_(_j_, 0) = _alpha_j_old_;
					continue;
				}




				_alphas_(_i_, 0) = _alphas_(_i_, 0) + ( y(_i_, 0) * y(_j_, 0) * (_alpha_j_old_ - _alphas_(_j_, 0)));

				
				transpose(_K_, _K_);
				
				_b_1_ = _b_ - _E_(_i_, 0) - (y(_i_, 0) * ( _alphas_(_i_, 0) - _alpha_i_old_ ) * _K_(_i_,_j_) ) 
					- (y(_j_, 0) * ( _alphas_(_j_, 0) - _alpha_j_old_ ) * _K_(_i_,_j_) ) ;


				_b_2_ =  _b_ - _E_(_j_, 0) - (y(_i_, 0) * ( _alphas_(_i_, 0) - _alpha_i_old_ ) * _K_(_i_,_j_) ) 
					- (y(_j_, 0) * ( _alphas_(_j_, 0) - _alpha_j_old_ ) * _K_(_j_,_j_) ) ;

				transpose(_K_, _K_);
				
				if  (_alphas_(_i_, 0) > 0 && _alphas_(_i_, 0) < C) 
				{
					_b_ = _b_1_;
				}
				else if (_alphas_(_j_, 0) > 0 && _alphas_(_j_, 0) < C) 
				{
					_b_ = _b_2_;
				}	
				else
				{
					_b_ = (_b_1_ + _b_2_) / 2;
				}


				_number_of_changed_alphas_++;

			}

		}
		
		if (_number_of_changed_alphas_ == 0)
		{
			_passes_++;
		}
		else
		{
			_passes_ = 0;
		}
		
	}
	
	int _Index_ = 0;
	
	for (int _i_ = 0 ; _i_ < _Rows_; _i_++)
	{
		if( _alphas_(_i_, 0) > 0 )
		{
			_Index_++;
		}
	}
	
	Mat_<double> _X_(_Index_, 1, CV_32F);
	Mat_<double> _Y_(_Index_, 1, CV_32F);
	Mat_<double> _ALPHA_(_Index_, 1, CV_32F);

	_Index_ = 0;
	for (int _i_ = 0 ; _i_ < _Rows_; _i_++)
	{
		if( _alphas_(_i_, 0) > 0 )
		{
			_X_(_Index_, 0) = X(_i_, 0);
			_Y_(_Index_, 0) = y(_i_, 0);
			_ALPHA_(_Index_, 0) = _alphas_(_i_, 0);
			_Index_++;
		}
	}
	


	this->M.X = _X_;
	this->M.y = _Y_;
	this->M.alphas = _ALPHA_;

	
	this->M.Type = Type;
	this->M.b = _b_;

	multiply(_alphas_, y, this->M.weights);
	transpose(this->M.weights, this->M.weights);
	this->M.weights = this->M.weights * X;

}


/* classification function, only Linear Kernel present right now */

void SVM_1::Classify_(const Model& M, Mat_<double>& _Test_, Kernel Type, Mat_<double>& p)
{
		if (Type == Linear)
		{
			p = _Test_ * M.weights + M.b;
		}

}




