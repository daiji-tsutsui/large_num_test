//
//  ldouble.h
//  large_num_test
//
//  Created by 筒井 大二 on 2018/10/31.
//  Copyright © 2018年 筒井 大二. All rights reserved.
//
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <cmath>
#define PRINT_MAT(X) cout << #X << ":\n" << X << endl << endl
#define PRINT_MAT2(X,DESC) cout << DESC << ":\n" << X << endl << endl
#define PRINT_FNC    cout << "[" << __func__ << "]" << endl
using namespace std;
using namespace Eigen;

#ifndef ldouble_h
#define ldouble_h

class lVector {
public:
	VectorXi exponent;
	VectorXd coeff;
	
	lVector();
	lVector(const VectorXd d);
//	lVector& operator = (double d);
//	lVector& operator * (const double r);
//	lVector& operator + (const lVector& r);
//	lVector& operator += (const lVector& r);
};

lVector::lVector(){}
lVector::lVector(const VectorXd d){
	exponent = d.array().log().cast<int>();
	coeff = d.array() / exponent.cast<double>().array().exp();
}

class lMatrix {
public:
	MatrixXi exponent;
	MatrixXd coeff;
	
	lMatrix();
	lMatrix(const MatrixXd d);
};

lMatrix::lMatrix(){}
lMatrix::lMatrix(const MatrixXd d){
	exponent = d.array().log().cast<int>();
	coeff = d.array() / exponent.cast<double>().array().exp();
}

/*--- DEMO FUNCTIONS ---*/
int frexp_demo() {
	int exp_part = 0;
	double coeff = 0.0;
	
	coeff = frexp(12345.67890, &exp_part);
	cout << "exp_part:\t" << exp_part << endl;
	cout << "coeff:\t" << coeff << endl;
	cout << setprecision(10) << ldexp(coeff, exp_part) << endl;
	
	return 0;
}

int lvect_demo(){
	VectorXd src = VectorXd::Random(2); src = src.array() - src.minCoeff() + 0.1;
	PRINT_MAT(src);
	
	lVector test(src);
	PRINT_MAT(test.exponent);
	PRINT_MAT(test.coeff);
	
	VectorXd trg = test.coeff.array() * (test.exponent.cast<double>()).array().exp();
	PRINT_MAT(trg);
	return 0;
}

#endif /* ldouble_h */
