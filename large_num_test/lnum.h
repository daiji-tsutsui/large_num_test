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
	lVector quotient(const lVector src);
	VectorXd asVector();
	lVector& operator = (const lVector& src);
};

lVector::lVector(){}
lVector::lVector(const VectorXd d){
	exponent = d.array().log().cast<int>();
	coeff = d.array() / exponent.cast<double>().array().exp();
}
lVector lVector::quotient(const lVector src){
	lVector trg;
	
	VectorXi quot_exp = exponent - src.exponent;
	VectorXd quot_coeff = coeff.array() / src.coeff.array();
	VectorXi moveup = quot_coeff.array().log().cast<int>();
	trg.exponent = quot_exp + moveup;
	trg.coeff = quot_coeff.array() / moveup.cast<double>().array().exp();
	
	return trg;
}
VectorXd lVector::asVector(){
	VectorXd trg = coeff.array() * (exponent.cast<double>()).array().exp();
	return trg;
}
lVector& lVector::operator = (const lVector& src){
	exponent = src.exponent;
	coeff = src.coeff;
	return *this;
}

class lMatrix {
public:
	MatrixXi exponent;
	MatrixXd coeff;
	
	lMatrix();
	lMatrix(const MatrixXd d);
	lVector operator * (const lVector& src);
	lMatrix& operator = (const lMatrix& src);
};

lMatrix::lMatrix(){}
lMatrix::lMatrix(const MatrixXd d){
	exponent = d.array().log().cast<int>();
	coeff = d.array() / exponent.cast<double>().array().exp();
}
lVector lMatrix::operator * (const lVector& src){
	lVector trg;

	//rowwise product
	MatrixXd d_exp = exponent.cast<double>();
	VectorXd d_exp_src = src.exponent.cast<double>();
	MatrixXd prod_exp = d_exp.rowwise() + d_exp_src.transpose();
	MatrixXd prod_coeff = coeff.array().rowwise() * src.coeff.array().transpose();
	MatrixXi moveup = coeff.array().log().cast<int>();
	prod_exp += moveup.cast<double>();
	prod_coeff = prod_coeff.array() / moveup.cast<double>().array().exp();

	//sum-up rowwisely to compute matrix multiply
	VectorXd max_exp = prod_exp.rowwise().maxCoeff();
	MatrixXd weights = (prod_exp.colwise() - max_exp).array().exp();
	trg.exponent = max_exp.cast<int>();
	trg.coeff = (prod_coeff.array() * weights.array()).rowwise().sum();

	return trg;
}
lMatrix& lMatrix::operator = (const lMatrix& src){
	exponent = src.exponent;
	coeff = src.coeff;
	return *this;
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

int lvect_prod_demo(){
	VectorXd v(3); v << 1,2,3;
	MatrixXd a(3,3); a << 1,0,0, 1,1,1, 1,0,1;
	a = a.array().max(1e-4);
	lVector lv(v);
	lMatrix la(a);

	lVector lu;
	lu = la * lv;
	VectorXd u = lu.coeff.array() * (lu.exponent.cast<double>()).array().exp();
	PRINT_MAT(u);
	return 0;
}

int lvect_quot_demo(){
	VectorXd v(3); v << 1.5,4,9;
	VectorXd u(3); u << 1,2,3;
	lVector lv(v);
	lVector lu(u);
	
	lVector lw;
	lw = lv.quotient(lu);
	VectorXd w = lw.coeff.array() * (lw.exponent.cast<double>()).array().exp();
	PRINT_MAT2(lv.asVector(),"v");
	PRINT_MAT2(lu.asVector(),"u");
	PRINT_MAT2(w,"v.array()/u.array()");
	return 0;
}

#endif /* ldouble_h */
