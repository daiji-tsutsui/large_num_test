//
//  main.cpp
//  large_num_test
//
//  Created by 筒井 大二 on 2018/10/31.
//  Copyright © 2018年 筒井 大二. All rights reserved.
//

#include "lnum.h"
#include <time.h>
using namespace std;
using namespace Eigen;
int speed_test1(int dim);
int speed_test2(int dim);

int main(int argc, const char * argv[]) {
	//Initialize
	srand((unsigned int)time(NULL));
	
	speed_test1(5000);
	
	return 0;
}

int speed_test1(int dim) {
	VectorXd log_p = VectorXd::Random(dim);
	VectorXd p = log_p.array().exp();
	VectorXd log_u = VectorXd::Random(dim);
	VectorXd u = log_u.array().exp();
	MatrixXd log_K = MatrixXd::Random(dim, dim);
	MatrixXd K = log_K.array().exp();
	VectorXd trg;
	clock_t start, end;
	
	//Simple calculation
	start = clock();
	trg = p.array() / (K * u).array();
	trg = trg.array().log();
	end = clock();
	cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
	PRINT_MAT2(trg.block(0,0,10,1),"simple");
	
	//LogSumExp
	MatrixXd log_Ku;
	VectorXd dom_Ku;
	VectorXd sub_Ku;
	start = clock();
	log_Ku = log_K.rowwise() + log_u.transpose();
	dom_Ku = log_Ku.rowwise().maxCoeff();
	sub_Ku = ((log_Ku.colwise() - dom_Ku).array().exp()).rowwise().sum();
	trg = log_p.array() - sub_Ku.array().log() - dom_Ku.array();
	end = clock();
	cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
	PRINT_MAT2(trg.block(0,0,10,1),"logsumexp");
	
	//lVector and lMatrix
	lVector lp(p);
	lVector lu(u);
	lMatrix lK(K);
	lVector ltrg;
	lVector lKu;
	start = clock();
	lKu = lK * lu;
	ltrg = lp.quotient(lKu);
	trg = ltrg.asLogVector();
	end = clock();
	cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
	PRINT_MAT2(trg.block(0,0,10,1),"lnum");
	
	return 0;
}


// absolutely the same as speed_test1 yet...
int speed_test2(int dim) {
	//
	// Suppose that only p and q are given.
	//
	
	VectorXd log_p = VectorXd::Random(dim);
	VectorXd p = log_p.array().exp();
	VectorXd log_u = VectorXd::Random(dim);
	VectorXd u = log_u.array().exp();
	MatrixXd log_K = MatrixXd::Random(dim, dim);
	MatrixXd K = log_K.array().exp();
	VectorXd trg;
	clock_t start, end;
	
	//Simple calculation
	start = clock();
	trg = p.array() / (K * u).array();
	trg = trg.array().log();
	end = clock();
	cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
	PRINT_MAT2(trg.block(0,0,10,1),"simple");
	
	//LogSumExp
	MatrixXd log_Ku;
	VectorXd dom_Ku;
	VectorXd sub_Ku;
	start = clock();
	log_Ku = log_K.rowwise() + log_u.transpose();
	dom_Ku = log_Ku.rowwise().maxCoeff();
	sub_Ku = ((log_Ku.colwise() - dom_Ku).array().exp()).rowwise().sum();
	trg = log_p.array() - sub_Ku.array().log() - dom_Ku.array();
	end = clock();
	cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
	PRINT_MAT2(trg.block(0,0,10,1),"logsumexp");
	
	//lVector and lMatrix
	lVector lp(p);
	lVector lu(u);
	lMatrix lK(K);
	lVector ltrg;
	lVector lKu;
	start = clock();
	lKu = lK * lu;
	ltrg = lp.quotient(lKu);
	trg = ltrg.asVector();
	trg = trg.array().log();
	end = clock();
	cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
	PRINT_MAT2(trg.block(0,0,10,1),"lnum");
	
	return 0;
}
