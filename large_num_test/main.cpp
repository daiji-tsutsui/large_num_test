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
#define LOGSUMEXP	0
#define LNUM		1
int speed_test1(int dim);
int speed_test2(int dim, int itrNum);

int main(int argc, const char * argv[]) {
	//Initialize
	srand((unsigned int)time(NULL));
	
	speed_test2(1000,1000);
	
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

inline int speed_test2(int dim, int itrNum) {
	//
	// Speed-check for Sinkhorn algorithm.
	//

	// Two (non-singular) probability distributions are given.
	VectorXd log_p = VectorXd::Random(dim);
	VectorXd p = log_p.array().exp(); p /= p.sum();
	VectorXd log_q = VectorXd::Random(dim);
	VectorXd q = log_q.array().exp(); q /= q.sum();

	// A kernel matrix is also given.
	MatrixXd log_K = MatrixXd::Random(dim, dim);
	MatrixXd K = log_K.array().exp();

	// The solutions are iteratively computed from a trivial vectors as initial values.
	VectorXd iniu = VectorXd::Ones(dim); iniu /= (double)dim;
	VectorXd u, v;
	VectorXd trg;
	clock_t start, end;

	/* Simple calculation */
	u = iniu;
	start = clock();
	for(int i = 0; i < itrNum; ++i){
		v = p.array() / (K * u).array();
		u = q.array() / (K.transpose() * v).array();
	}
	trg = u.array().log();
	trg = trg.array() - trg(dim-1);
	end = clock();
	cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
	PRINT_MAT2(trg.block(0,0,10,1),"simple");

	/* LogSumExp */
	if(LOGSUMEXP){
		u = iniu;
		VectorXd log_u = u.array().log();
		VectorXd log_v;
		MatrixXd log_Ku, log_Kv;
		VectorXd dom_Ku;
		RowVectorXd dom_Kv;
		VectorXd sub_Ku;
		RowVectorXd sub_Kv;
		start = clock();
		for(int i = 0; i < itrNum; ++i){
			log_Ku = log_K.rowwise() + log_u.transpose();
			dom_Ku = log_Ku.rowwise().maxCoeff();
			sub_Ku = ((log_Ku.colwise() - dom_Ku).array().exp()).rowwise().sum();
			log_v = log_p.array() - sub_Ku.array().log() - dom_Ku.array();
			log_Kv = log_K.colwise() + log_v;
			dom_Kv = log_Kv.colwise().maxCoeff();
			sub_Kv = ((log_Kv.rowwise() - dom_Kv).array().exp()).colwise().sum();
			log_u = log_q.array() - sub_Kv.array().log().transpose() - dom_Kv.array().transpose();
		}
		trg = log_u.array() - log_u(dim-1);
		end = clock();
		cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
		PRINT_MAT2(trg.block(0,0,10,1),"logsumexp");
	}

	/* lVector and lMatrix */
	if(LNUM){
		u = iniu;
		lVector lp(p);
		lVector lq(q);
		lVector lu(u);
		lVector lv;
		lMatrix lK(K);
		lVector lKu, lKv;
		start = clock();
		for(int i = 0; i < itrNum; ++i){
			lKu = lK * lu;
//			lKu = lu;
			lv = lp.quotient(lKu);
			lKv = lv * lK;
//			lKv = lv;
			lu = lq.quotient(lKv);
		}
		trg = lu.asLogVector();
		trg = trg.array() - trg(dim-1);
		end = clock();
		cout << "time = " << (float)(end - start) / CLOCKS_PER_SEC << "sec.\n";
		PRINT_MAT2(trg.block(0,0,10,1),"lnum");
	}

	return 0;
}
