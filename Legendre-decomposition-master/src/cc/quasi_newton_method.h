// #define EIGEN_USE_MKL_ALL
#include <iostream>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <numeric>
#include <random>
#include <functional>
#include <utility>
#include <iomanip>
#include <tuple>
#include <time.h>
#include <chrono>
#include "Eigen/Dense"
#include <cstdlib>
#include <string.h>

#define Int int32_t
#define Tensor vector<vector<vector<double>>>
#define Poset vector<vector<vector<node>>>
#define PosetIndex vector<vector<pair<Int, Int>>>
#define D 3

using namespace std;
using namespace Eigen;
using namespace std::chrono;

double EPSILON = 1e-300;

// node structure
class node {
public:
  bool nonzero;
  Int id, id_org;
  double p, p_tmp;
  double theta, theta_prev, theta_sum, theta_sum_prev;
  double eta, eta_prev;
};
Poset S_global;

// output a tensor，输出一个tensor 
ostream &operator<<(ostream& out, const Tensor& X) {
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (Int i = 0; i < (Int)vec.size() - 1; ++i) {
		out << vec[i] << ", ";
      }
      out << vec.back() << endl;
    }
    out << endl;
  }
  return out;
}

// for "reverse" in range-based loop
template<class Cont> class const_reverse_wrapper {
  const Cont& container;
public:
  const_reverse_wrapper(const Cont& cont) : container(cont){ }
  decltype(container.rbegin()) begin() const { return container.rbegin(); }
  decltype(container.rend()) end() const { return container.rend(); }
};
template<class Cont> class reverse_wrapper {
  Cont& container;
public:
  reverse_wrapper(Cont& cont) : container(cont){ }
  decltype(container.rbegin()) begin() { return container.rbegin(); }
  decltype(container.rend()) end() { return container.rend(); }
};
template<class Cont> const_reverse_wrapper<Cont> reverse(const Cont& cont) {
  return const_reverse_wrapper<Cont>(cont);
}
template<class Cont> reverse_wrapper<Cont> reverse(Cont& cont) {
  return reverse_wrapper<Cont>(cont);
}


// for sort
struct greaterP { bool operator()(const tuple<Int, Int, Int>& b1, const tuple<Int, Int, Int>& b2) const { return S_global[get<0>(b1)][get<1>(b1)][get<2>(b1)].p > S_global[get<0>(b2)][get<1>(b2)][get<2>(b2)].p; } };

double computeDKL(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta){
  double DKL = 0.0;
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  double fai_theta = 0;
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	fai_theta += exp(S[i][j][k].theta_sum);
      }
    }
  }
  fai_theta = log(fai_theta);
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	DKL += S[i][j][k].p * (-S[i][j][k].theta_sum + fai_theta);
      }
    }
  }
  return DKL;
}


// ====================================== //
// ========= Quasi_Newton Method ======== //
// ====================================== //
double computeDKL(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta){
  double DKL = 0.0;
  double fai_theta = 0;
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  double max = -10000;
  double min = 10000;
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	fai_theta += exp(S[i][j][k].theta_sum);
      }
    }
  }
  fai_theta = log(fai_theta);
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
        if (S[i][j][k].p * (-S[i][j][k].theta_sum + fai_theta) > max) max = S[i][j][k].p * (-S[i][j][k].theta_sum + fai_theta);
        if (S[i][j][k].p * (-S[i][j][k].theta_sum + fai_theta) < min) min = S[i][j][k].p * (-S[i][j][k].theta_sum + fai_theta);
	DKL += S[i][j][k].p * (-S[i][j][k].theta_sum + fai_theta);
      }
    }
  }
  return DKL;
}

double quasi_find_lambda(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta, VectorXd& dd, VectorXd& eta_vec){
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  double m = -1.5;
  double min_lambda = 0.0;
  double min_DKL = 10000;
  
  Poset S_tmp = Poset(n1, vector<vector<node>>(n2, vector<node>(n3))); 
  double DKL_S = computeDKL(S, beta);
  cout<<"DKL:"<<DKL_S<<endl;
  while (m <= 2){
    for (Int i = 0; i < n1; ++i) {
      for (Int j = 0; j < n2; ++j) {
        for (Int k = 0; k < n3; ++k) {
	  S_tmp[i][j][k].theta_sum = S[i][j][k].theta_sum;
	  S_tmp[i][j][k].p = S[i][j][k].p;
          S_tmp[i][j][k].theta = S[i][j][k].theta;
        }
      }
    }
    for (Int i =0; i < beta.size(); i++){
      Int I1 = get<0>(beta[i].first);
      Int I2 = get<1>(beta[i].first);
      Int I3 = get<2>(beta[i].first);
      S_tmp[I1][I2][I3].theta += m * dd[i];
    }
    for (Int i = 0; i < n1; ++i) {
      for (Int j = 0; j < n2; ++j) {
        for (Int k = 0; k < n3; ++k) {
	  if (S_tmp[i][j][k].p > EPSILON) {
	    double theta_sum = 0.0;
	      for (auto&& b : beta) {
	        if (get<0>(b.first) <= i && get<1>(b.first) <= j && get<2>(b.first) <= k) {
	          theta_sum += S_tmp[get<0>(b.first)][get<1>(b.first)][get<2>(b.first)].theta;
	        }
	      }
	    S_tmp[i][j][k].theta_sum = theta_sum;
	    S_tmp[i][j][k].p = exp(theta_sum);
	  }
        }
      }
    }
    if (computeDKL(S_tmp, beta) < min_DKL){
      min_DKL = computeDKL(S_tmp, beta);
      min_lambda = m;
    }
    double timess = (eta_vec.transpose()) * dd;
    m = m + 0.01;
  }
  return min_lambda;
}


void quasi(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta, MatrixXd& DFP) {
  MatrixXd temp(beta.size(), beta.size());
  //初始化theta和eta 

  VectorXd theta_vec = VectorXd::Zero(beta.size());
  VectorXd eta_vec = VectorXd::Zero(beta.size());
  VectorXd eta_vec_org = VectorXd::Zero(beta.size());
  VectorXd dd = VectorXd::Zero(beta.size());

  for (Int i = 0; i < beta.size(); i++) {
    theta_vec[i] = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta; 
    eta_vec[i] = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta - beta[i].second;
    eta_vec_org[i] = beta[i].second;
  }

  dd = -1 * (DFP * eta_vec_org);
  double lambda_min = quasi_find_lambda(S, beta, dd, eta_vec);

  theta_vec = theta_vec + dd * lambda_min;
  dd = dd * lambda_min;
  temp = DFP + dd * (dd.transpose()) / ((dd.transpose()) * eta_vec) - (DFP * eta_vec * (eta_vec.transpose()) * DFP) / ((eta_vec.transpose()) * DFP * eta_vec);
  DFP = temp;
  for (Int i = 0; i < beta.size(); i++) {
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta_prev = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta;
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta = theta_vec[i];
  }
  // update p
  computeP(S, beta);
  renormalize(S);
  computeEta(S);
}

