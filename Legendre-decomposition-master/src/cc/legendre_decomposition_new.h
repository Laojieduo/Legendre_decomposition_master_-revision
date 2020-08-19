// #define EIGEN_USE_MKL_ALL
#include <iostream>
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


// read a database file
void readTensorFromCSV(Tensor& X, Int num_mat, ifstream& ifs) {
  vector<vector<double>> data;
  string line;
  while (getline(ifs, line)) {
    stringstream lineStream(line);
    string cell;
    vector<double> tmp;
    while (getline(lineStream, cell, ',')) {
      tmp.push_back(stod(cell));
    }
    data.push_back(tmp);
  }
  
  //检测是否the depth size (the number of matrices)可以整除 
  if (data.size() % num_mat != 0) {
    cerr << endl << "The size specification of the input tensor (= " << num_mat << ") is invalid!" << endl;
    exit(1);
  }
  
  //构建原始矩阵 
  X = Tensor(num_mat, vector<vector<double>>(data.size() / num_mat, vector<double>(data.front().size(), 0)));

  for (Int k = 0; k < num_mat; ++k) {
    for (Int j = 0; j < X.front().size(); ++j) {
      for (Int i = 0; i < X.front().front().size(); ++i) {
	X[k][j][i] = data[j + k * X.front().size()][i];
      }
    }
  }
}

// preprocess X by a one step Sinkhorn balancing
//做论文2中的一个normalization过程，每个元素除以和 
double preprocess(Tensor& X) {
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();

  double X_sum = 0.0;
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      X_sum += accumulate(vec.begin(), vec.end(), 0.0);    //求和 
    }
  }
  for (auto&& mat : X) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	x /= X_sum;
      }
    }
  }
  return X_sum;
}


// make a node index，形成索引 
void makePosetIndex(Tensor& X, PosetIndex& idx_tp) {
  vector<Int> n{(Int)X.size(), (Int)X.front().size(), (Int)X.front().front().size()};

  // traverse a matrix in the topological order
  // 按拓扑顺序构建一个矩阵 
  for (Int ii = 0; ii < D; ++ii) {
    Int i1 = ii;
    Int i2 = ii + 1; if (i2 > 2) i2 -= D;
    Int i3 = ii + 2; if (i3 > 2) i3 -= D;
    for (Int i = 0; i < n[i2] + n[i3] - 1; ++i) {
      Int j = i + 1 - n[i2];
      if (j < 0) j = 0;
      for (; j <= min(i, n[i3] - 1); j++) {
	    idx_tp[i1].push_back(make_pair(i - j, j));
      }
    }
  }
}

// make a node matrix from eigen matrix
// 由特征矩阵构建节点矩阵，此处为初始化，即构建值等于原tensor的节点，并添加其他属性 
// Poset类型为 vector<vector<vector<node>>>，node为节点类型

void makePosetTensor(Tensor& X, Poset& S) {
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();

  // initialization
  // EPSILON为 1e-300，接近于0，默认为误差 
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	    S[i][j][k].p = X[i][j][k];
	    S[i][j][k].theta = 0; S[i][j][k].theta_sum = 0; S[i][j][k].theta_sum_prev = 0;
	    S[i][j][k].eta = 0;
		S[i][j][k].nonzero = S[i][j][k].p > EPSILON ? true : false;
      }
    }
  }
  S[0][0][0].nonzero = true;
}

// make a beta the submanifold，参见文中2.2 
// 将β构建成为子流形 
void prepareForBeta(Poset& S) {
  for (Int i = 0; i < S.size(); ++i) {
    for (Int j = 0; j < S.front().size(); ++j) {
      for (Int k = 0; k < S.front().front().size(); ++k) {
	S[i][j][k].p_tmp = S[i][j][k].p;
      }
    }
  }
  S[0][0][0].p_tmp = -1;    //这一步还没懂什么意思 ，初始化的意义？ 
}

void makeBetaCore(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta, Int core_size, bool random) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();

  // for sorting
  //定义容器大小 
  S_global.resize(n1);    
  for (auto&& s : S_global) {
    s.resize(n2);
    for (auto&& u : s) u.resize(n3);
  }
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	S_global[i][j][k].p = S[i][j][k].p;
      }
    }
  }

  vector<tuple<Int, Int, Int>> univ;
  vector<tuple<Int, Int, Int>> s;
  
  //产生大量伪随机数 
  mt19937 g(1);
  for (Int i = 0; i < n1; ++i) {
    univ.clear(); s.clear();
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
		if (S[i][j][k].p_tmp > 0) {
	  	  univ.push_back(make_tuple(i, j, k));
		}	
      }
    }
    
    if (random) shuffle(univ.begin(), univ.end(), g);	//产生一个随机排列 
    else sort(univ.begin(), univ.end(), greaterP());	//按照内容大小排序 
    
    if (core_size > univ.size()) core_size = univ.size();	//修改core_size使之不大于正数个数 
    for (Int c = 0; c < core_size; ++c) {
      tuple<Int, Int, Int>& b = univ[c];
      beta.push_back(make_pair(b, S[get<0>(b)][get<1>(b)][get<2>(b)].eta));	//存储所有的η 
      S[get<0>(b)][get<1>(b)][get<2>(b)].p_tmp = -1.0;	//没懂什么意思 
    }
  }
}
void makeBetaNorm(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta, Int core_size) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();

  vector<Int> idx1;
  if (core_size < n1) {
    Int diff1 = n1 / core_size;
    for (Int i = 0; i < core_size; ++i) idx1.push_back(i * diff1);
  } else {
    idx1.resize(n1); iota(idx1.begin(), idx1.end(), 0);
  }
  vector<Int> idx2;
  if (core_size < n2) {
    Int diff2 = n2 / core_size;
    for (Int i = 0; i < core_size; ++i) idx2.push_back(i * diff2);
  } else {
    idx2.resize(n2); iota(idx2.begin(), idx2.end(), 0);
  }
  vector<Int> idx3;
  if (core_size < n3) {
    Int diff3 = n3 / core_size;
    for (Int i = 0; i < core_size; ++i) idx3.push_back(i * diff3);
  } else {
    idx3.resize(n3); iota(idx3.begin(), idx3.end(), 0);
  }

  // 存入三个方向的eta 
  for (Int i = 1; i < n1; ++i) {
    if (S[i][0][0].p_tmp > 0) {
      beta.push_back(make_pair(make_tuple(i, 0, 0), S[i][0][0].eta));
      S[i][0][0].p_tmp = -1;
    }
  }
  for (Int j = 1; j < n2; ++j) {
    beta.push_back(make_pair(make_tuple(0, j, 0), S[0][j][0].eta));
    S[0][j][0].p_tmp = -1;
  }
  for (Int k = 1; k < n3; ++k) {
    beta.push_back(make_pair(make_tuple(0, 0, k), S[0][0][k].eta));
    S[0][0][k].p_tmp = -1;
  }

  for (auto&& i : idx1) {
    for (auto&& j : idx2) {
      if (S[i][j][0].p_tmp > 0) {
	beta.push_back(make_pair(make_tuple(i, j, 0), S[i][j][0].eta));
	S[i][j][0].p_tmp = -1;
      }
    }
    for (auto&& k : idx3) {
      if (S[i][0][k].p_tmp > 0) {
	beta.push_back(make_pair(make_tuple(i, 0, k), S[i][0][k].eta));
	S[i][0][k].p_tmp = -1;
      }
    }
  }
}


// compute eta for all entries
void computeEta(Poset& S) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();

  vector<Int> idx_row(n1 - 1);
  vector<Int> idx_col(n2 - 1);
  vector<Int> idx_dep(n3 - 1);
  iota(idx_row.begin(), idx_row.end(), 0);
  iota(idx_col.begin(), idx_col.end(), 0);
  iota(idx_dep.begin(), idx_dep.end(), 0);

  S[idx_row.size()][idx_col.size()][idx_dep.size()].eta = S[idx_row.size()][idx_col.size()][idx_dep.size()].p;
  for (auto&& i : reverse(idx_row))
    S[i][idx_col.size()][idx_dep.size()].eta = S[i][idx_col.size()][idx_dep.size()].p + S[i + 1][idx_col.size()][idx_dep.size()].eta;
  for (auto&& j : reverse(idx_col))
    S[idx_row.size()][j][idx_dep.size()].eta = S[idx_row.size()][j][idx_dep.size()].p + S[idx_row.size()][j + 1][idx_dep.size()].eta;
  for (auto&& k : reverse(idx_dep))
    S[idx_row.size()][idx_col.size()][k].eta = S[idx_row.size()][idx_col.size()][k].p + S[idx_row.size()][idx_col.size()][k + 1].eta;

  for (auto&& i : reverse(idx_row)) {
    for (auto&& j : reverse(idx_col)) {
      S[i][j][idx_dep.size()].eta = S[i][j][idx_dep.size()].p + S[i + 1][j][idx_dep.size()].eta + S[i][j + 1][idx_dep.size()].eta - S[i + 1][j + 1][idx_dep.size()].eta;
    }
  }
  for (auto&& j : reverse(idx_col)) {
    for (auto&& k : reverse(idx_dep)) {
      S[idx_row.size()][j][k].eta = S[idx_row.size()][j][k].p + S[idx_row.size()][j + 1][k].eta + S[idx_row.size()][j][k + 1].eta - S[idx_row.size()][j + 1][k + 1].eta;
    }
  }
  for (auto&& k : reverse(idx_dep)) {
    for (auto&& i : reverse(idx_row)) {
      S[i][idx_col.size()][k].eta = S[i][idx_col.size()][k].p + S[i + 1][idx_col.size()][k].eta + S[i][idx_col.size()][k + 1].eta - S[i + 1][idx_col.size()][k + 1].eta;
    }
  }

  for (auto&& i : reverse(idx_row)) {
    for (auto&& j : reverse(idx_col)) {
      for (auto&& k : reverse(idx_dep)) {
	S[i][j][k].eta = S[i][j][k].p + S[i + 1][j][k].eta + S[i][j + 1][k].eta + S[i][j][k + 1].eta - S[i + 1][j + 1][k].eta - S[i + 1][j][k + 1].eta - S[i][j + 1][k + 1].eta + S[i + 1][j + 1][k + 1].eta;
      }
    }
  }
}

//e-projection，进行e映射 
void computeP(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
		if (S[i][j][k].p > EPSILON) {
	      double theta_sum = 0.0;
	      double theta_sum_prev = 0.0;
	      for (auto&& b : beta) {
	        if (get<0>(b.first) <= i && get<1>(b.first) <= j && get<2>(b.first) <= k) {
	          theta_sum += S[get<0>(b.first)][get<1>(b.first)][get<2>(b.first)].theta;
	          theta_sum_prev += S[get<0>(b.first)][get<1>(b.first)][get<2>(b.first)].theta_prev;
	        }
	      }
	      S[i][j][k].theta_sum = theta_sum;
	      S[i][j][k].theta_sum_prev = theta_sum_prev;
	      S[i][j][k].p = exp(theta_sum);
	    }
      }
    }
  }
}

void renormalize(Poset& S) {
  // total sum
  double p_sum = 0.0;
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x.p > EPSILON) p_sum += x.p;
      }
    }
  }
  // store the previous theta
  S[0][0][0].theta_prev = S[0][0][0].theta;
  // update theta(\bot)
  S[0][0][0].theta = S[0][0][0].theta_prev - log(p_sum);
  // update p
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	if (x.p > EPSILON) x.p *= exp(S[0][0][0].theta - S[0][0][0].theta_prev);
      }
    }
  }
}

//对node节点矩阵进行初始化 
void initialize(Poset& S) {
  Int n1 = S.size();
  Int n2 = S.front().size();
  Int n3 = S.front().front().size();
  double size = 0.0;
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	if (S[i][j][k].nonzero) size += 1.0; //统计非零个数 
      }
    }
  }
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
		S[i][j][k].p = S[i][j][k].nonzero ? 1.0 / size : 0.0;	//存疑 
		S[i][j][k].theta = 0.0;
		S[i][j][k].theta_prev = 0.0;
      }
    }
  }
  //参考2.2 式子2 
  S[0][0][0].theta = log(S[0][0][0].p);
  S[0][0][0].theta_prev = log(S[0][0][0].p);
  computeEta(S);
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
  // cout<<"fai_theta:"<<fai_theta<<" "<<"max="<<max<<"   min="<<min<<"   ";
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
    theta_vec[i] = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta;    // repeat中第一步 ？ 
    eta_vec[i] = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta - beta[i].second;    //repeat中第二步 
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
  cout<<"I am the true DKL:"<<computeDKL(S, beta)<<endl;
}


// ====================================== //
// ========== Natural gradient ========== //
// ====================================== //
void eProject(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  Int S_size = 0;
  //正数个数
  for (auto&& mat : S) {
    for (auto&& vec : mat) {
      for (auto&& x : vec) {
	    if (x.p > EPSILON) S_size++;
      }
    }
  }
  //初始化theta和eta 
  VectorXd theta_vec = VectorXd::Zero(beta.size());
  VectorXd eta_vec = VectorXd::Zero(beta.size());
  
  for (Int i = 0; i < beta.size(); i++) {
    theta_vec[i] = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta;    
    eta_vec[i] = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta - beta[i].second;   
  }
  
  
  MatrixXd J(beta.size(), beta.size()); // Jacobian matrix
  for (Int i1 = 0; i1 < beta.size(); i1++) {
    for (Int i2 = 0; i2 < beta.size(); i2++) {
      Int i1_i = get<0>(beta[i1].first);
      Int i1_j = get<1>(beta[i1].first);
      Int i1_k = get<2>(beta[i1].first);
      Int i2_i = get<0>(beta[i2].first);
      Int i2_j = get<1>(beta[i2].first);
      Int i2_k = get<2>(beta[i2].first);
      J(i1, i2) = S[max(i1_i, i2_i)][max(i1_j, i2_j)][max(i1_k, i2_k)].eta;
      J(i1, i2) -= S[i1_i][i1_j][i1_k].eta * S[i2_i][i2_j][i2_k].eta;
    }
  }
  
  //QR分解后 计算参数G并处理theta 
  theta_vec += ((-1 * J).colPivHouseholderQr().solve(eta_vec));
  // theta_vec += (-1 * J).fullPivHouseholderQr().solve(eta_vec);
  // theta_vec += (-1 * J).fullPivLu().solve(eta_vec);

  // store theta
  for (Int i = 0; i < beta.size(); i++) {
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta_prev = S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta;
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta = theta_vec[i];
  }
  // update p
  computeP(S, beta);
  renormalize(S);
  computeEta(S);
}


// ====================================== //
// ========== Gradient descent ========== //
// ====================================== //
void grad(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  double eps = 0.1;    //步长 
  for (Int i = 0; i < beta.size(); i++) {
    S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].theta += eps * (beta[i].second - S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta);
    // update p
    computeP(S, beta);
    renormalize(S);
    computeEta(S);
  }
}

// ***********************   compute the residual，残差计算    ************************************** 
// 计算beta.double和记录位置的对应值的η残差 
double computeResidual(Poset& S, vector<pair<tuple<Int, Int, Int>, double>>& beta) {
  double res = 0.0;
  for (Int i = 0; i < beta.size(); i++) {
    res += pow(S[get<0>(beta[i].first)][get<1>(beta[i].first)][get<2>(beta[i].first)].eta - beta[i].second, 2.0);
  }
  return sqrt(res);    
}

//均方根误差计算 
double computeRMSE(Tensor& X, Tensor& Y) {
  double rmse = 0.0;
  for (Int i = 0; i < X.size(); ++i) {
    for (Int j = 0; j < X.front().size(); ++j) {
      for (Int k = 0; k < X.front().front().size(); ++k) {
	rmse += pow(X[i][j][k] - Y[i][j][k], 2.0);
      }
    }
  }
  rmse /= (double)X.size() * (double)X.front().size() * (double)X.front().front().size();
  rmse = sqrt(rmse);
  return rmse;
}

// the main function for Legendre decomposition by natural gradient
// 主程序部分 
double LegendreDecomposition(Tensor& X, Int core_size, double error_tol, double rep_max, bool verbose, Int type, Int const_type, Int *num_param) {
  Int n1 = X.size();
  Int n2 = X.front().size();
  Int n3 = X.front().front().size();
  clock_t ts, te;
  
  double X_sum = preprocess(X);    // preprocess，初始化，除以和 
  Poset S = Poset(n1, vector<vector<node>>(n2, vector<node>(n3))); 
  makePosetTensor(X, S);    // make a node matrix，初始化节点矩阵 
  computeEta(S);    //计算节点矩阵的η值 

  vector<pair<tuple<Int, Int, Int>, double>> beta;
  prepareForBeta(S);
  
  //分解基的选择，type of a decomposition basis (currently can be 1 [complex] or 2 [simple])
  //下面函数内的true和false代表是否产生一个随机排列，如果否则按照内容大小排序，其中剔除0元素所占位置 
  if (const_type == 1) {
    makeBetaNorm(S, beta, core_size);
    makeBetaCore(S, beta, core_size, true);
  } else if (const_type == 2) {
    makeBetaCore(S, beta, core_size, false);
  } else {
    makeBetaCore(S, beta, core_size, false);
  }
  
  initialize(S);
  
  cout << "  Number of parameters: " << beta.size() << endl << flush;
  *num_param = beta.size();

  // run Legendre deomposition
  if (verbose) cout << "----- Start Legendre decomposition -----" << endl << flush;    //详细模式，额外输出一些状态 
  double res = 0.0;
  double step = 1.0;
  Int exponent = 0;
  auto t_start = system_clock::now();

  MatrixXd DFP(beta.size(), beta.size()); // DFP matrix
  for (Int i = 0; i < beta.size(); i++){
    DFP(i, i) = 1.0;
  }

  while (step <= rep_max) { //小于最大迭代次数 

    if (type == 1) {    //type=1代表natural，type=2代表梯度下降 
      eProject(S, beta);     //进行e-映射 
    } else if (type == 2) {
      grad(S, beta); // perform gradient descent
    } else if (type == 3){
      quasi(S, beta, DFP);
    } else{
      eProject(S, beta); // perform e-projection，如果type传参错误（非1非2） 
    }
    cout<<beta.size()<<endl;
    double res_prev = res;
    res = computeResidual(S, beta);
    //如果开始发散，直接结束 
    if (res_prev >= EPSILON && res > res_prev) { 
      cout << "  Terminate with current residual = " << res << endl;
      return step;
    }

    // output the residual
    if (verbose) {
      cout << "Step\t" << step << "\t" << "Residual\t" << res << endl << flush;
    } else {
      if (res < pow(10, -1.0 * (double)exponent)) {
	cout << "  Step "; if (step < 10.0) cout << " ";
	cout << step << ", Residual: " << res << endl;
	exponent++;
      }
    }
    if (res < error_tol) break;
    step += 1.0;
  }
  if (verbose) cout << "----- End   Legendre decomposition -----" << endl;

  // put results to X
  for (Int i = 0; i < n1; ++i) {
    for (Int j = 0; j < n2; ++j) {
      for (Int k = 0; k < n3; ++k) {
	X[i][j][k] = S[i][j][k].p * X_sum;
      }
    }
  }

  return step;
}
