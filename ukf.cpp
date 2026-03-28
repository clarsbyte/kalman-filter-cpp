#include "ukf.hpp"

kf::UnscentedKalmanFilter::UnscentedKalmanFilter(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> _f, // state transition
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _h, // measurement model
    const Eigen::MatrixXd& _P,
    const Eigen::MatrixXd& _R, 
    const Eigen::MatrixXd& _Q, // Process noise covariance.
    double alpha,
    double beta,
    double kappa
)
: f(_f)
, h(_h)
, P(_P)
, P0(_P)
, R(_R)
, Q(_Q)
, x_hat(_P.rows())
, sigma_points(_P.rows(), alpha, beta, kappa)
{
}

kf::UnscentedKalmanFilter::~UnscentedKalmanFilter() {

}

void kf::UnscentedKalmanFilter::init() {

  P = P0;
  x_hat.setZero();
}

void kf::UnscentedKalmanFilter::init(const Eigen::VectorXd& x) {

  init();
  x_hat = x;
}

void kf::UnscentedKalmanFilter::predict(const Eigen::VectorXd& u) {
  int n = x_hat.size();

  // Generate sigma points
  sp::SigmaPointsResult sp_result = sigma_points.generate(x_hat, P);
  Eigen::MatrixXd& chi = sp_result.sigma_points;
  Eigen::VectorXd& Wm = sp_result.Wm;
  Eigen::VectorXd& Wc = sp_result.Wc;

  // Y = f(chi)
  Eigen::MatrixXd Y(n, 2 * n + 1);
  for (int i = 0; i < 2 * n + 1; i++) {
    Y.col(i) = f(chi.col(i), u);
  }

  x_hat = Y * Wm;

  P = Q;
  for (int i = 0; i < 2 * n + 1; i++) {
    Eigen::VectorXd diff = Y.col(i) - x_hat;
    P += Wc(i) * diff * diff.transpose();
  }
}

void kf::UnscentedKalmanFilter::update(const Eigen::VectorXd& z) {
  int n = x_hat.size();

  // Generate sigma points from predicted state
  sp::SigmaPointsResult sp_result = sigma_points.generate(x_hat, P);
  Eigen::MatrixXd& Y = sp_result.sigma_points;
  Eigen::VectorXd& Wm = sp_result.Wm;
  Eigen::VectorXd& Wc = sp_result.Wc;

  // Z = h(Y)
  int m = h(Y.col(0)).size();
  Eigen::MatrixXd Z(m, 2 * n + 1);
  for (int i = 0; i < 2 * n + 1; i++) {
    Z.col(i) = h(Y.col(i));
  }

  Eigen::VectorXd mu_z = Z * Wm;

  Eigen::MatrixXd Pz = R;
  for (int i = 0; i < 2 * n + 1; i++) {
    Eigen::VectorXd diff_z = Z.col(i) - mu_z;
    Pz += Wc(i) * diff_z * diff_z.transpose();
  }

  Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(n, m);
  for (int i = 0; i < 2 * n + 1; i++) {
    Eigen::VectorXd diff_x = Y.col(i) - x_hat;
    Eigen::VectorXd diff_z = Z.col(i) - mu_z;
    Pxz += Wc(i) * diff_x * diff_z.transpose();
  }

  K = Pxz * Pz.inverse();

  x_hat += K * (z - mu_z);

  P -= K * Pz * K.transpose();
}

Eigen::VectorXd kf::UnscentedKalmanFilter::get_state() {
  return x_hat;
}

Eigen::MatrixXd kf::UnscentedKalmanFilter::get_covariance() {
  return P;
}
