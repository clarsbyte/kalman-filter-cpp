#include "ekf.hpp"

kf::UnscentedKalmanFilter::UnscentedKalmanFilter(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> _f, // function for state transition
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> _F, // jacobian of f
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> _h, // measurement model
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> _H, // jacobian of H
    const Eigen::MatrixXd& _P, // Estimate error covariance.
    const Eigen::MatrixXd& _R, // Measurement noise covariance.
    const Eigen::MatrixXd& _Q  // Process noise covariance.
)
: f(_f)
, F(_F)
, h(_h)
, H(_H)
, P(_P)
, P0(_P)
, R(_R)
, Q(_Q)
, I(_P.rows(), _P.rows())
, x_hat(_P.rows())
{
  I.setIdentity();
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
  Eigen::VectorXd x = x_hat;
  x_hat = f(x_hat, u);
  P = F(x, u) * P * F(x, u).transpose() + Q;
}

void kf::UnscentedKalmanFilter::update(const Eigen::VectorXd& y) {
	Eigen::VectorXd x = x_hat;
    K = P*H(x).transpose()*(H(x)*P*H(x).transpose() + R).inverse();
    x_hat += K * (y - h(x));
	P = (I - K*H(x))*P;
}

Eigen::VectorXd kf::UnscentedKalmanFilter::get_state() {
  return x_hat;
}

Eigen::MatrixXd kf::UnscentedKalmanFilter::get_covariance() {
  return P;
}

