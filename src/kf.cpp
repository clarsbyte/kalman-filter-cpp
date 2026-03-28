#include "kf.hpp"

kf::KalmanFilter::KalmanFilter(
    const Eigen::MatrixXd& _F, // System dynamics.
    const Eigen::MatrixXd& _B, // Control.
    const Eigen::MatrixXd& _H, // Output.
    const Eigen::MatrixXd& _P, // Estimate error covariance.
    const Eigen::MatrixXd& _R, // Measurement noise covariance.
    const Eigen::MatrixXd& _Q  // Process noise covariance.
)
: F(_F)
, B(_B)
, H(_H)
, P(_P)
, P0(_P)
, R(_R)
, Q(_Q)
, I(_F.rows(), _F.rows())
, x_hat(_F.rows())
{
  I.setIdentity();
}

kf::KalmanFilter::~KalmanFilter() {

}

void kf::KalmanFilter::init() {

  P = P0;
  x_hat.setZero();
}

void kf::KalmanFilter::init(const Eigen::VectorXd& x) {

  init();
  x_hat = x;
}

void kf::KalmanFilter::predict(const Eigen::VectorXd& u) {

  x_hat = F * x_hat + B * u;
  P = F*P*F.transpose() + Q;
}

void kf::KalmanFilter::update(const Eigen::VectorXd& y) {

	K = P*H.transpose()*(H*P*H.transpose() + R).inverse();
	x_hat += K * (y - H*x_hat);
	P = (I - K*H)*P;
}

Eigen::VectorXd kf::KalmanFilter::get_state() {
  return x_hat;
}

Eigen::MatrixXd kf::KalmanFilter::get_covariance() {
  return P;
}