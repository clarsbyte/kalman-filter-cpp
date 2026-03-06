#include "ukf.hpp"

ukf::UnscentedKalmanFilter::UnscentedKalmanFilter(
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

ukf::UnscentedKalmanFilter::~UnscentedKalmanFilter() {

}

Eigen::VectorXd kf::UnscentedKalmanFilter::sigma_points(){
    
}

void ukf::UnscentedKalmanFilter::init() {

  P = P0;
  x_hat.setZero();
}

void ukf::UnscentedKalmanFilter::init(const Eigen::VectorXd& x) {

  init();
  x_hat = x;
}

void ukf::UnscentedKalmanFilter::predict(const Eigen::VectorXd& u) {

  x_hat = F * x_hat + B * u;
  P = F*P*F.transpose() + Q;
}

void ukf::UnscentedKalmanFilter::update(const Eigen::VectorXd& y) {

	K = P*H.transpose()*(H*P*H.transpose() + R).inverse();
	x_hat += K * (y - H*x_hat);
	P = (I - K*H)*P;
}

Eigen::VectorXd ukf::UnscentedKalmanFilter::get_state() {
  return x_hat;
}

Eigen::MatrixXd ukf::UnscentedKalmanFilter::get_covariance() {
  return P;
}
