#pragma once

#include <Eigen/Dense>

namespace kf {

class KalmanFilter {

public:
  KalmanFilter(
    const Eigen::MatrixXd& F, // System dynamics.
    const Eigen::MatrixXd& B, // Control.
    const Eigen::MatrixXd& H, // Output.
    const Eigen::MatrixXd& P, // Estimate error covariance.
    const Eigen::MatrixXd& R, // Measurement noise covariance.
    const Eigen::MatrixXd& Q  // Process noise covariance.
    );
  ~KalmanFilter();

  void init();
  void init(const Eigen::VectorXd& x);
  void predict(const Eigen::VectorXd& u);
  void update(const Eigen::VectorXd& y);
  Eigen::VectorXd get_state();
  Eigen::MatrixXd get_covariance();

private:
  Eigen::MatrixXd F, B, H, P, R, Q;
  Eigen::MatrixXd P0;   // initial P.
  Eigen::MatrixXd K;    // kalman gain.
  Eigen::MatrixXd I;    // unit matrix.

  Eigen::VectorXd x_hat; // estimated state.
};

}  // namespace kf