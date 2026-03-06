#pragma once

#include <Eigen/Dense>
#include <vector>

namespace rts {

class Smoother {

public:
  Smoother(
    const Eigen::MatrixXd& F, // State transition matrix.
    const Eigen::MatrixXd& Q  // Process noise covariance.
  );

  struct Result {
    std::vector<Eigen::VectorXd> x;
    std::vector<Eigen::MatrixXd> P;
    std::vector<Eigen::MatrixXd> K;
    std::vector<Eigen::MatrixXd> Pp;
  };

  // Run RTS smoother on batch Kalman filter output
  // Xs: forward-pass state estimates (length n)
  // Ps: forward-pass covariance estimates (length n)
  Result smooth(
    const std::vector<Eigen::VectorXd>& Xs,
    const std::vector<Eigen::MatrixXd>& Ps
  );

private:
  Eigen::MatrixXd F, Q;
};

}  // namespace rts
