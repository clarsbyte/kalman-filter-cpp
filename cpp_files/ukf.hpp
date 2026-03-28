#pragma once
#include <Eigen/Dense>
#include <functional>
#include "sigma_points.hpp"

namespace kf{
    class UnscentedKalmanFilter {
    public:
        UnscentedKalmanFilter(
            std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f, // state transition
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h, // measurement model
            const Eigen::MatrixXd& P, // Estimate error covariance.
            const Eigen::MatrixXd& R, // Measurement noise covariance.
            const Eigen::MatrixXd& Q, // Process noise covariance.
            double alpha = 1e-3,
            double beta = 2.0,
            double kappa = 0.0
        );
        ~UnscentedKalmanFilter();

        void init();
        void init(const Eigen::VectorXd& x);
        void predict(const Eigen::VectorXd& u);
        void update(const Eigen::VectorXd& y);
        Eigen::VectorXd get_state();
        Eigen::MatrixXd get_covariance();

    private:
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h;
    Eigen::MatrixXd P, R, Q;
    Eigen::MatrixXd P0; // initial P.
    Eigen::MatrixXd K;  // kalman gain.

    Eigen::VectorXd x_hat; // estimated state.

    sp::SigmaPoints sigma_points;
    };
}
