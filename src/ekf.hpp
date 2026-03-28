#pragma once
#include <Eigen/Dense>
#include <functional>

namespace kf{
    class ExtendedKalmanFilter {
    public:
        ExtendedKalmanFilter(
            std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
            std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> F,
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
            std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H,
            const Eigen::MatrixXd& P, // Estimate error covariance.
            const Eigen::MatrixXd& R, // Measurement noise covariance.
            const Eigen::MatrixXd& Q  // Process noise covariance.
        );
        ~ExtendedKalmanFilter();

        void init();
        void init(const Eigen::VectorXd& x);
        void predict(const Eigen::VectorXd& u);
        void update(const Eigen::VectorXd& y);
        Eigen::VectorXd get_state();
        Eigen::MatrixXd get_covariance();
    

    private:
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> F;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H;
    Eigen::MatrixXd P, R, Q;
    Eigen::MatrixXd P0; // initial P.
    Eigen::MatrixXd K;    // kalman gain.
    Eigen::MatrixXd I;    // unit matrix.

    Eigen::VectorXd x_hat; // estimated state.
    };

}