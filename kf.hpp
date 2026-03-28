#pragma once

#include <Eigen/Dense>

namespace kf {

class KalmanFilter {
public:
    KalmanFilter(
        const Eigen::MatrixXd& F,
        const Eigen::MatrixXd& B,
        const Eigen::MatrixXd& H,
        const Eigen::MatrixXd& P,
        const Eigen::MatrixXd& R,
        const Eigen::MatrixXd& Q
    )
    : F(F), B(B), H(H), P(P), P0(P), R(R), Q(Q)
    , I(Eigen::MatrixXd::Identity(F.rows(), F.rows()))
    , x_hat(Eigen::VectorXd::Zero(F.rows()))
    {}

    ~KalmanFilter() {}

    void init() {
        P = P0;
        x_hat.setZero();
    }

    void init(const Eigen::VectorXd& x) {
        init();
        x_hat = x;
    }

    void predict(const Eigen::VectorXd& u) {
        x_hat = F * x_hat + B * u;
        P     = F * P * F.transpose() + Q;
    }

    void update(const Eigen::VectorXd& y) {
        K     = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        x_hat += K * (y - H * x_hat);
        P     = (I - K * H) * P;
    }

    Eigen::VectorXd get_state()      { return x_hat; }
    Eigen::MatrixXd get_covariance() { return P; }

private:
    Eigen::MatrixXd F, B, H, P, R, Q;
    Eigen::MatrixXd P0, K, I;
    Eigen::VectorXd x_hat;
};

}  // namespace kf
