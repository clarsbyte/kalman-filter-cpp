#pragma once

#include <Eigen/Dense>
#include <functional>

namespace kf {

class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter(
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> F,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H,
        const Eigen::MatrixXd& P,
        const Eigen::MatrixXd& R,
        const Eigen::MatrixXd& Q
    )
    : f(f), F(F), h(h), H(H), P(P), P0(P), R(R), Q(Q)
    , I(Eigen::MatrixXd::Identity(P.rows(), P.rows()))
    , x_hat(Eigen::VectorXd::Zero(P.rows()))
    {}

    ~ExtendedKalmanFilter() {}

    void init() {
        P = P0;
        x_hat.setZero();
    }

    void init(const Eigen::VectorXd& x) {
        init();
        x_hat = x;
    }

    void predict(const Eigen::VectorXd& u) {
        Eigen::VectorXd x = x_hat;
        x_hat = f(x_hat, u);
        P     = F(x, u) * P * F(x, u).transpose() + Q;
    }

    void update(const Eigen::VectorXd& y) {
        Eigen::VectorXd x = x_hat;
        K     = P * H(x).transpose() * (H(x) * P * H(x).transpose() + R).inverse();
        x_hat += K * (y - h(x));
        P     = (I - K * H(x)) * P;
    }

    Eigen::VectorXd get_state()      { return x_hat; }
    Eigen::MatrixXd get_covariance() { return P; }

private:
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> F;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> H;
    Eigen::MatrixXd P, R, Q, P0, K, I;
    Eigen::VectorXd x_hat;
};

}  // namespace kf
