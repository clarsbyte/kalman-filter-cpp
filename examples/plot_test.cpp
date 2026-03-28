#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "../kf.hpp"
#include "../ekf.hpp"
#include "../ukf.hpp"
#include "../plot.hpp"

const double dt = 0.05;
const double gL = 9.81;

Eigen::VectorXd f_pendulum(const Eigen::VectorXd& x, const Eigen::VectorXd& /*u*/) {
    Eigen::VectorXd x_new(2);
    x_new(0) = x(0) + dt * x(1);
    x_new(1) = x(1) - dt * gL * std::sin(x(0));
    return x_new;
}

Eigen::MatrixXd F_jacobian(const Eigen::VectorXd& x, const Eigen::VectorXd& /*u*/) {
    Eigen::MatrixXd F(2, 2);
    F << 1, dt,
         -dt * gL * std::cos(x(0)), 1;
    return F;
}

Eigen::VectorXd h_nonlinear(const Eigen::VectorXd& x) {
    Eigen::VectorXd z(1);
    z(0) = std::sin(x(0));
    return z;
}

Eigen::MatrixXd H_jacobian(const Eigen::VectorXd& x) {
    Eigen::MatrixXd H(1, 2);
    H << std::cos(x(0)), 0;
    return H;
}

// Linear approximation 
Eigen::MatrixXd make_F_linear() {
    Eigen::MatrixXd F(2, 2);
    F << 1, dt,
         -dt * gL, 1; 
    return F;
}

Eigen::MatrixXd make_H_linear() {
    Eigen::MatrixXd H(1, 2);
    H << 1, 0;  
    return H;
}

double randn() {
    double u1 = ((double)rand() / RAND_MAX);
    double u2 = ((double)rand() / RAND_MAX);
    if (u1 < 1e-10) u1 = 1e-10;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

int main() {
    srand(42);

    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(2, 2) * 0.1;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * 0.05;

    // KF
    kf::KalmanFilter kf_filter(make_F_linear(), Eigen::MatrixXd::Zero(2, 1), make_H_linear(), P, R, Q);

    // EKF 
    kf::ExtendedKalmanFilter ekf(f_pendulum, F_jacobian, h_nonlinear, H_jacobian, P, R, Q);

    // UKF 
    kf::UnscentedKalmanFilter ukf(f_pendulum, h_nonlinear, P, R, Q);

    Eigen::VectorXd x_true(2);
    x_true << 1.2, 0.0;

    Eigen::VectorXd x_init(2);
    x_init << 0.8, 0.0;  
    kf_filter.init(x_init);
    ekf.init(x_init);
    ukf.init(x_init);

    kf::StateHistory kf_hist("KF");
    kf::StateHistory ekf_hist("EKF");
    kf::StateHistory ukf_hist("UKF");
    kf::StateHistory truth_hist("Truth");

    Eigen::VectorXd u(1);
    u << 0.0;

    int steps = 200;
    for (int i = 0; i < steps; i++) {
        double t = i * dt;

        x_true = f_pendulum(x_true, u);

        // Noisy measurement of sin(theta)
        Eigen::VectorXd z(1);
        z(0) = std::sin(x_true(0)) + std::sqrt(0.05) * randn();

        kf_filter.predict(u);
        kf_filter.update(z);

        ekf.predict(u);
        ekf.update(z);

        ukf.predict(u);
        ukf.update(z);

        kf_hist.record(t, kf_filter.get_state(), kf_filter.get_covariance());
        ekf_hist.record(t, ekf.get_state(), ekf.get_covariance());
        ukf_hist.record(t, ukf.get_state(), ukf.get_covariance());
        truth_hist.record(t, x_true);
    }

    kf::plot_states(
        {truth_hist, kf_hist, ekf_hist, ukf_hist},
        {"Theta (rad)", "Theta_dot (rad/s)"},
        "Pendulum - Filter Comparison",
        false,
        "kf_comparison"
    );

    return 0;
}
