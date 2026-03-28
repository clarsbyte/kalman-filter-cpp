#include <iostream>
#include <cmath>
#include "../filtercpp.h"

// State: x = [theta, theta_dot]
// pendulum with dt=0.1s, g/L=9.81
// theta_ddot = -(g/L)*sin(theta)
const double dt = 0.1;
const double gL = 9.81;

// motion: Euler integration of pendulum equations
Eigen::VectorXd f(const Eigen::VectorXd& x, const Eigen::VectorXd& /*u*/) {
    Eigen::VectorXd x_new(2);
    x_new(0) = x(0) + dt * x(1);
    x_new(1) = x(1) - dt * gL * std::sin(x(0));
    return x_new;
}

// Jacobian of f w.r.t. x
Eigen::MatrixXd F(const Eigen::VectorXd& x, const Eigen::VectorXd& /*u*/) {
    Eigen::MatrixXd Fmat(2, 2);
    Fmat(0, 0) = 1.0;
    Fmat(0, 1) = dt;
    Fmat(1, 0) = -dt * gL * std::cos(x(0));
    Fmat(1, 1) = 1.0;
    return Fmat;
}

// Measurement: we only observe sin(theta)
Eigen::VectorXd h(const Eigen::VectorXd& x) {
    Eigen::VectorXd z(1);
    z(0) = std::sin(x(0));
    return z;
}

// Jacobian of h w.r.t. x
Eigen::MatrixXd H(const Eigen::VectorXd& x) {
    Eigen::MatrixXd Hmat(1, 2);
    Hmat(0, 0) = std::cos(x(0));
    Hmat(0, 1) = 0.0;
    return Hmat;
}

int main() {
    // Noise covariances
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(2, 2) * 0.1;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * 0.05;

    filtercpp::ExtendedKalmanFilter ekf(f, F, h, H, P, R, Q);

    Eigen::VectorXd x_true(2);
    x_true << 0.5, 0.0;

    Eigen::VectorXd x_init(2);
    x_init << 0.3, 0.1;
    ekf.init(x_init);

    Eigen::VectorXd u(1);
    u << 0.0; 

    std::cout << "step | true_theta | est_theta\n";
    for (int i = 0; i < 20; ++i) {
        x_true = f(x_true, u);

        ekf.predict(u);

        // Simulate a noisy measurement of sin(theta)
        Eigen::VectorXd y(1);
        y(0) = std::sin(x_true(0)) + 0.05 * ((double)rand() / RAND_MAX - 0.5);
        ekf.update(y);

        Eigen::VectorXd x_hat = ekf.get_state();
        std::cout << i + 1 << "    | " << x_true(0) << "       | " << x_hat(0) << "\n";
    }

    return 0;
}
