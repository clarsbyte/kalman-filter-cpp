#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "../kf.hpp"
#include "../rts.hpp"
#include "../plot.hpp"

const double dt = 0.1;

double randn() {
    double u1 = ((double)rand() / RAND_MAX);
    double u2 = ((double)rand() / RAND_MAX);
    if (u1 < 1e-10) u1 = 1e-10;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

int main() {
    srand(42);

    Eigen::MatrixXd F(2, 2);
    F << 1, dt,
         0, 1;

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(2, 1);

    Eigen::MatrixXd H(1, 2);
    H << 1, 0;

    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(2, 2) * 1.0;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * 2.0;  // noisy measurements

    kf::KalmanFilter filter(F, B, H, P, R, Q);

    Eigen::VectorXd x_true(2);
    x_true << 0.0, 1.0;

    Eigen::VectorXd x_init(2);
    x_init << 0.0, 1.0;
    filter.init(x_init);

    Eigen::VectorXd u(1);
    u << 0.0;

    // Forward pass
    std::vector<Eigen::VectorXd> Xs;
    std::vector<Eigen::MatrixXd> Ps;

    kf::StateHistory kf_hist("KF");
    kf::StateHistory truth_hist("Truth");

    int steps = 50;
    for (int i = 0; i < steps; i++) {
        double t = i * dt;
        x_true = F * x_true;

        Eigen::VectorXd z(1);
        z(0) = x_true(0) + std::sqrt(2.0) * randn();

        filter.predict(u);
        filter.update(z);

        Xs.push_back(filter.get_state());
        Ps.push_back(filter.get_covariance());

        kf_hist.record(t, filter.get_state(), filter.get_covariance());
        truth_hist.record(t, x_true);
    }

    // RTS backward pass
    rts::Smoother smoother(F, Q);
    rts::Smoother::Result result = smoother.smooth(Xs, Ps);

    kf::StateHistory rts_hist("RTS");
    for (int i = 0; i < steps; i++) {
        rts_hist.record(i * dt, result.x[i], result.P[i]);
    }

    kf::plot_states(
        {truth_hist, kf_hist, rts_hist},
        {"Position", "Velocity"},
        "KF vs RTS Smoother",
        false,
        "rts_comparison"
    );

    return 0;
}
