#pragma once

//
// filtercpp.h — Header-only Kalman filtering library
//
// Filters:  KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
// Smoother: RTSSmoother
// Plotting: StateHistory, plot_states, plot_state, plot_measurements
//           (enabled by #define FILTERCPP_PLOT before including this header)
//
// All classes live in the filtercpp namespace.
// Core dependency: Eigen3
// Plot dependency: matplotlibcpp (Python 3 + matplotlib)


#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <vector>

#ifdef FILTERCPP_PLOT
#include "src/matplotlibcpp.h"
#include <string>
#include <map>
#endif

namespace filtercpp {


// Sigma Points (used internally by UnscentedKalmanFilter)

struct SigmaPointsResult {
    Eigen::MatrixXd sigma_points;
    Eigen::VectorXd Wm;
    Eigen::VectorXd Wc;
};

class SigmaPoints {
public:
    SigmaPoints(int n, double alpha = 1e-3, double beta = 2.0, double kappa = 0.0)
        : n(n), alpha(alpha), beta(beta), kappa(kappa)
    {
        lambda = (alpha * alpha) * (n + kappa) - n;
    }

    SigmaPointsResult generate(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
        SigmaPointsResult result;

        Eigen::LLT<Eigen::MatrixXd> llt(P);
        Eigen::MatrixXd L = llt.matrixL();
        Eigen::MatrixXd A = std::sqrt(n + lambda) * L;

        result.sigma_points = Eigen::MatrixXd(n, 2 * n + 1);
        result.sigma_points.col(0) = x;
        for (int i = 0; i < n; i++) {
            result.sigma_points.col(i + 1)     = x + A.col(i);
            result.sigma_points.col(i + 1 + n) = x - A.col(i);
        }

        result.Wm = Eigen::VectorXd(2 * n + 1);
        result.Wc = Eigen::VectorXd(2 * n + 1);
        result.Wm(0) = lambda / (n + lambda);
        result.Wc(0) = lambda / (n + lambda) + (1 - alpha * alpha + beta);
        double w = 1.0 / (2.0 * (n + lambda));
        for (int i = 1; i < 2 * n + 1; i++) {
            result.Wm(i) = w;
            result.Wc(i) = w;
        }

        return result;
    }

private:
    int n;
    double alpha, beta, kappa, lambda;
};

// Kalman Filter


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


// Extended Kalman Filter


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

// Unscented Kalman Filter

class UnscentedKalmanFilter {
public:
    UnscentedKalmanFilter(
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h,
        const Eigen::MatrixXd& P,
        const Eigen::MatrixXd& R,
        const Eigen::MatrixXd& Q,
        double alpha = 1e-3,
        double beta  = 2.0,
        double kappa = 0.0
    )
    : f(f), h(h), P(P), P0(P), R(R), Q(Q)
    , x_hat(Eigen::VectorXd::Zero(P.rows()))
    , sigma_points(P.rows(), alpha, beta, kappa)
    {}

    void init() {
        P = P0;
        x_hat.setZero();
    }

    void init(const Eigen::VectorXd& x) {
        init();
        x_hat = x;
    }

    void predict(const Eigen::VectorXd& u) {
        int n = x_hat.size();

        SigmaPointsResult sp_result = sigma_points.generate(x_hat, P);
        Eigen::MatrixXd& chi = sp_result.sigma_points;
        Eigen::VectorXd& Wm  = sp_result.Wm;
        Eigen::VectorXd& Wc  = sp_result.Wc;

        Eigen::MatrixXd Y(n, 2 * n + 1);
        for (int i = 0; i < 2 * n + 1; i++)
            Y.col(i) = f(chi.col(i), u);

        x_hat = Y * Wm;

        P = Q;
        for (int i = 0; i < 2 * n + 1; i++) {
            Eigen::VectorXd diff = Y.col(i) - x_hat;
            P += Wc(i) * diff * diff.transpose();
        }
    }

    void update(const Eigen::VectorXd& z) {
        int n = x_hat.size();

        SigmaPointsResult sp_result = sigma_points.generate(x_hat, P);
        Eigen::MatrixXd& Y  = sp_result.sigma_points;
        Eigen::VectorXd& Wm = sp_result.Wm;
        Eigen::VectorXd& Wc = sp_result.Wc;

        int m = h(Y.col(0)).size();
        Eigen::MatrixXd Z(m, 2 * n + 1);
        for (int i = 0; i < 2 * n + 1; i++)
            Z.col(i) = h(Y.col(i));

        Eigen::VectorXd mu_z = Z * Wm;

        Eigen::MatrixXd Pz = R;
        for (int i = 0; i < 2 * n + 1; i++) {
            Eigen::VectorXd dz = Z.col(i) - mu_z;
            Pz += Wc(i) * dz * dz.transpose();
        }

        Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(n, m);
        for (int i = 0; i < 2 * n + 1; i++) {
            Eigen::VectorXd dx = Y.col(i) - x_hat;
            Eigen::VectorXd dz = Z.col(i) - mu_z;
            Pxz += Wc(i) * dx * dz.transpose();
        }

        K      = Pxz * Pz.inverse();
        x_hat += K * (z - mu_z);
        P     -= K * Pz * K.transpose();
    }

    Eigen::VectorXd get_state()      { return x_hat; }
    Eigen::MatrixXd get_covariance() { return P; }

private:
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> f;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> h;
    Eigen::MatrixXd P, R, Q, P0, K;
    Eigen::VectorXd x_hat;
    SigmaPoints sigma_points;
};

// RTS Smoother

class RTSSmoother {
public:
    RTSSmoother(
        const Eigen::MatrixXd& F,
        const Eigen::MatrixXd& Q
    )
    : F(F), Q(Q)
    {}

    struct Result {
        std::vector<Eigen::VectorXd> x;
        std::vector<Eigen::MatrixXd> P;
        std::vector<Eigen::MatrixXd> K;
        std::vector<Eigen::MatrixXd> Pp;
    };

    Result smooth(
        const std::vector<Eigen::VectorXd>& Xs,
        const std::vector<Eigen::MatrixXd>& Ps)
    {
        const int n     = Xs.size();
        const int dim_x = F.rows();

        Result res;
        res.x  = Xs;
        res.P  = Ps;
        res.K.resize(n,  Eigen::MatrixXd::Zero(dim_x, dim_x));
        res.Pp.resize(n, Eigen::MatrixXd::Zero(dim_x, dim_x));

        for (int k = n - 2; k >= 0; k--) {
            res.Pp[k]  = F * res.P[k] * F.transpose() + Q;
            res.K[k]   = res.P[k] * F.transpose() * res.Pp[k].inverse();
            res.x[k]  += res.K[k] * (res.x[k+1] - F * res.x[k]);
            res.P[k]  += res.K[k] * (res.P[k+1] - res.Pp[k]) * res.K[k].transpose();
        }

        return res;
    }

    Result get_state(
        const std::vector<Eigen::VectorXd>& Xs,
        const std::vector<Eigen::MatrixXd>& Ps,
        int timestep)
    {
        const int dim_x = F.rows();

        Result res;
        res.x  = Xs;
        res.P  = Ps;
        res.K.resize(timestep,  Eigen::MatrixXd::Zero(dim_x, dim_x));
        res.Pp.resize(timestep, Eigen::MatrixXd::Zero(dim_x, dim_x));

        for (int k = timestep - 2; k >= 0; k--) {
            res.Pp[k]  = F * res.P[k] * F.transpose() + Q;
            res.K[k]   = res.P[k] * F.transpose() * res.Pp[k].inverse();
            res.x[k]  += res.K[k] * (res.x[k+1] - F * res.x[k]);
            res.P[k]  += res.K[k] * (res.P[k+1] - res.Pp[k]) * res.K[k].transpose();
        }

        return res;
    }

private:
    Eigen::MatrixXd F, Q;
};

}  // namespace filtercpp


// Plotting utilities (opt-in: #define FILTERCPP_PLOT before including) ──

#ifdef FILTERCPP_PLOT

namespace filtercpp {
namespace plt = matplotlibcpp;

struct StateHistory {
    std::string name;
    std::vector<double> timestamps;
    std::vector<Eigen::VectorXd> states;
    std::vector<Eigen::MatrixXd> covariances;

    StateHistory(const std::string& name) : name(name) {}

    void record(double t, const Eigen::VectorXd& x) {
        timestamps.push_back(t);
        states.push_back(x);
    }

    void record(double t, const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
        timestamps.push_back(t);
        states.push_back(x);
        covariances.push_back(P);
    }

    void clear() {
        timestamps.clear();
        states.clear();
        covariances.clear();
    }
};

inline void plot_states(const std::vector<StateHistory>& histories,
                        const std::vector<std::string>& state_labels = {},
                        const std::string& title = "State Estimation History",
                        bool show_covariance = true,
                        const std::string& save_prefix = "") {
    if (histories.empty() || histories[0].states.empty()) return;

    int n = histories[0].states[0].size();

    for (int si = 0; si < n; si++) {
        plt::figure();

        for (const auto& hist : histories) {
            std::vector<double> vals;
            for (const auto& s : hist.states)
                vals.push_back(s(si));
            plt::named_plot(hist.name, hist.timestamps, vals);

            if (show_covariance && !hist.covariances.empty()) {
                std::vector<double> upper, lower;
                for (size_t i = 0; i < hist.states.size(); i++) {
                    double sigma = std::sqrt(hist.covariances[i](si, si));
                    upper.push_back(hist.states[i](si) + 2.0 * sigma);
                    lower.push_back(hist.states[i](si) - 2.0 * sigma);
                }
                plt::plot(hist.timestamps, upper, "--");
                plt::plot(hist.timestamps, lower, "--");
            }
        }

        std::string state_label = (si < (int)state_labels.size())
            ? state_labels[si]
            : "x[" + std::to_string(si) + "]";

        plt::title(title + " - " + state_label);
        plt::ylabel(state_label);
        plt::xlabel("Time");
        plt::legend();

        if (!save_prefix.empty())
            plt::save(save_prefix + "_" + std::to_string(si) + ".png");
    }

    plt::show();
}

inline void plot_state(const StateHistory& history,
                       int state_idx = 0,
                       const std::string& label = "",
                       bool show_covariance = true) {
    if (history.states.empty()) return;

    std::vector<double> vals;
    for (const auto& s : history.states)
        vals.push_back(s(state_idx));

    std::string plot_label = label.empty() ? history.name : label;
    plt::named_plot(plot_label, history.timestamps, vals);

    if (show_covariance && !history.covariances.empty()) {
        std::vector<double> upper, lower;
        for (size_t i = 0; i < history.states.size(); i++) {
            double sigma = std::sqrt(history.covariances[i](state_idx, state_idx));
            upper.push_back(history.states[i](state_idx) + 2.0 * sigma);
            lower.push_back(history.states[i](state_idx) - 2.0 * sigma);
        }
        plt::plot(history.timestamps, upper, "--");
        plt::plot(history.timestamps, lower, "--");
    }

    plt::ylabel("x[" + std::to_string(state_idx) + "]");
    plt::xlabel("Time");
    plt::legend();
}

inline void plot_measurements(const std::vector<double>& timestamps,
                              const std::vector<Eigen::VectorXd>& measurements,
                              const std::vector<std::string>& labels = {}) {
    if (measurements.empty()) return;

    int m = measurements[0].size();
    for (int i = 0; i < m; i++) {
        std::vector<double> vals;
        for (const auto& z : measurements)
            vals.push_back(z(i));

        std::string label = (i < (int)labels.size())
            ? labels[i]
            : "z[" + std::to_string(i) + "]";

        plt::named_plot(label, timestamps, vals, "o");
    }
    plt::legend();
}

}  // namespace filtercpp

#endif  // FILTERCPP_PLOT
