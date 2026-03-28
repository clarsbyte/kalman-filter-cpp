#include "plot.hpp"
#include <cmath>

kf::StateHistory::StateHistory(const std::string& name) : name(name) {}

void kf::StateHistory::record(double t, const Eigen::VectorXd& x) {
    timestamps.push_back(t);
    states.push_back(x);
}

void kf::StateHistory::record(double t, const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
    timestamps.push_back(t);
    states.push_back(x);
    covariances.push_back(P);
}

void kf::StateHistory::clear() {
    timestamps.clear();
    states.clear();
    covariances.clear();
}

void kf::plot_states(const std::vector<StateHistory>& histories,
                     const std::vector<std::string>& state_labels,
                     const std::string& title,
                     bool show_covariance,
                     const std::string& save_prefix) {
    if (histories.empty() || histories[0].states.empty()) return;

    int n = histories[0].states[0].size();

    for (int si = 0; si < n; si++) {
        plt::figure();

        for (const auto& hist : histories) {
            std::vector<double> vals;
            for (const auto& s : hist.states) {
                vals.push_back(s(si));
            }
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

        std::string state_label;
        if (si < (int)state_labels.size()) {
            state_label = state_labels[si];
        } else {
            state_label = "x[" + std::to_string(si) + "]";
        }
        plt::title(title + " - " + state_label);
        plt::ylabel(state_label);
        plt::xlabel("Time");
        plt::legend();

        if (!save_prefix.empty()) {
            plt::save(save_prefix + "_" + std::to_string(si) + ".png");
        }
    }

    plt::show();
}

void kf::plot_state(const StateHistory& history,
                    int state_idx,
                    const std::string& label,
                    bool show_covariance) {
    if (history.states.empty()) return;

    std::vector<double> vals;
    for (const auto& s : history.states) {
        vals.push_back(s(state_idx));
    }

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

void kf::plot_measurements(const std::vector<double>& timestamps,
                           const std::vector<Eigen::VectorXd>& measurements,
                           const std::vector<std::string>& labels) {
    if (measurements.empty()) return;

    int m = measurements[0].size();

    for (int i = 0; i < m; i++) {
        std::vector<double> vals;
        for (const auto& z : measurements) {
            vals.push_back(z(i));
        }

        std::string label;
        if (i < (int)labels.size()) {
            label = labels[i];
        } else {
            label = "z[" + std::to_string(i) + "]";
        }

        plt::named_plot(label, timestamps, vals, "o");
    }
    plt::legend();
}
