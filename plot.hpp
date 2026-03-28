#pragma once

#include "cpp_files/matplotlibcpp.h"
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>

namespace plt = matplotlibcpp;

namespace kf {

struct StateHistory {
    std::string name;
    std::vector<double> timestamps;
    std::vector<Eigen::VectorXd> states;
    std::vector<Eigen::MatrixXd> covariances;

    StateHistory(const std::string& name);

    void record(double t, const Eigen::VectorXd& x);
    void record(double t, const Eigen::VectorXd& x, const Eigen::MatrixXd& P);
    void clear();
};

void plot_states(const std::vector<StateHistory>& histories,
                 const std::vector<std::string>& state_labels = {},
                 const std::string& title = "State Estimation History",
                 bool show_covariance = true,
                 const std::string& save_prefix = "");

void plot_state(const StateHistory& history,
                int state_idx = 0,
                const std::string& label = "",
                bool show_covariance = true);

void plot_measurements(const std::vector<double>& timestamps,
                       const std::vector<Eigen::VectorXd>& measurements,
                       const std::vector<std::string>& labels = {});

}  // namespace kf
