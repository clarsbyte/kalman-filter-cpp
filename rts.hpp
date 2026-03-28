#pragma once

#include <Eigen/Dense>
#include <vector>

namespace rts {

class Smoother {
public:
    Smoother(
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

}  // namespace rts
