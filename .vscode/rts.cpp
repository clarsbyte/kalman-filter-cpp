#include "rts.hpp"

rts::Smoother::Smoother(
    const Eigen::MatrixXd& _F,
    const Eigen::MatrixXd& _Q
)
: F(_F)
, Q(_Q)
{}

rts::Smoother::Result rts::Smoother::smooth(
    const std::vector<Eigen::VectorXd>& Xs,
    const std::vector<Eigen::MatrixXd>& Ps)
{
    const int n = Xs.size();
    const int dim_x = F.rows();

    Result res;
    res.x  = Xs;
    res.P  = Ps;
    res.K.resize(n,  Eigen::MatrixXd::Zero(dim_x, dim_x));
    res.Pp.resize(n, Eigen::MatrixXd::Zero(dim_x, dim_x));

    for (int k = n - 2; k >= 0; k--) {
        res.Pp[k] = F * res.P[k] * F.transpose() + Q;
        res.K[k]  = res.P[k] * F.transpose() * res.Pp[k].inverse();
        res.x[k] += res.K[k] * (res.x[k+1] - F * res.x[k]);
        res.P[k] += res.K[k] * (res.P[k+1] - res.Pp[k]) * res.K[k].transpose();
    }

    return res;
}
