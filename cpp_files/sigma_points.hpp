#include <Eigen/Dense>

namespace sp {
    struct SigmaPointsResult {
        Eigen::MatrixXd sigma_points;
        Eigen::VectorXd Wm;
        Eigen::VectorXd Wc;
    };

    class SigmaPoints {
    public:
        SigmaPoints(int n, double alpha = 1e-3, double beta = 2.0, double kappa = 0.0);
        ~SigmaPoints();

        SigmaPointsResult generate(const Eigen::VectorXd& x, const Eigen::MatrixXd& P);
    private:
        int n; // state dimension
        double alpha;
        double beta;
        double kappa;
        double lambda; // scaling parameter

        double clampSpread(double spread);
    };
}