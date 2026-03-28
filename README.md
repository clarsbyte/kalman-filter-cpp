# filtercpp

A single-header C++ library for Kalman filtering using Eigen.

```cpp
#include "filtercpp.h"
```

## Features

- **KalmanFilter** — Linear Kalman Filter
- **ExtendedKalmanFilter** — Extended Kalman Filter (user-supplied Jacobians)
- **UnscentedKalmanFilter** — Unscented Kalman Filter (Merwe scaled sigma points)
- **RTSSmoother** — Rauch-Tung-Striebel smoother (batch backward pass over KF output)
- **Single header** — just copy `filtercpp.h` and include it
- Optional **plotting** via matplotlibcpp (`plot.hpp` / `plot.cpp`)

Everything lives in the `filtercpp` namespace.

## Dependencies

- [Eigen3](https://eigen.tuxfamily.org) — linear algebra
- C++17 or later

For plotting only:
- Python 3 with [matplotlib](https://matplotlib.org)
- matplotlibcpp (`src/matplotlibcpp.h` is included)

## Installation

Copy `filtercpp.h` into your project and compile with Eigen on the include path:

```bash
g++ -std=c++17 -I/usr/include/eigen3 your_file.cpp -o your_program
```

## Usage

### Kalman Filter

State: `x = [position, velocity]`, measurement: position only.

```cpp
#include "filtercpp.h"

Eigen::MatrixXd F(2, 2);
F << 1, dt,
     0,  1;

Eigen::MatrixXd B = Eigen::MatrixXd::Zero(2, 1);

Eigen::MatrixXd H(1, 2);
H << 1, 0;

Eigen::MatrixXd P = Eigen::MatrixXd::Identity(2, 2) * 1.0;
Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1) * 0.5;

filtercpp::KalmanFilter filter(F, B, H, P, R, Q);

Eigen::VectorXd x0(2);
x0 << 0.0, 1.0;
filter.init(x0);

filter.predict(u);
filter.update(z);

Eigen::VectorXd x_hat = filter.get_state();
Eigen::MatrixXd P_hat = filter.get_covariance();
```

### Extended Kalman Filter

For nonlinear systems. You provide the dynamics function `f`, its Jacobian `F`, the measurement function `h`, and its Jacobian `H`.

```cpp
#include "filtercpp.h"

// Nonlinear pendulum: state = [theta, theta_dot]
Eigen::VectorXd f(const Eigen::VectorXd& x, const Eigen::VectorXd& /*u*/) {
    Eigen::VectorXd x_new(2);
    x_new(0) = x(0) + dt * x(1);
    x_new(1) = x(1) - dt * gL * std::sin(x(0));
    return x_new;
}

Eigen::MatrixXd F(const Eigen::VectorXd& x, const Eigen::VectorXd& /*u*/) {
    Eigen::MatrixXd Fmat(2, 2);
    Fmat << 1,                    dt,
           -dt * gL * cos(x(0)),  1;
    return Fmat;
}

Eigen::VectorXd h(const Eigen::VectorXd& x) {
    Eigen::VectorXd z(1);
    z(0) = std::sin(x(0));
    return z;
}

Eigen::MatrixXd H(const Eigen::VectorXd& x) {
    Eigen::MatrixXd Hmat(1, 2);
    Hmat << std::cos(x(0)), 0;
    return Hmat;
}

filtercpp::ExtendedKalmanFilter ekf(f, F, h, H, P, R, Q);
ekf.init(x0);

ekf.predict(u);
ekf.update(z);
```

### Unscented Kalman Filter

Same nonlinear system as EKF, but no Jacobians required — just `f` and `h`.

```cpp
#include "filtercpp.h"

filtercpp::UnscentedKalmanFilter ukf(f, h, P, R, Q);
// Optional sigma point tuning (defaults work well in most cases):
// filtercpp::UnscentedKalmanFilter ukf(f, h, P, R, Q, alpha, beta, kappa);

ukf.init(x0);

ukf.predict(u);
ukf.update(z);
```

### RTS Smoother

Run the KF forward pass first, collect all state estimates and covariances, then run the RTS backward pass over them. The smoother uses future measurements to refine past estimates.

```cpp
#include "filtercpp.h"

// Forward pass
std::vector<Eigen::VectorXd> Xs;
std::vector<Eigen::MatrixXd> Ps;

for (int i = 0; i < steps; i++) {
    filter.predict(u);
    filter.update(z[i]);
    Xs.push_back(filter.get_state());
    Ps.push_back(filter.get_covariance());
}

// Backward pass
filtercpp::RTSSmoother smoother(F, Q);
filtercpp::RTSSmoother::Result result = smoother.smooth(Xs, Ps);

// result.x[k] — smoothed state at step k
// result.P[k] — smoothed covariance at step k
```

To smooth only up to a specific timestep:

```cpp
auto result = smoother.get_state(Xs, Ps, timestep);
```

## Plotting

`plot.hpp` / `plot.cpp` provide a `StateHistory` recorder and plotting functions built on matplotlibcpp. Since matplotlibcpp links against Python, `plot.cpp` must be compiled alongside your code.

```bash
g++ -std=c++17 -I/usr/include/eigen3 $(python3-config --includes) \
    your_file.cpp plot.cpp \
    $(python3-config --ldflags --embed) -o your_program
```

### Recording history

```cpp
#include "plot.hpp"

kf::StateHistory kf_hist("KF");
kf::StateHistory truth_hist("Truth");

for (int i = 0; i < steps; i++) {
    filter.predict(u);
    filter.update(z[i]);

    kf_hist.record(t, filter.get_state(), filter.get_covariance());
    truth_hist.record(t, x_true);
}
```

### Plotting multiple filters

`plot_states` creates one figure per state dimension, comparing all provided histories side by side.

```cpp
kf::plot_states(
    {truth_hist, kf_hist, ekf_hist, ukf_hist},
    {"Position", "Velocity"},   // state labels (optional)
    "Filter Comparison",        // title
    false,                      // show 2-sigma covariance bounds
    "output"                    // save prefix — saves output_0.png, output_1.png, ...
);
```

To plot a single state dimension from one filter:

```cpp
kf::plot_state(kf_hist, 0 /*state index*/);
```

To overlay raw measurements:

```cpp
kf::plot_measurements(timestamps, measurements, {"position meas"});
```

## API Reference

### `filtercpp::KalmanFilter`

| Method | Description |
|---|---|
| `KalmanFilter(F, B, H, P, R, Q)` | Constructor |
| `init()` | Reset state to zero, restore initial P |
| `init(x)` | Reset and set initial state |
| `predict(u)` | Predict step with control input |
| `update(y)` | Update step with measurement |
| `get_state()` | Returns current `x_hat` |
| `get_covariance()` | Returns current `P` |

### `filtercpp::ExtendedKalmanFilter`

| Method | Description |
|---|---|
| `ExtendedKalmanFilter(f, F, h, H, P, R, Q)` | Constructor — `f`, `F`, `h`, `H` are `std::function` |
| `init()` / `init(x)` | Reset |
| `predict(u)` | Propagates state through `f`, updates P using `F` Jacobian |
| `update(y)` | Updates state using `h` and `H` Jacobian |
| `get_state()` / `get_covariance()` | Accessors |

### `filtercpp::UnscentedKalmanFilter`

| Method | Description |
|---|---|
| `UnscentedKalmanFilter(f, h, P, R, Q, alpha, beta, kappa)` | Constructor — `alpha`, `beta`, `kappa` are optional |
| `init()` / `init(x)` | Reset |
| `predict(u)` | Propagates sigma points through `f` |
| `update(z)` | Updates using sigma points propagated through `h` |
| `get_state()` / `get_covariance()` | Accessors |

### `filtercpp::RTSSmoother`

| Method | Description |
|---|---|
| `RTSSmoother(F, Q)` | Constructor |
| `smooth(Xs, Ps)` | Full backward pass over all KF estimates |
| `get_state(Xs, Ps, timestep)` | Backward pass up to a given timestep |

**`Result` fields:** `x` (smoothed states), `P` (smoothed covariances), `K` (smoother gains), `Pp` (predicted covariances)

### `kf::StateHistory` (plot.hpp)

| Method | Description |
|---|---|
| `StateHistory(name)` | Constructor |
| `record(t, x)` | Record state at time `t` |
| `record(t, x, P)` | Record state and covariance at time `t` |
| `clear()` | Clear all recorded data |

## Examples

All examples are in the `examples/` directory. Compile from the repo root.

### `plot_test.cpp` — Filter comparison on a nonlinear pendulum

Runs KF, EKF, and UKF on a pendulum (state: `[theta, theta_dot]`, measurement: `sin(theta)`) and plots all three estimates against ground truth. Demonstrates how the linear KF diverges at large angles while EKF and UKF track correctly.

```bash
g++ -std=c++17 -I/usr/include/eigen3 $(python3-config --includes) \
    examples/plot_test.cpp plot.cpp \
    $(python3-config --ldflags --embed) \
    -o examples/plot_test

./examples/plot_test
```

Saves `kf_comparison_0.png` (theta) and `kf_comparison_1.png` (theta_dot).

### `test_ekf.cpp` — EKF on a nonlinear pendulum

```bash
g++ -std=c++17 -I/usr/include/eigen3 examples/test_ekf.cpp -o examples/test_ekf
./examples/test_ekf
```

### `test_ukf.cpp` — UKF on a nonlinear pendulum

```bash
g++ -std=c++17 -I/usr/include/eigen3 examples/test_ukf.cpp -o examples/test_ukf
./examples/test_ukf
```

### `test_rts.cpp` — KF forward pass + RTS backward pass with plotting

```bash
g++ -std=c++17 -I/usr/include/eigen3 $(python3-config --includes) \
    examples/test_rts.cpp plot.cpp \
    $(python3-config --ldflags --embed) \
    -o examples/test_rts

./examples/test_rts
```

Saves `rts_comparison_0.png` and `rts_comparison_1.png`.

## Project Structure

```
kalman-filter-cpp/
├── filtercpp.h         # Single-header library (KF, EKF, UKF, RTSSmoother)
├── plot.hpp            # Plotting interface (requires plot.cpp)
├── plot.cpp            # Plotting implementation
├── src/                # Original split .hpp/.cpp sources + matplotlibcpp.h
└── examples/
    ├── plot_test.cpp   # KF vs EKF vs UKF on nonlinear pendulum with plots
    ├── test_ekf.cpp    # EKF on nonlinear pendulum
    ├── test_ukf.cpp    # UKF on nonlinear pendulum
    └── test_rts.cpp    # KF + RTS smoother with plotting
```
