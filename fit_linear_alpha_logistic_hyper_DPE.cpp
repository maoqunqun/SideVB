#include <RcppArmadillo.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <cmath>

using namespace Rcpp;
using namespace arma;

// Helper function: sigmoid with bounds checking to avoid overflow
inline double sigmoid_cpp(double x) {
  if (x > 32) {
    return 1.0;
  } else if (x < -32) {
    return 0.0;
  } else {
    return 1.0 / (1.0 + std::exp(-x));
  }
}

// Vectorized sigmoid with bounds checking
arma::vec sigmoid_vec_cpp(const arma::vec& x) {
  arma::vec result(x.n_elem);
  for (uword i = 0; i < x.n_elem; i++) {
    result(i) = sigmoid_cpp(x(i));
  }
  return result;
}

// Helper function: entropy of Bernoulli
arma::vec entrop_cpp(const arma::vec& gamma) {
  arma::vec result(gamma.n_elem, arma::fill::zeros);
  for (uword i = 0; i < gamma.n_elem; i++) {
    double g = gamma(i);
    // clamp values to avoid -Inf
    if ((g > 1e-10) && (g < 1.0 - 1e-10)) {
      result(i) = -g * std::log(g) - (1.0 - g) * std::log(1.0 - g);
    }
  }
  return result;
}

// Helper function: compute diagonal of X'X
arma::vec gram_diag_cpp(const arma::mat& X) {
  arma::vec result(X.n_cols);
  for (uword j = 0; j < X.n_cols; j++) {
    result(j) = std::pow(arma::norm(X.col(j)), 2);
  }
  return result;
}

// Build DPE prior covariance for theta
arma::mat build_side_sigma_dpe(int q, int p, double k, double rest_var) {
  arma::vec v(q, fill::ones);
  v *= rest_var;
  v(0) = std::exp(k) / (double)p;
  return diagmat(v);
}

//' Fit Linear Alpha Logistic Hyperparameters using Variational Bayes (C++ version)
//'
//' @param X Matrix of predictors (n x p)
//' @param Y Response vector (n x 1)
//' @param mu Initial mean vector for the variational distribution (p x 1)
//' @param gamma Initial inclusion probabilities for predictors (p x 1)
//' @param noise_sd_0 Initial noise standard deviation
//' @param update_order Vector specifying the order in which predictors are updated (1-indexed from R)
//' @param max_iter Maximum number of iterations for the VB algorithm
//' @param tol Convergence tolerance for the VB algorithm
//' @param side_prior List with side_mu (q x 1) and side_Sigma (q x q)
//' @param Z Matrix of side information (p x q)
//' @param gamma_prior_hyper_precision List with gamma_prior_hyper_precision_a and _b
//' @param dpe_exponents Integer vector of DPE steps
//' @param dpe_rest_variance Prior variance used for theta components 2:q (default 0.5)
//' @param dpe_update_prior_mean Logical; if TRUE, update prior mean dynamically
//' @param dpe_tol Convergence tolerance for DPE
//' @param dpe_metric String: "entropy", "max_abs_gamma", or "max_abs_theta"
//' @param vb_min_iter Minimum VB iterations before checking convergence
//' @param true_signal Optional true signal vector for computing MSE (can be R_NilValue)
//' @param verbose If true, print diagnostics
//'
//' @return A list containing the final VB results and DPE path
// [[Rcpp::export]]
List fit_linear_alpha_hyper_DPE(
    const arma::mat& X,
    const arma::vec& Y,
    arma::vec mu,
    arma::vec gamma,
    double noise_sd_0,
    const arma::uvec& update_order,  // 1-indexed from R
    int max_iter,
    double tol,
    List side_prior,
    const arma::mat& Z,
    List gamma_prior_hyper_precision,
    const arma::vec& dpe_exponents,
    double dpe_rest_variance = 0.5,
    bool dpe_update_prior_mean = true,
    double dpe_tol = 1e-6,
    std::string dpe_metric = "entropy",
    int vb_min_iter = 10,
    Rcpp::Nullable<Rcpp::NumericVector> true_signal = R_NilValue,
    bool verbose = true
) {
  
  // Extract from lists (matching R interface)
  arma::vec side_mu = as<arma::vec>(side_prior["side_mu"]);
  arma::mat side_Sigma = as<arma::mat>(side_prior["side_Sigma"]);
  double gamma_prior_hyper_precision_a = as<double>(gamma_prior_hyper_precision["gamma_prior_hyper_precision_a"]);
  double gamma_prior_hyper_precision_b = as<double>(gamma_prior_hyper_precision["gamma_prior_hyper_precision_b"]);
  
  // Convert dpe_metric string to int
  int dpe_metric_int = 0;
  if (dpe_metric == "max_abs_gamma") {
    dpe_metric_int = 1;
  } else if (dpe_metric == "max_abs_theta") {
    dpe_metric_int = 2;
  }
  
  // Handle nullable true_signal
  arma::vec true_signal_vec;
  bool has_true_signal = false;
  if (true_signal.isNotNull()) {
    true_signal_vec = as<arma::vec>(true_signal.get());
    has_true_signal = true;
  }
  
  // int n = X.n_rows;  // unused
  int p = X.n_cols;
  int q = side_mu.n_elem;
  int n_dpe_steps = dpe_exponents.n_elem;
  
  // Store DPE path
  List dpe_path(n_dpe_steps);
  
  // These get carried forward across DPE steps
  arma::vec mu_k = mu;
  arma::vec gamma_k = gamma;
  arma::vec side_prior_mu_k = side_mu;
  double gamma_hyper_a = gamma_prior_hyper_precision_a;
  double gamma_hyper_b = gamma_prior_hyper_precision_b;
  
  int last_step_idx = 0;
  
  // Precompute
  arma::vec YX_vec = X.t() * Y;
  arma::vec half_diag = gram_diag_cpp(X);
  
  for (int step_idx = 0; step_idx < n_dpe_steps; step_idx++) {
    double k = dpe_exponents(step_idx);
    
    if (verbose) {
      Rcout << "DPE step " << (step_idx + 1) << " with exponent " << k << std::endl;
    }
    
    // Build DPE prior covariance
    arma::mat side_Sigma_step = build_side_sigma_dpe(q, p, k, dpe_rest_variance);
    arma::vec side_mu_step = side_prior_mu_k;
    
    // Keep a copy for DPE metric
    arma::vec gamma_prev_step = gamma_k;
    arma::vec theta_prev_step;
    bool has_theta_prev = false;
    if (step_idx > 0) {
      List prev_step = dpe_path[step_idx - 1];
      if (prev_step.containsElementNamed("side_mu_vb")) {
        arma::vec temp = as<arma::vec>(prev_step["side_mu_vb"]);
        theta_prev_step = temp;
        has_theta_prev = true;
      }
    }
    
    // Initialize variables
    arma::vec old_entr = entrop_cpp(gamma_k);
    arma::vec approx_mean = gamma_k % mu_k;
    arma::vec X_appm = X * approx_mean;
    double noise_sd = noise_sd_0;
    int count = 0;
    
    // Extract prior parameters for side information
    arma::rowvec logit_pi = side_mu_step.t() * Z.t();
    arma::mat side_P = inv_sympd(side_Sigma_step);
    arma::vec side_Pmu = side_P * side_mu_step;
    
    // Expected precision
    double expected_precision_post;
    if (std::isnan(gamma_hyper_a / gamma_hyper_b)) {
      expected_precision_post = 1.0;
    } else {
      expected_precision_post = gamma_hyper_a / gamma_hyper_b;
    }
    
    // Initialize omega
    arma::vec side_omega(p, fill::ones);
    side_omega *= 0.25;
    
    // VB results (declared outside loop)
    arma::vec s_p(p);
    double const_lodds;
    arma::mat side_Sigma_vb;
    arma::vec side_mu_vb;
    double gamma_prior_hyper_precision_a_post, gamma_prior_hyper_precision_b_post;
    
    // Variational Bayes (VB) loop
    for (int i = 0; i < max_iter; i++) {
      
      const_lodds = 0.5 * std::log(expected_precision_post);
      s_p = 1.0 / sqrt(half_diag / (noise_sd * noise_sd) + expected_precision_post);
      
      // Update gamma and mu (using 1-indexed update_order from R, convert to 0-indexed)
      for (uword idx = 0; idx < update_order.n_elem; idx++) {
        int j = update_order(idx) - 1;  // Convert to 0-indexed
        
        X_appm -= approx_mean(j) * X.col(j);
        mu_k(j) = s_p(j) * s_p(j) / (noise_sd * noise_sd) * 
                  (YX_vec(j) - dot(X.col(j), X_appm));
        gamma_k(j) = sigmoid_cpp(logit_pi(j) + const_lodds + std::log(s_p(j)) + 
                                  0.5 * std::pow(mu_k(j) / s_p(j), 2));
        approx_mean(j) = gamma_k(j) * mu_k(j);
        X_appm += approx_mean(j) * X.col(j);
      }
      
      // Update precision of b
      arma::vec expect_b_square = (1.0 - gamma_k) * (1.0 / expected_precision_post) + 
                                   gamma_k % (mu_k % mu_k + s_p % s_p);
      
      gamma_prior_hyper_precision_a_post = gamma_hyper_a + (double)p / 2.0;
      gamma_prior_hyper_precision_b_post = gamma_hyper_b + 0.5 * sum(expect_b_square);
      expected_precision_post = gamma_prior_hyper_precision_a_post / gamma_prior_hyper_precision_b_post;
      
      // Logistic (side info) VB update
      arma::mat Z_omega = Z.each_col() % side_omega;
      arma::mat side_P_vb = Z_omega.t() * Z + side_P;
      side_Sigma_vb = inv_sympd(side_P_vb);
      side_mu_vb = side_Sigma_vb * (Z.t() * (gamma_k - 0.5) + side_Pmu);
      
      arma::vec side_eta = Z * side_mu_vb;
      arma::mat ZSigma = Z * side_Sigma_vb;
      arma::vec side_xi = sqrt(side_eta % side_eta + sum(ZSigma % Z, 1));
      side_omega = tanh(side_xi / 2.0) / (2.0 * side_xi);
      side_omega.replace(datum::nan, 0.25);
      
      logit_pi = side_mu_vb.t() * Z.t();
      
      // Check convergence
      arma::vec new_entr = entrop_cpp(gamma_k);
      if (i >= vb_min_iter && max(abs(new_entr - old_entr)) <= tol) {
        break;
      }
      old_entr = new_entr;
      count++;
    }
    
    if (verbose) {
      Rcout << "Iteration: " << count << std::endl;
    }
    
    // --- Compute metrics for this DPE step ---
    arma::vec theta_mean = side_mu_vb;
    arma::vec theta_se = sqrt(diagvec(side_Sigma_vb));
    
    // Sparsity measures: compute pi_j for each j
    arma::vec pi_vec(p);
    for (int j = 0; j < p; j++) {
      double eta_j = dot(Z.row(j).t(), side_mu_vb);
      pi_vec(j) = sigmoid_cpp(eta_j);
    }
    double sparsity_pi = sum(pi_vec) / (double)p;
    double sparsity_psi = sum(gamma_k) / (double)p;
    
    // MSE of signal
    double mse_signal = NA_REAL;
    arma::vec estimated_beta = mu_k % gamma_k;
    if (has_true_signal) {
      arma::vec estimated_signal = X * estimated_beta;
      arma::vec diff = true_signal_vec - estimated_signal;
      mse_signal = mean(diff % diff);
    }
    
    // Print diagnostics
    if (verbose) {
      Rcout << "\n--- DPE Step " << (step_idx + 1) << " (v_0 = exp(" << k << ")/p) ---" << std::endl;
      Rcout << "Theta posterior means:   ";
      for (int i = 0; i < q; i++) Rcout << theta_mean(i) << " ";
      Rcout << std::endl;
      Rcout << "Theta posterior SEs:     ";
      for (int i = 0; i < q; i++) Rcout << theta_se(i) << " ";
      Rcout << std::endl;
      Rcout << "Sparsity (mean psi_j):   " << sparsity_psi << std::endl;
      Rcout << "Sparsity (mean pi_j):    " << sparsity_pi << std::endl;
      if (!std::isnan(mse_signal)) {
        Rcout << "MSE of signal:           " << mse_signal << std::endl;
      }
      Rcout << std::endl;
    }
    
    // Store DPE step result
    List res_step = List::create(
      Named("mu") = mu_k,
      Named("s") = noise_sd * s_p,
      Named("gamma") = gamma_k,
      Named("const_lodds") = const_lodds,
      Named("noise_sd") = noise_sd,
      Named("count") = count,
      Named("side_mu_vb") = side_mu_vb,
      Named("side_Sigma_vb") = side_Sigma_vb,
      Named("gamma_prior_hyper_precision_a_post") = gamma_prior_hyper_precision_a_post,
      Named("gamma_prior_hyper_precision_b_post") = gamma_prior_hyper_precision_b_post,
      Named("dpe_k") = k,
      Named("dpe_side_prior_Sigma") = side_Sigma_step,
      Named("dpe_side_prior_mu") = side_mu_step,
      Named("theta_mean") = theta_mean,
      Named("theta_se") = theta_se,
      Named("sparsity_psi") = sparsity_psi,
      Named("sparsity_pi") = sparsity_pi,
      Named("mse_signal") = mse_signal
    );
    dpe_path[step_idx] = res_step;
    last_step_idx = step_idx;
    
    // ----- DPE early stopping -----
    double dpe_diff;
    switch (dpe_metric_int) {
      case 0: { // entropy
        arma::vec curr_entr = entrop_cpp(gamma_k);
        arma::vec prev_entr = entrop_cpp(gamma_prev_step);
        dpe_diff = max(abs(curr_entr - prev_entr));
        break;
      }
      case 1: { // max_abs_gamma
        dpe_diff = max(abs(gamma_k - gamma_prev_step));
        break;
      }
      case 2: { // max_abs_theta
        if (has_theta_prev) {
          dpe_diff = max(abs(side_mu_vb - theta_prev_step));
        } else {
          dpe_diff = R_PosInf;
        }
        break;
      }
      default:
        dpe_diff = R_PosInf;
    }
    
    if (R_finite(dpe_diff) && dpe_diff <= dpe_tol) {
      break;
    }
    
    // Dynamic update for next DPE step
    if (dpe_update_prior_mean) {
      side_prior_mu_k = side_mu_vb;
    }
  }
  
  // Trim path and return
  List final_dpe_path(last_step_idx + 1);
  for (int i = 0; i <= last_step_idx; i++) {
    final_dpe_path[i] = dpe_path[i];
  }
  
  List final = clone(as<List>(dpe_path[last_step_idx]));
  final["dpe_exponents"] = dpe_exponents;
  arma::vec dpe_exponents_used = dpe_exponents.subvec(0, last_step_idx);
  final["dpe_exponents_used"] = dpe_exponents_used;
  final["dpe_path"] = final_dpe_path;
  final["dpe_tol"] = dpe_tol;
  final["dpe_metric"] = dpe_metric;
  
  return final;
}
