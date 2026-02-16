## ------------------------------------------------------------------------
## SideVB wrapper (hyper + DPE)
## ------------------------------------------------------------------------

#' Run SideVB with side information (DPE wrapper)
#'
#' @description
#' Wrapper for the SideVB variational Bayes (VB) linear regression with side
#' information and DPE (Dynamic Posterior Exploration). This function prepares
#' common initial values (via LASSO and ridge) and then calls
#' `fit_linear_alpha_hyper_DPE()`.
#'
#' This file is designed for sourcing in scripts. It assumes helper utilities
#' (e.g., `logit()`) are available (typically via sourcing
#' `VB_functions_for_load.R`), and the core VB routine is available (typically
#' via sourcing `fit_linear_alpha_logistic_hyper_DPE.R`).
#'
#' @param data A list with elements `X` (n x p design matrix), `Y` (length-n
#' response), `Z` (p x q side information matrix), and scalars `p` and `q`.
#' Typically `Z[, 1]` is an intercept column of 1s.
#' @param side_prior_variance Numeric prior variance for the logistic
#' regression coefficients. Can be a scalar (applied to all q coefficients) or
#' a length-q vector.
#' @param tol Numeric convergence tolerance passed to `fit_linear_alpha_hyper_DPE()`.
#' @param max_iter Integer maximum iterations passed to `fit_linear_alpha_hyper_DPE()`.
#' @param mu_0 Optional numeric vector (length p) initial VB mean for beta.
#' @param update_order Optional integer vector of indices (length p) giving the
#' coordinate update order.
#' @param s_hat Optional binary vector (length p) used as an initial selection
#' indicator from LASSO.
#' @param noise_sd_hat Optional numeric initial estimate of noise SD.
#' @param gamma Optional numeric vector (length p) initial inclusion
#' probabilities.
#' @param estimated_beta_lasso Optional numeric vector (length p) of LASSO
#' coefficients for reporting.
#' @param seed_val Integer seed used for reproducible LASSO / ridge
#' initialization.
#' @param gamma_prior_hyper_precision_a Numeric hyperparameter (shape) for the
#' gamma prior precision.
#' @param gamma_prior_hyper_precision_b Numeric hyperparameter (rate) for the
#' gamma prior precision.
#' @param theta0_intercept_only Logical. If `TRUE`, initialize theta prior mean
#' as intercept-only (`(logit(pi0), 0, ..., 0)`). If `FALSE`, try to initialize
#' theta via a logistic regression of `s_hat` on `Z[, -1]`.
#' @param dpe_exponents Integer vector controlling the DPE schedule.
#' @param dpe_rest_variance Numeric, rest variance used by DPE.
#' @param dpe_update_prior_mean Logical, whether DPE updates prior mean.
#' @param dpe_tol Numeric tolerance for DPE-level convergence.
#' @param dpe_metric Character. DPE metric passed to `fit_linear_alpha_hyper_DPE()`.
#' @param vb_min_iter Integer. Minimum VB iterations per DPE step.
#'
#' @return A list with elements:
#' - `sideVB_result`: full return from `fit_linear_alpha_hyper_DPE()`.
#' - `estimated_beta`: posterior mean beta (`mu * gamma`).
#' - `estimated_post_gamma`: posterior inclusion probabilities.
#' - `estimated_theta`: posterior mean theta.
#' - `estimated_pi`: posterior inclusion probabilities from the logistic model.
#' - plus initialization diagnostics (`estimated_beta_lasso`, `mu_0`, etc.).
#'
#' @seealso
#' `fit_linear_alpha_hyper_DPE` (core routine), `logit` (utility), and
#' `glmnet::cv.glmnet` (for initialization).

SideVB_wrapper_hyper_DPE <- function(
  data, # Dataset containing X, Y, Z, q, p
  side_prior_variance,
  tol,
  max_iter,
  mu_0,
  update_order,
  s_hat,
  noise_sd_hat,
  gamma,
  estimated_beta_lasso,
  seed_val,
  gamma_prior_hyper_precision_a,
  gamma_prior_hyper_precision_b,
  theta0_intercept_only = FALSE,
  # ---- DPE controls ----
  dpe_exponents = 0:6,
  dpe_rest_variance = 1/2,
  dpe_update_prior_mean = TRUE,
  dpe_tol = tol,
  dpe_metric = c("entropy", "max_abs_gamma", "max_abs_theta"),
  # ---- VB-in-DPE controls (match core) ----
  vb_min_iter = 10
) {

  dpe_metric <- match.arg(dpe_metric)

  ### Extract data components ###
  X <- data$X  # Design matrix
  Y <- data$Y  # Response vector
  Z <- data$Z  # Side information matrix
  q <- data$q  # Number of side information columns
  p <- data$p  # Number of features

  ### Fit LASSO to get starting values ###
  if (missing(s_hat)) {
    set.seed(seed_val)
    s_hat <- rep(1, p)  # Initialize s_hat as all ones
    while (sum(s_hat) + 2 > length(Y)) {  # Ensure p>n
      cvfit <- glmnet::cv.glmnet(X, Y, intercept = TRUE, alpha = 1)  # Fit LASSO
      s_hat <- 1 * I(coef(cvfit, s = "lambda.min")[-1] != 0)  # Extract non-zero coefficients locations (1 if selected, 0 otherwise)
      # print(paste0("Sum of s_hat is: ", sum(s_hat)))
    }
  }

  # Estimate beta coefficients and noise standard deviation
  if (missing(noise_sd_hat) || missing(estimated_beta_lasso)) {
    # If s_hat was supplied by the caller, cvfit may not exist. Only in that
    # case, refit LASSO to obtain estimated_beta_lasso and noise_sd_hat.
    if (!exists("cvfit", inherits = FALSE)) {
      set.seed(seed_val)
      cvfit <- glmnet::cv.glmnet(X, Y, intercept = TRUE, alpha = 1)
    }
    estimated_beta_lasso <- coef(cvfit, s = "lambda.min")[-1]
    y_hat <- predict(cvfit, X, s = "lambda.min")
    noise_sd_hat <- sqrt(sum((Y - y_hat)^2) / max((length(Y) - sum(s_hat) - 1), 1))
    # print(paste("noise_sd_hat:", round(noise_sd_hat, 2)))
  }

  ### Fit ridge regression to get initial mu_0 and update_order ###
  if (missing(mu_0) || missing(update_order)) {
    set.seed(seed_val)
    cvfit_ridge <- glmnet::cv.glmnet(X, Y, intercept = TRUE, alpha = 0)  # Fit ridge regression
    mu_0 <- coef(cvfit_ridge, s = "lambda.min")[-1]  # Extract coefficients
    approx_mean <- mu_0
    update_order <- order(abs(approx_mean[1:p]), decreasing = TRUE)  # Order by magnitude
  }

  ### Ensure at least one feature is selected ###
  if (max(s_hat) == 0) {
    s_hat[which.max(abs(mu_0))] <- 1  # Select the feature with the largest coefficient
  }

  ### Set prior for logistic regression ###
  # Allow side_prior_variance to be scalar or length-q vector
  if (length(side_prior_variance) == 1) {
    side_Prior_sigma <- diag(rep(side_prior_variance, q), q)
  } else {
    side_Prior_sigma <- diag(side_prior_variance, q)
  }

  # Use LASSO selection (s_hat) to initialize theta prior mean via ridge-logistic on Z
  # side_prior_mu_s_hat <- NULL
  # if (!missing(s_hat) && length(s_hat) == p && q > 0) {
  if (theta0_intercept_only == FALSE) {
    side_prior_mu_s_hat <- tryCatch({
      # Treat Z columns as the full linear predictor basis (including intercept if Z has a 1s column)
      # Lasso logistic regression (alpha=1) to get stable initialization
      cvfit_side <- glm(as.numeric(s_hat) ~ Z[,-1], 
                        family = binomial
      )
      theta0 <- as.numeric(coef(cvfit_side))  # length q
      # cat("theta0:", theta0, "\n")
      # Guard against numerical explosions (separation); keep finite, mild clamp
      # theta0[!is.finite(theta0)] <- 0
      # theta0 <- pmin(pmax(theta0, -10), 10)
      theta0
    }, error = function(e) NULL)
  }
  
  # Fallback: original intercept-only initialization
  if (theta0_intercept_only == TRUE) {
    side_pi_0 <- max(c(sum(s_hat) / p, 0.01))  # Initial inclusion probability
    side_prior_mu_s_hat <- c(logit(side_pi_0), rep(0, q - 1))
  }

  # If the non-intercept initialization failed, fall back to intercept-only.
  # (This does not affect typical runs where glm() succeeds.)
  if (is.null(side_prior_mu_s_hat) || length(side_prior_mu_s_hat) != q) {
    side_pi_0 <- max(c(sum(s_hat) / p, 0.01))
    side_prior_mu_s_hat <- c(logit(side_pi_0), rep(0, q - 1))
  }
  
  # side_prior_mu_s_hat[1] <- logit(max(c(sum(s_hat) / p, 0.01)))  # Set intercept prior mean based --- very useful (intercept scale changed to ward true)
  cat("side_prior_mu_s_hat:", side_prior_mu_s_hat, "\n")
  
  side_prior <- list(side_mu = side_prior_mu_s_hat, side_Sigma = side_Prior_sigma)  # Combine prior parameters
  gamma_prior_hyper_precision <- list(
    gamma_prior_hyper_precision_a = gamma_prior_hyper_precision_a,
    gamma_prior_hyper_precision_b = gamma_prior_hyper_precision_b
  )

  ### Initialize gamma if missing ###
  if (missing(gamma)) {
    gamma <- rep(0.5, p)  # Initialize gamma as 0.5 for all features
  }

  ### Run the VB algorithm (DPE) ###
  start_time <- Sys.time()
  sideVB_result <- fit_linear_alpha_hyper_DPE(
    X = X, Y = Y,
    mu = mu_0,
    gamma = gamma,
    noise_sd_0 = noise_sd_hat,
    update_order = update_order,
    max_iter = max_iter,
    tol = tol,
    side_prior = side_prior,
    Z = Z,
    gamma_prior_hyper_precision = gamma_prior_hyper_precision,
    # ---- pass DPE controls through (NEW) ----
    dpe_exponents = dpe_exponents,
    dpe_rest_variance = dpe_rest_variance,
    dpe_update_prior_mean = dpe_update_prior_mean,
    dpe_tol = dpe_tol,
    dpe_metric = dpe_metric,
    # ---- pass VB-in-DPE control through ----
    vb_min_iter = vb_min_iter
  )
  end_time <- Sys.time()

  sidevb_runtime <- round(difftime(end_time, start_time, units = "secs"), 1)
  print(paste("SideVB run time:", sidevb_runtime, "secs"))

  ### Extract results ###
  estimated_beta <- sideVB_result$mu * sideVB_result$gamma  # Posterior mean of coefficients
  estimated_post_gamma <- sideVB_result$gamma  # Posterior inclusion probabilities
  estimated_theta <- sideVB_result$side_mu_vb  # Posterior mean of logistic regression coefficients
  theta_Z <- t(estimated_theta) %*% t(Z)  # Compute theta * Z
  estimated_pi <- c(exp(theta_Z) / (1 + exp(theta_Z)))  # Posterior probabilities of inclusion for side information

  ### Return results ###
  return(list(
    sideVB_result = sideVB_result,
    estimated_beta = estimated_beta,
    estimated_post_gamma = estimated_post_gamma,
    estimated_theta = estimated_theta,
    estimated_pi = estimated_pi,
    estimated_beta_lasso = estimated_beta_lasso,
    iteration_count = sideVB_result$count,
    gamma_prior_hyper_precision_a_post = sideVB_result$gamma_prior_hyper_precision_a_post,
    gamma_prior_hyper_precision_b_post = sideVB_result$gamma_prior_hyper_precision_b_post,
    s_hat = s_hat,
    noise_sd_hat = noise_sd_hat,
    mu_0 = mu_0,
    update_order = update_order,
    sidevb_runtime = sidevb_runtime
    # ,
    # gamma = gamma
  ))
}


