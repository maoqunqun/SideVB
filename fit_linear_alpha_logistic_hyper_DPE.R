#' Fit Linear Alpha Logistic Hyperparameters using Variational Bayes
#'
#' This function performs Variational Bayes (VB) inference to fit a linear model 
#' with logistic hyperparameters. It updates the posterior distributions of the 
#' parameters iteratively until convergence or the maximum number of iterations is reached.
#'
#' @param X Matrix of predictors (n x p).
#' @param Y Response vector (n x 1).
#' @param mu Initial mean vector for the variational distribution (p x 1).
#' @param gamma Initial inclusion probabilities for predictors (p x 1).
#' @param noise_sd_0 Initial noise standard deviation.
#' @param update_order Vector specifying the order in which predictors are updated.
#' @param max_iter Maximum number of iterations for the VB algorithm.
#' @param tol Convergence tolerance for the VB algorithm.
#' @param side_prior List containing prior parameters for the side information:
#'   - `side_mu`: Mean vector for the prior (q x 1).
#'   - `side_Sigma`: Covariance matrix for the prior (q x q).
#' @param Z Matrix of side information (p x q). (One row per predictor.)
#' @param gamma_prior_hyper_precision List containing hyperparameters for the gamma prior:
#'   - `gamma_prior_hyper_precision_a`: Shape parameter of the gamma prior.
#'   - `gamma_prior_hyper_precision_b`: Rate parameter of the gamma prior.
#' @param dpe_exponents Integer vector of DPE steps. Use 0 for the old behavior.
#'   If `0:6`, the prior variance for theta is set to
#'   (exp(k)/p, 1/2, 1/2, 1/2, ...) at step k.
#' @param dpe_rest_variance Prior variance used for theta components 2:q (default 1/2).
#' @param dpe_tol Convergence tolerance for DPE.
#' @param dpe_metric Character string specifying the DPE convergence metric:
#'   - `"entropy"`: Entropy of the gamma distribution.
#'   - `"max_abs_gamma"`: Maximum absolute difference in inclusion probabilities.
#' @param true_signal Optional true signal vector (X %*% true_beta) for computing MSE. If NULL, MSE is not computed.
#' @param verbose Logical; if TRUE, print diagnostics for each DPE step (default TRUE).
#'
#' @return A list containing the final VB results, plus (if DPE is used) `dpe_path`.
fit_linear_alpha_hyper_DPE <- function(X, Y, mu, gamma,
                                       noise_sd_0, update_order,
                                       max_iter, tol,
                                       side_prior, Z,
                                       gamma_prior_hyper_precision,
                                       dpe_exponents = 0,
                                       dpe_rest_variance = 1/2,
                                       dpe_tol = tol,
                                       dpe_metric = c("entropy", "max_abs_gamma","max_abs_theta"),
                                       vb_min_iter = 10,
                                       true_signal = NULL,
                                       verbose = TRUE) {

  dpe_metric <- match.arg(dpe_metric)

  # --- helper: build the DPE prior covariance for theta ---
  build_side_sigma_dpe <- function(q, p, k, rest_var) {
    v <- rep(rest_var, q)
    v[1] <- exp(k) / p
    diag(v, nrow = q)
  }

  p <- ncol(X)
  q <- length(side_prior$side_mu)

  # Keep a path if multiple DPE steps are requested
  dpe_path <- vector("list", length(dpe_exponents))

  # These get carried forward across DPE steps
  mu_k <- mu
  gamma_k <- gamma
  gamma_hyper_k <- gamma_prior_hyper_precision

  # Initialize omega (length p because Z is p x q; omega is per-row / per-predictor)
  side_omega <- rep(1 / 4, p)
  
  # Extract gamma prior hyperparameters (carried across steps)
  gamma_prior_hyper_precision_a <- gamma_hyper_k$gamma_prior_hyper_precision_a
  gamma_prior_hyper_precision_b <- gamma_hyper_k$gamma_prior_hyper_precision_b
  # Compute initial expected precision
  if (is.nan(gamma_prior_hyper_precision_a / gamma_prior_hyper_precision_b)) {
    expected_precision_post <- 1
  } else {
    expected_precision_post <- gamma_prior_hyper_precision_a / gamma_prior_hyper_precision_b
  }
  # prior for theta
   
  side_prior_mu <- side_prior$side_mu
  logit_pi <- NULL
  
  last_step_idx <- 0
  

  for (step_idx in seq_along(dpe_exponents)) {
    print(paste0("DPE step ", step_idx, " with exponent ", dpe_exponents[step_idx]))
    k <- dpe_exponents[step_idx]

    # Update side prior variance according to schedule:
    # (exp(k)/p, 1/2, 1/2, 1/2, ...)
    # fix the prior mean for each step_idx
    side_prior$side_Sigma <- build_side_sigma_dpe(q = q, p = p, k = k, rest_var = dpe_rest_variance)

    # Initialize variables
    lowerbound <- numeric(max_iter)
    old_entr <- entrop(gamma_k)
    YX_vec <- as.vector(t(Y) %*% X)
    half_diag <- gram_diag(X)
    approx_mean <- gamma_k * mu_k
    cat("approx_mean[1:10", "] at start of DPE step:", sprintf("%.4f", approx_mean[1:10]), "\n")
    X_appm <- X %*% approx_mean
    noise_sd <- noise_sd_0
    count <- 0

    # Extract prior parameters for side information (for this DPE step)
    if(is.null(logit_pi)){
      logit_pi <- t(side_prior_mu) %*% t(Z)
    }
    
    side_P    <- solve(side_prior$side_Sigma)
    side_Pmu  <- c(side_P %*% side_prior_mu)



    
    # Keep a copy for a possible DPE metric based on gamma changes
    gamma_prev_step <- gamma_k
    theta_prev_step <- NULL  # will be set after VB finishes previous step if you store it
    # If you have a previous step in dpe_path, you can use it:
    if (step_idx > 1 && !is.null(dpe_path[[step_idx - 1]]$side_mu_vb)) {
      theta_prev_step <- as.vector(dpe_path[[step_idx - 1]]$side_mu_vb)
    }

    # Variational Bayes (VB) loop
    for (i in 1:max_iter) {

      const_lodds <- 0.5 * log(expected_precision_post)
      s_p <- 1 / sqrt(half_diag / (noise_sd^2) + expected_precision_post)

      # Update gamma and mu
      for (j in update_order) {
        X_appm <- X_appm - approx_mean[j] * X[, j]
        mu_k[j] <- s_p[j]^2 / (noise_sd^2) * (YX_vec[j] - sum(X[, j] * X_appm))
        gamma_k[j] <- sigmoid(logit_pi[j] + const_lodds + log(s_p[j]) + 0.5 * (mu_k[j] / s_p[j])^2)
        approx_mean[j] <- gamma_k[j] * mu_k[j]
        X_appm <- X_appm + approx_mean[j] * X[, j]
      }

      # update precision of b
      expect_b_square <- (1 - gamma_k) * (1 / expected_precision_post) + gamma_k * (mu_k^2 + s_p^2)
      gamma_prior_hyper_precision_a_post <- gamma_prior_hyper_precision_a + p / 2
      gamma_prior_hyper_precision_b_post <- gamma_prior_hyper_precision_b + 0.5 * sum(expect_b_square)
      expected_precision_post <- gamma_prior_hyper_precision_a_post / gamma_prior_hyper_precision_b_post

      # logistic (side info) VB update
      side_P_vb       <- crossprod(Z * side_omega, Z) + side_P
      side_Sigma_vb   <- solve(side_P_vb)
      side_mu_vb      <- side_Sigma_vb %*% (crossprod(Z, gamma_k - 0.5) + side_Pmu)

      side_eta        <- c(Z %*% side_mu_vb)
      side_xi         <- sqrt(side_eta^2 + rowSums((Z %*% side_Sigma_vb) * Z))
      side_omega      <- tanh(side_xi / 2) / (2 * side_xi)
      side_omega[is.nan(side_omega)] <- 0.25

      logit_pi <- t(side_mu_vb) %*% t(Z)

      # Check convergence
      new_entr <- entrop(gamma_k)
      if (i >= vb_min_iter && max(abs(new_entr - old_entr)) <= tol) {
        break
      } else {
        old_entr <- new_entr
      }
      count <- count + 1
    }
    
    cat("Iteration:", count,"\n")
    
    # --- Compute metrics for this DPE step ---
    # Posterior mean and SE for theta
    theta_mean <- as.vector(side_mu_vb)
    theta_se <- sqrt(diag(side_Sigma_vb))
    
    # Sparsity measures
    # pi_j = sigmoid(Z_j * theta) - inclusion prob based on side info
    # Compute each pi_j individually: pi_j = sigmoid(theta_0 + theta_1*Z_j1 + theta_2*Z_j2 + ...)
    pi_vec <- numeric(p)
    for (j in 1:p) {
      eta_j <- sum(Z[j, ] * side_mu_vb)  # linear predictor for predictor j
      pi_vec[j] <- sigmoid(eta_j)
    }
    sparsity_pi <- sum(pi_vec) / p  # (1/p) * sum_{j=1}^p pi_j
    
    # psi_j = gamma_j - actual posterior inclusion probabilities
    sparsity_psi <- sum(gamma_k) / p  # (1/p) * sum_{j=1}^p psi_j
    
    # MSE of signal (if true_signal provided)
    estimated_beta <- mu_k * gamma_k
    if (!is.null(true_signal)) {
      estimated_signal <- X %*% estimated_beta
      mse_signal <- mean((true_signal - estimated_signal)^2)
    } else {
      mse_signal <- NA
    }
    
    # Print diagnostics
    if (verbose) {
      cat("\n--- DPE Step", step_idx, "(v_0 = exp(", k, ")/p) ---\n")
      cat("Theta posterior means:  ", sprintf("%.4f", theta_mean), "\n")
      cat("Theta posterior SEs:    ", sprintf("%.4f", theta_se), "\n")
      cat("Sparsity (mean psi_j):  ", sprintf("%.4f", sparsity_psi), "\n")
      cat("Sparsity (mean pi_j):", sprintf("%.4f", sparsity_pi), "\n")
      if (!is.na(mse_signal)) {
        cat("MSE of signal:          ", sprintf("%.6f", mse_signal), "\n")
      }
      
      cat("\n")
    }
    
    # Store DPE step result
    res_step <- list(
      mu = mu_k, s = noise_sd * s_p, gamma = gamma_k,
      const_lodds = const_lodds,
      noise_sd = noise_sd,
      count = i,
      side_mu_vb = side_mu_vb, side_Sigma_vb = side_Sigma_vb,
      gamma_prior_hyper_precision_a_post = gamma_prior_hyper_precision_a_post,
      gamma_prior_hyper_precision_b_post = gamma_prior_hyper_precision_b_post,
      dpe_k = k,
      # New metrics
      theta_mean = theta_mean,
      theta_se = theta_se,
      sparsity_psi = sparsity_psi,
      sparsity_pi = sparsity_pi,
      mse_signal = mse_signal
    )
    dpe_path[[step_idx]] <- res_step
    last_step_idx <- step_idx

    # ----- DPE early stopping (outer-loop break) -----
    dpe_diff <- switch(
      dpe_metric,
      entropy = {
        # mirror VB inner loop: compare entropy vectors elementwise, then take max
        max(abs(entrop(gamma_k) - entrop(gamma_prev_step)))
      },
      max_abs_gamma = {
        # alternative: directly compare gamma between two consecutive DPE steps
        max(abs(gamma_k - gamma_prev_step))
      },
      max_abs_theta = {
        if (is.null(theta_prev_step)) Inf else max(abs(as.vector(side_mu_vb) - theta_prev_step))
      }
    )

    if (is.finite(dpe_diff) && (dpe_diff <= dpe_tol)) {
      break
    }
    # -----------------------------------------------
  }

  # Trim path if we broke early, and return last completed step
  dpe_path <- dpe_path[seq_len(last_step_idx)]
  final <- dpe_path[[last_step_idx]]
  final$dpe_exponents <- dpe_exponents
  final$dpe_exponents_used <- dpe_exponents[seq_len(last_step_idx)]
  final$dpe_path <- dpe_path
  final$dpe_tol <- dpe_tol
  final$dpe_metric <- dpe_metric

  return(final)
}