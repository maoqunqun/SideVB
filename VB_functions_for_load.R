## SideVB helper functions
##
## This file contains core utilities used by the SideVB simulation/analysis code.
## The goal is to keep these functions self-contained and reproducible.

# -------------------------------------------------------------------------

#' Variational Bayes linear regression (no side information)
#'
#' @description
#' Variational Bayes updates for a spike-and-slab style linear regression without
#' side information ("alpha_0" version). The algorithm iteratively updates the
#' posterior mean (`mu`), posterior standard deviation (`s`), and inclusion
#' probabilities (`gamma`).
#'
#' @param X Numeric matrix (n x p). Design matrix.
#' @param Y Numeric vector (length n) or matrix (n x 1). Response.
#' @param mu Numeric vector (length p). Initial posterior mean of coefficients.
#' @param gamma Numeric vector (length p). Initial inclusion probabilities.
#' @param sigma_b Numeric scalar. Prior standard deviation for coefficients.
#' @param noise_sd_0 Numeric scalar. Initial noise standard deviation.
#' @param update_order Integer vector. Coordinate update order (indices in 1:p).
#' @param max_iter Integer. Maximum iterations.
#' @param tol Numeric. Convergence tolerance on entropy difference.
#' @param a_pi Numeric scalar. Beta prior shape parameter a.
#' @param b_pi Numeric scalar. Beta prior shape parameter b.
#'
#' @return A list with elements:
#'   - `mu`: posterior mean (length p)
#'   - `s`: posterior sd (length p)
#'   - `gamma`: inclusion probabilities (length p)
#'   - `const_lodds`: constant log-odds term used in updates
#'   - `iter`: number of VB iterations executed
#'   - `noise_sd`: final noise sd (currently returned as initialized)
fit_linear_alpha_0 <- function(
  X,
  Y,
  mu,
  gamma,
  sigma_b,
  noise_sd_0,
  update_order,
  max_iter,
  tol,
  a_pi,
  b_pi
) {
  # Initialize variables
  old_entr <- entrop(gamma)  # Compute initial entropy of inclusion probabilities.
  YX_vec <- as.vector(t(Y) %*% X)  # Precompute the product of Y and each column of X.
  p <- ncol(X)  # Number of predictors.
  n <- nrow(X)  # Number of samples.
  half_diag <- gram_diag(X)  # Compute the diagonal of X'X (sum of squares for each column of X).
  approx_mean <- gamma * mu  # Initialize the approximate mean of the coefficients.
  X_appm <- X %*% approx_mean  # Compute the contribution of the approximate mean to the response.
  logit_pi <- digamma(a_pi) - digamma(b_pi)  # Compute the logit of the prior inclusion probability.
  pi <- a_pi / (a_pi + b_pi)  # Compute the prior inclusion probability.
  noise_sd <- noise_sd_0  # Initialize the noise standard deviation.
  const_lodds <- -log(sigma_b)  # Compute the constant log-odds term for the prior on coefficients.
  s_p <- 1 / sqrt(half_diag + 1 / sigma_b^2)  # Compute the posterior standard deviation for each coefficient.
  s <- noise_sd * s_p  # Scale the posterior standard deviation by the noise standard deviation.
  count <- 0  # Initialize the iteration counter.

  # Iterative updates
  for (i in 1:max_iter) {
    for (j in update_order) {
      # Remove the contribution of the current coefficient from the approximate mean
      X_appm <- X_appm - approx_mean[j] * X[, j]
      
      # Update the posterior mean for the current coefficient
      mu[j] <- s_p[j]^2 * (YX_vec[j] - sum(X[, j] * X_appm))
      
      # Update the inclusion probability for the current coefficient
      gamma[j] <- sigmoid(
        logit_pi + 
          const_lodds + log(s_p[j]) +  # Log of posterior standard deviation, diffent from Ray Szabo
          0.5 * (mu[j] / (noise_sd * s_p[j]))^2  # Contribution from the squared posterior mean
      )
      
      # Update the approximate mean for the current coefficient
      approx_mean[j] <- gamma[j] * mu[j]
      
      # Add the updated contribution of the current coefficient to the approximate mean
      X_appm <- X_appm + approx_mean[j] * X[, j]
    }
    
    # Update the posterior standard deviation
    s <- noise_sd * s_p
    count <- count + 1  # Increment the iteration counter
    
    # Check for convergence using entropy
    new_entr <- entrop(gamma)
    if (max(abs(new_entr - old_entr)) <= tol) {
      break
    } else {
      old_entr <- new_entr
    }
  }
  
  # Return the results
  return(list(
    mu = mu,               # Posterior mean of the coefficients
    s = s,                 # Posterior standard deviation of the coefficients
    gamma = gamma,         # Posterior inclusion probabilities
    const_lodds = const_lodds, # Constant log-odds term
    iter = count,          # Number of iterations performed
    noise_sd = noise_sd    # Final noise standard deviation
  ))
}

# -------------------------------------------------------------------------

#' Bernoulli entropy (elementwise)
#'
#' @description
#' Computes $H(x) = -x\log(x) - (1-x)\log(1-x)$ for each element of `x` when
#' `x` is strictly between 0 and 1. Values extremely close to 0 or 1 are treated
#' as having entropy 0 for numerical stability.
#'
#' @param x Numeric vector of probabilities.
#'
#' @return Numeric vector of entropies (same length as `x`).
entrop <- function(x) {
  # Initialize an empty vector to store entropy values
  ent <- numeric(length(x))
  
  # Loop through each element in the input vector
  for (j in 1:length(x)) {
    # Compute entropy only for valid probabilities (0 < x < 1)
    if (x[j] > 1e-10 && x[j] < 1 - 1e-10) {
      ent[j] <- -x[j] * log(x[j]) - (1 - x[j]) * log(1 - x[j])
    }
  }
  
  # Return the computed entropy values
  return(ent)
}

# -------------------------------------------------------------------------

#' Numerically-stable sigmoid
#'
#' @description
#' Computes the logistic sigmoid function with hard cutoffs to avoid overflow.
#'
#' @param x Numeric scalar.
#'
#' @return Numeric scalar in [0, 1].
sigmoid <- function(x) {
  if (x > 32) {
    return(1)
  } else if (x < -32) {
    return(0)
  } else {
    return(1 / (1 + exp(-x)))
  }
}

# -------------------------------------------------------------------------

#' Diagonal of Gram matrix
#'
#' @description
#' Computes `diag(t(X) %*% X)` efficiently via columnwise sums of squares.
#'
#' @param X Numeric matrix (n x p).
#'
#' @return Numeric vector (length p) with sum of squares of each column of `X`.
gram_diag <- function(X) {
  # Initialize a vector to store the diagonal values
  diag_vals <- numeric(ncol(X))
  
  # Compute the sum of squares for each column of X
  for (i in 1:ncol(X)) {
    diag_vals[i] <- sum(X[, i]^2)
  }
  
  # Return the diagonal values
  return(diag_vals)
}

# -------------------------------------------------------------------------

#' Log-determinant
#'
#' @description
#' Returns `log(det(X))` using a numerically stable decomposition. If `X` is a
#' scalar (not a matrix), returns `log(X)`.
#'
#' @param X Numeric matrix or scalar.
#'
#' @return Numeric scalar (log-determinant).
ldet <- function(X) {
  # If X is not a matrix, return the log of X
  if (!is.matrix(X)) {
    return(log(X))
  }
  
  # Compute the log-determinant of the matrix
  result <- determinant(X, logarithm = TRUE)$modulus
  
  # Return the log-determinant
  return(result)
}
# -------------------------------------------------------------------------

#' Logit transform
#'
#' @description
#' Computes `log(x / (1 - x))`.
#'
#' @param x Numeric vector of probabilities.
#'
#' @return Numeric vector of logit-transformed values.
logit <- function(x) {
  # Compute the logit transformation
  result <- log(x / (1 - x))
  
  # Return the result
  return(result)
}
# -------------------------------------------------------------------------

#' Generate a 1D Matern covariance matrix
#'
#' @description
#' Generates a covariance matrix using a 1D grid of locations `1:dim` and the
#' Matern covariance function implemented in [Matern()].
#'
#' @param dim Integer. Dimension of the covariance matrix.
#' @param range Numeric. Range parameter.
#' @param smoothness Numeric. Smoothness parameter ($\nu$).
#' @param phi Numeric. Marginal variance.
#'
#' @return A symmetric `dim x dim` covariance matrix.
generate_covariance <- function(dim, range, smoothness, phi) {
  # Generate locations as integers from 1 to dim
  locs <- 1:dim
  
  # Compute the pairwise distance matrix between locations
  dist_matrix <- as.matrix(dist(locs))
  
  # Apply the Matern covariance function to the distance matrix
  if(range >=0){
  cov_matrix <- Matern(dist_matrix, range, smoothness, phi)
  } else {
    cov_matrix <- diag(dim) + matrix(1, nrow=dim, ncol=dim)
  }
  
  # Return the resulting covariance matrix
  return(cov_matrix)
}

# -------------------------------------------------------------------------

#' Matern covariance function
#'
#' @description
#' Computes the Matern covariance $\text{Cov}(d)$ for pairwise distances `d`.
#' This implementation includes special-cases for $\nu \in \{0.5, 1.5, 2.5\}$.
#'
#' @param d Numeric vector/matrix. Pairwise distances (nonnegative).
#' @param range Numeric. Range parameter.
#' @param smoothness Numeric. Smoothness parameter ($\nu$).
#' @param phi Numeric. Marginal variance.
#'
#' @return Numeric vector/matrix of covariances with the same shape as `d`.
Matern <- function(d, range = 1, smoothness = 0.5, phi = 1.0) {
  # default in fields package
  alpha = 1 / range 
  nu = smoothness
  # Check for negative distances
  if (any(d < 0)) {
    stop("Distance argument must be nonnegative")
  }
  
  # Rescale distances by the range parameter
  d <- d * alpha
  
  # Handle special cases for smoothness (nu)
  if (nu == 0.5) {
    return(phi * exp(-d))  # Exponential covariance
  }
  if (nu == 1.5) {
    return(phi * (1 + d) * exp(-d))  # Matern covariance for nu = 1.5
  }
  if (nu == 2.5) {
    return(phi * (1 + d + d^2 / 3) * exp(-d))  # Matern covariance for nu = 2.5
  }
  
  # General case: Use the Bessel function
  # Avoid exact zero distances to prevent issues with the Bessel function
  d[d == 0] <- 1e-10
  
  # Compute the constant factor for the Matern covariance
  con <- (2^(nu - 1)) * gamma(nu)  # Gamma function for nu
  con <- 1 / con
  
  # Compute the Matern covariance using the Bessel function
  matern_cov <- phi * con * (d^nu) * besselK(d, nu)
  
  # Return the covariance values
  return(matern_cov)
}
# -------------------------------------------------------------------------

#' Confusion matrix for LFDR-based classification
#'
#' @description
#' Computes two confusion matrices comparing:
#' 1) LFDR-based rejection indicators from [MTR()] versus truth;
#' 2) thresholding `estimated_post_gamma > 0.5` versus truth.
#'
#' @param estimated_post_gamma Numeric vector. Posterior probability of being a signal.
#' @param s_prob Numeric vector. True signal probabilities (currently unused).
#' @param selection Integer/numeric vector of 0/1. Ground truth signal indicators.
#' @param p Integer. Total number of hypotheses.
#' @param alpha_mtr Numeric. Target FDR level for MTR.
#' @param method Character. Label used in printed messages.
#'
#' @return A list with `confusion_table`, `confusion_table_lfdr`, and `threshold_used`.
confusion_matrix_lfdr <- function(estimated_post_gamma,
                                  s_prob,
                                  selection,
                                  p,
                                  alpha_mtr,
                                  method) {
  # Order hypotheses based on true probabilities
  
  # Identify true signal locations
  signal_location_list <- which(selection == 1)
  
  # Compute LFDR-based metrics using the MTR function
  mtr <- MTR( lfdr =  1 - estimated_post_gamma,
              # s_prob = s_prob,
             alpha = alpha_mtr, 
             signal = signal_location_list,
             method = method)

  # Equivalent to the prior magrittr pipeline: mtr$LFDR_full_res %>% unlist() %>% as.vector()
  LFDR_full <- as.vector(unlist(mtr$LFDR_full_res))
  
  # Initialize the confusion matrix
  confusion_table <- matrix(0, nrow = 2, ncol = 2)
  confusion_table_lfdr <- matrix(0, nrow = 2, ncol = 2)
  colnames(confusion_table) <- c("True Signal", "False Signal")
  rownames(confusion_table) <- c("Est. Signal", "Est. non-Signal")
  colnames(confusion_table_lfdr) <- c("True Signal", "False Signal")
  rownames(confusion_table_lfdr) <- c("Est. Signal", "Est. non-Signal")
  
  # Populate the confusion matrix
  for (i in 1:p) {
    row_index <- ifelse(LFDR_full[i] == 1, 1, 2)  # Estimated signal or non-signal
    col_index <- ifelse(selection[i] == 1, 1, 2)  # True signal or non-signal
    confusion_table_lfdr[row_index, col_index] <- confusion_table_lfdr[row_index, col_index] + 1
  }
  
  for (i in 1:p) {
    row_index <- ifelse(estimated_post_gamma[i] > 0.5, 1, 2)  # Estimated signal or non-signal
    col_index <- ifelse(selection[i] == 1, 1, 2)  # True signal or non-signal
    confusion_table[row_index, col_index] <- confusion_table[row_index, col_index] + 1
  }

  # Return the confusion matrix
  return(list(confusion_table = confusion_table,
              confusion_table_lfdr = confusion_table_lfdr,
              threshold_used = mtr$threshold_used))
}
# -------------------------------------------------------------------------

#' MTR thresholding for LFDR control
#'
#' @description
#' Implements an LFDR threshold selection rule that chooses the largest LFDR value
#' such that the cumulative mean LFDR among rejections is below `alpha`.
#'
#' @param lfdr Numeric vector. Local FDR values.
#' @param alpha Numeric scalar. Target level.
#' @param signal Optional integer vector. Indices of true signals (for metrics).
#' @param method Character. Label used in printed messages.
#'
#' @return A list with summary statistics, full rejection indicators, and the threshold used.
MTR <- function(lfdr, alpha, signal = NULL, method) {
  # Total number of hypotheses
  M <- length(lfdr)
  
  # Replace NA values in LFDR with 1
  lfdr[is.na(lfdr)] <- 1
  
  # Determine the LFDR threshold
  threshold <- 0
  if (min(lfdr) < alpha) {
    # explore why threshold is too small
    threshold <- max(sort(lfdr)[cumsum(sort(lfdr))/(1:M) < alpha])
    # cummean <-round(cumsum(sort(lfdr))/(1:M),2)
    # print("LFDR values (first 10):", cummean[1:10] )
  print(paste0("Method ", method, ", LFDR threshold: ", round(threshold, 4)))
  } else {
    threshold <- 0.5
    print(paste0("!! Method ", method, ", LFDR threshold set to 0.5 due to high initial cumulative LFDR (> alpha_mtr) ", 
                 round(min(lfdr), 3)))
  }
  
  # Identify rejections based on the LFDR threshold
  LFDR_full_res <- data.frame(LFDR = 1 * I(lfdr <= threshold))
  
  # Initialize metrics
  LFDR_sum_res <- NULL
  TPR <- NULL
  TNR <- NULL
  RAND <- NULL
  
  # Compute metrics if true signal locations are provided
  if (!is.null(signal)) {
    M1 <- length(signal)  # Number of true signals
    M0 <- M - M1          # Number of null hypotheses
    n_signal <- !(1:M %in% signal)
    
    # Summarize results
    LFDR_sum_res <- c(
      sum(LFDR_full_res$LFDR),               # Total number of rejections (R)
      sum(LFDR_full_res$LFDR[signal]),       # Total number of true positives (TP)
      sum(LFDR_full_res$LFDR[n_signal])      # Total number of false positives (FP)
    )
    
    # Compute TPR, TNR, and RAND Index
    TPR <- LFDR_sum_res[2] / M1
    TNR <- (M0 - LFDR_sum_res[3]) / M0
    RAND <- (M0 - LFDR_sum_res[3] + LFDR_sum_res[2]) / M
  }
  
  # Return the results as a list
  return(list(
    LFDR_sum_res = LFDR_sum_res,
    LFDR_full_res = LFDR_full_res,
    M1 = M1,
    TPR = TPR,
    TNR = TNR,
    RAND = RAND,
    threshold_used = threshold
  ))
}
# -------------------------------------------------------------------------

#' Assign sign-pattern groups
#'
#' @description
#' Maps a numeric row vector to a pattern string of "P"/"N" corresponding to
#' positive/non-positive entries.
#'
#' @param row Numeric vector.
#'
#' @return Character scalar pattern (e.g., "PNP").
assign_group_general <- function(row) {
  # Convert each value in the row to "P" (if positive) or "N" (if negative)
  pattern <- paste0(ifelse(row > 0, "P", "N"), collapse = "")
  
  # Return the resulting pattern as a string
  return(pattern)
}

#' Generalized logistic function
#'
#' @description
#' A flexible logistic link that reduces to the standard logistic when
#' `k = 1`, `c = 0`, and `v = 1`.
#'
#' @param x Numeric vector.
#' @param k Numeric. Steepness.
#' @param c Numeric. Location (midpoint).
#' @param v Numeric. Shape.
#'
#' @return Numeric vector in (0, 1).
generalized_logistic <- function(x, k = 1, c = 0, v = 1) {
  1 / (1 + exp(-k * (x - c)))^v
}
