# Stroop Task Modeling Script
# linear and generalized linear mixed-effects models (LMM/GLMM).
# It models Reaction Time (RT) and Accuracy as a function of stimulation
# condition, trial congruency, and other covariates.

# --- 1. Load Libraries ---
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(jsonlite)
library(ggplot2)

# --- 2a. Plotting Helpers ---

# Extract fixed effects with Wald CIs and return a tidy data.frame
get_fixed_effects_df <- function(model) {
  fe <- lme4::fixef(model)
  fe_names <- names(fe)
  # Wald CIs are fast and adequate for quick visualization
  ci <- suppressMessages(suppressWarnings(confint(model, method = "Wald")))
  ci_fe <- ci[rownames(ci) %in% fe_names, , drop = FALSE]
  # Ensure row order matches fe_names
  ci_fe <- ci_fe[fe_names, , drop = FALSE]
  df <- data.frame(
    term = fe_names,
    estimate = as.numeric(fe[fe_names]),
    conf.low = as.numeric(ci_fe[, 1]),
    conf.high = as.numeric(ci_fe[, 2]),
    stringsAsFactors = FALSE
  )
  df$term <- factor(df$term, levels = rev(df$term))
  df
}

# Create and save a forest plot for fixed effects
save_coef_forest <- function(model, title, out_file) {
  df <- get_fixed_effects_df(model)
  p <- ggplot(df, aes(x = estimate, y = term)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0) +
    geom_point(size = 2) +
    labs(title = title, x = "Estimate (Wald 95% CI)", y = NULL)
  ggsave(out_file, p, width = 8, height = 6)
}

# Save basic diagnostics for LMM: residuals vs fitted and QQ plot
save_lmm_diagnostics <- function(model, out_file_prefix) {
  df <- data.frame(fitted = fitted(model), resid = residuals(model))
  p1 <- ggplot(df, aes(x = fitted, y = resid)) +
    geom_point(alpha = 0.3) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = "Residuals vs Fitted", x = "Fitted", y = "Residuals")
  qq <- qqnorm(residuals(model), plot.it = FALSE)
  dfqq <- data.frame(theoretical = qq$x, sample = qq$y)
  p2 <- ggplot(dfqq, aes(x = theoretical, y = sample)) +
    geom_point(alpha = 0.3) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = "Normal Q-Q")
  pdf(paste0(out_file_prefix, "_Diagnostics.pdf"), width = 8, height = 6)
  print(p1)
  print(p2)
  dev.off()
}

# Save basic diagnostics for GLMM: predicted prob hist by class and calibration
save_glmm_diagnostics <- function(model, data, response_col, out_file_prefix) {
  prob <- as.numeric(predict(model, type = "response"))
  dfp <- data.frame(prob = prob, actual = data[[response_col]])
  # Ensure factor with levels 0/1 for display
  dfp$actual <- factor(as.character(dfp$actual), levels = c("0", "1"))
  p1 <- ggplot(dfp, aes(x = prob, fill = actual)) +
    geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
    labs(title = "Predicted Probabilities by Class", x = "Predicted P(correct=1)", y = "Count", fill = "Actual")
  # Calibration by deciles
  dfp$bin <- cut(dfp$prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE, right = TRUE)
  cal <- suppressMessages(dfp %>% dplyr::group_by(bin) %>% dplyr::summarise(
    pred = mean(prob, na.rm = TRUE),
    obs = mean(as.numeric(as.character(actual)), na.rm = TRUE),
    .groups = "drop"
  ))
  p2 <- ggplot(cal, aes(x = pred, y = obs)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = "Calibration (Deciles)", x = "Mean predicted", y = "Observed frequency") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1))
  pdf(paste0(out_file_prefix, "_Diagnostics.pdf"), width = 8, height = 6)
  print(p1)
  print(p2)
  dev.off()
}

# --- 2. Set Global Parameters ---
DATA_DIR <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/analyzed-data/"
OUTPUT_DIR <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-taVNS-analysis/analyzed-data"

# Create output directory for today's date
today <- format(Sys.Date(), "%Y%m%d")
output_path <- file.path(OUTPUT_DIR, today)
if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = TRUE)
}

# Set a global theme
theme_set(
  theme_minimal() +
    theme(
      axis.line = element_line(linewidth = 1),
      axis.text = element_text(size = 12),
      axis.title = element_text(size = 14),
      axis.ticks.length = unit(0.25, "cm"),
      axis.ticks = element_line(linewidth = 1),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top"
    )
)
# update_geom_defaults("point", list(size = 3, shape = 3, stroke = 1))

# --- 3. Data Loading Function ---

#' Load and combine all stroopTrials.csv files for a given experiment type.
#'
#' @param data_dir Top-level directory containing subject folders.
#' @param start_date The earliest session date to include (YYYYMMDD).
#' @param end_date The latest session date to include (YYYYMMDD).
#' @param experiment_type The name of the experiment to load (e.g., "SCWT").
#' @return A single dataframe containing all trial data for the experiment.
load_stroop_data <- function(data_dir, experiment_type) {
  # find most recent date folder with {experiment_type}_Trials.csv
  analysis_dirs <- list.dirs(data_dir, recursive = TRUE, full.names = TRUE) %>% sort(decreasing = TRUE)

  for (dir in analysis_dirs) {
    date_folder <- basename(dir)
    if (!grepl("^\\d{8}$", date_folder)) next

    csv_file1 <- file.path(dir, paste0(experiment_type, "_Trials.csv"))
    csv_file2 <- file.path(dir, paste0("Stroop_", experiment_type, "_Trials.csv"))
    chosen_file <- if (file.exists(csv_file1)) csv_file1 else if (file.exists(csv_file2)) csv_file2 else NA
    if (!is.na(chosen_file)) {
      trial_df <- read.csv(chosen_file, stringsAsFactors = FALSE)
      print(paste("Loaded data from:", chosen_file, "with", nrow(trial_df), "rows."))
      return(trial_df)
    }
  }
  return(data.frame())  # Return an empty data frame if no data found
}


# --- 5. Core Modeling Function ---

#' Run LMM/GLMM analysis for a given experiment.
#'
#' @param experiment_type The name of the experiment (e.g., "SCWT").
#' @param report_file_path Path to sink the model summaries.
run_stroop_analysis <- function(experiment_type, report_file_path) {
  
  # # Load and prepare data
  trial_data <- load_stroop_data(DATA_DIR, experiment_type)
  if (nrow(trial_data) == 0) {
    cat(paste("\nNo data found for experiment:", experiment_type, "\n"), file = report_file_path, append = TRUE)
    return()
  }

  trial_data <- trial_data %>%
    mutate(
      subject = as.factor(subject),
      condition = as.factor(condition),
      congruent = as.factor(congruent),
      # Robustly normalize order encodings to STTS/TSST
      order_raw = tolower(as.character(order)),
      order = case_when(
        order_raw %in% c("0", "stts") ~ "STTS",
        order_raw %in% c("1", "tsst") ~ "TSST",
        grepl("stts", order_raw) ~ "STTS",
        grepl("tsst", order_raw) ~ "TSST",
        TRUE ~ NA_character_
      ),
      order = factor(order, levels = c("STTS", "TSST")),
      block = factor(as.character(block), ordered = TRUE, levels = c("1", "2", "3", "4")),
      rt = as.numeric(rt),
      correct = factor(correct, levels = c(0, 1), labels = c("0", "1")),
      trial_number = as.integer(trial_number),
      post_correct = factor(prev_correct, levels = c(0, 1), labels = c("0", "1"))
    ) %>%
    select(-order_raw)
  str(trial_data)

  # --- RT Model (LMM on correct trials only) ---
  data_correct <- trial_data %>% filter(.data$correct == "1")
  print(paste("Modeling RT on", nrow(data_correct), "correct trials."))
  # condition by trial no, condition * congruency, condition * correct, condition*order, condition*block, amplitude*condition, rating*condition
  rt_model <- lmer(
    rt ~  condition * congruent + 
           condition * block + 
           congruent * block +
           condition * order +
           block * order +
           (1 | subject),
    data = data_correct)
  
  # --- Accuracy Model (GLMM on all trials) ---
  acc_model <- glmer(
    correct ~ condition * congruent + 
           condition * block + 
           congruent * block +
           condition * order +
           block * order +
           (1 | subject),
    data = trial_data,
    family = binomial
  )



  # --- Write results to report file ---
  sink(report_file_path, append = TRUE)
  
  cat(paste(rep("-", 80), collapse = ""))
  cat(paste("\nExperiment:", experiment_type, "\n"))
  cat(paste(rep("-", 80), collapse = ""))
  
  cat("\n\n--- Reaction Time Model (log(RT) on Correct Trials) ---\n\n")
  print(summary(rt_model))
  # Basic LMM metrics
  rt_resid <- residuals(rt_model)
  rt_fit <- fitted(rt_model)
  rt_rmse <- sqrt(mean(rt_resid^2, na.rm = TRUE))
  rt_mae <- mean(abs(rt_resid), na.rm = TRUE)
  rt_r2 <- suppressWarnings(cor(data_correct$rt, rt_fit, use = "complete.obs")^2)
  cat(sprintf("\nRT model metrics: RMSE=%.3f, MAE=%.3f, pseudo-R2=%.3f\n", rt_rmse, rt_mae, rt_r2))
  
  cat("\n\n--- Accuracy Model (Correctness) ---\n\n")
  print(summary(acc_model))
  # Basic GLMM metrics
  acc_prob <- as.numeric(predict(acc_model, type = "response"))
  acc_pred <- ifelse(acc_prob >= 0.5, "1", "0")
  acc_truth <- as.character(trial_data$correct)
  acc_rate <- mean(acc_pred == acc_truth, na.rm = TRUE)
  cat(sprintf("\nAccuracy model metrics: AIC=%.1f, Accuracy@0.5=%.3f\n", AIC(acc_model), acc_rate))

  cat("\n\n\n")
  sink()

  # --- Save plots ---
  rt_coef_file <- file.path(output_path, paste0("Stroop_", experiment_type, "_RT_Coefficients.pdf"))
  save_coef_forest(rt_model, paste0("RT Fixed Effects (", experiment_type, ")"), rt_coef_file)
  save_lmm_diagnostics(rt_model, file.path(output_path, paste0("Stroop_", experiment_type, "_RT")))

  acc_coef_file <- file.path(output_path, paste0("Stroop_", experiment_type, "_Accuracy_Coefficients.pdf"))
  save_coef_forest(acc_model, paste0("Accuracy Fixed Effects (", experiment_type, ")"), acc_coef_file)
  save_glmm_diagnostics(acc_model, trial_data, "correct", file.path(output_path, paste0("Stroop_", experiment_type, "_Accuracy")))
}


# --- 6. Main Execution ---

# Define the report file path
report_file <- file.path(output_path, "Stroop_Modeling_Report.txt")

# Clear the report file if it already exists
if (file.exists(report_file)) {
  file.remove(report_file)
}

# Define the experiments to analyze
experiment_types <- c("SCWT", "StroopSquared")

# Run the analysis for each experiment
for (exp_type in experiment_types) {
  tryCatch({
    run_stroop_analysis(exp_type, report_file)
  }, error = function(e) {
    # Log errors to the report file
    sink(report_file, append = TRUE)
    cat(paste("\n\nERROR processing", exp_type, ":", e$message, "\n\n"))
    sink()
  })
}

print(paste("Analysis complete. Report saved to:", report_file))