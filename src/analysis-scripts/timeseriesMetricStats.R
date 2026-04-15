# Stroop Stats by Contrast
# Computes incongruent − congruent contrasts per block and runs paired t-tests
# and Wilcoxon signed-rank tests on metrics and contrasts by condition.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
})

DATA_DIR <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/analyzed-data"
    # Save per-transition means (aggregated across subjects), mirroring metrics/contrasts
# trans_condition_summary <- trans_means %>% dplyr::group_by(metric, condition) %>% dplyr::summarise(
#       n_subjects = dplyr::n_distinct(subject),
#       mean = mean(mean_value, na.rm = TRUE),
#       sem = safe_sem(mean_value),
#       .groups = "drop"
#     )
#     readr::write_csv(trans_condition_summary, file.path(day_dir, sprintf("%s_PairedMeans_Transitions.csv", exp)))
EXPERIMENTS <- c("SCWT")
CONDITION_COLORS <- c(sham = "#6dc8bf", taVNS = "#f15a22")
# Set subject IDs to exclude from analysis/plots (e.g., c("S001"))
EXCLUDE_SUBJECTS <- c("")

# Optional per-experiment manual exclusions (override only for specific experiments)
# Example: EXCLUDE_SUBJECTS_BY_EXPERIMENT$StroopSquared <- c("S012")
EXCLUDE_SUBJECTS_BY_EXPERIMENT <- list(
  SCWT = character(0),
  StroopSquared = c("ERC22")
)

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
update_geom_defaults("point", list(size = 1, shape = 3, stroke = 0.5))

# Find most recent dated folder containing summary CSVs
find_latest_day_dir <- function(root) {
  dirs <- list.dirs(root, recursive = FALSE, full.names = TRUE)
  # Filter to directories whose basename is exactly 8 digits (YYYYMMDD)
  dated <- dirs[grepl("^\\d{8}$", basename(dirs))]
  if (!length(dated)) return(NA_character_)
  dates <- basename(dated)
  dated[order(dates, decreasing = TRUE)][1]
}

# Safely compute SEM
safe_sem <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x) / sqrt(length(x))
}

# Load all timeseries metrics CSVs from latest day directory
load_timeseries_metrics <- function(day_dir) {
  files <- list.files(day_dir, pattern = "timeseries_metrics\\.csv$", recursive = TRUE, full.names = TRUE)
  if (!length(files)) return(tibble())
  dfs <- lapply(files, function(f) {
    tryCatch(readr::read_csv(f, show_col_types = FALSE), error = function(e) tibble())
  })
  dfs <- Filter(function(x) nrow(x) > 0, dfs)
  if (!length(dfs)) return(tibble())
  dplyr::bind_rows(dfs)
}

# Compute subject-level means per condition for timeseries metrics
subject_condition_means_for_timeseries <- function(metrics_df) {
  if (nrow(metrics_df) == 0) return(tibble())
  metrics_df %>%
    dplyr::filter(!is.na(.data$condition), !is.na(.data$id)) %>%
    dplyr::group_by(.data$expt, .data$event, .data$signal, .data$metric, .data$id, .data$condition) %>%
    dplyr::summarise(mean_value = mean(.data$value, na.rm = TRUE), .groups = "drop")
}

# Paired tests (t-test and Wilcoxon) for sham vs taVNS within subjects
run_paired_tests <- function(means_df) {
  conds <- sort(unique(as.character(means_df$condition)))
  if (length(conds) != 2) return(tibble())
  if (!all(c("sham", "taVNS") %in% conds)) return(tibble())
  wide <- means_df %>% tidyr::pivot_wider(names_from = condition, values_from = mean_value)
  if (!all(c("sham", "taVNS") %in% names(wide))) return(tibble())
  wide <- wide %>% dplyr::filter(is.finite(.data$sham), is.finite(.data$taVNS))
  results <- wide %>% dplyr::group_by(.data$expt, .data$event, .data$signal, .data$metric) %>% dplyr::group_split()
  out <- lapply(results, function(dfm) {
    x <- dfm$sham
    y <- dfm$taVNS
    if (length(x) < 2 || length(y) < 2) return(NULL)
    d <- y - x
    sh <- tryCatch(stats::shapiro.test(d), error = function(e) NULL)
    t_res <- tryCatch(stats::t.test(y, x, paired = TRUE), error = function(e) NULL)
    w_res <- tryCatch(stats::wilcox.test(y, x, paired = TRUE, exact = FALSE), error = function(e) NULL)
    ci_low <- if (!is.null(t_res) && !is.null(t_res$conf.int)) as.numeric(t_res$conf.int[1]) else NA_real_
    ci_high <- if (!is.null(t_res) && !is.null(t_res$conf.int)) as.numeric(t_res$conf.int[2]) else NA_real_
    tibble(
      expt = dfm$expt[1],
      event = dfm$event[1],
      signal = dfm$signal[1],
      metric = dfm$metric[1],
      n = length(d),
      normal = if (!is.null(sh)) as.numeric(sh$p.value) else NA_real_,
      t_p = if (!is.null(t_res)) as.numeric(t_res$p.value) else NA_real_,
      w_p = if (!is.null(w_res)) as.numeric(w_res$p.value) else NA_real_,
      mean_diff = if (!is.null(t_res)) as.numeric(t_res$estimate) else NA_real_,
      ci_low = ci_low,
      ci_high = ci_high
    )
  })
  out <- Filter(Negate(is.null), out)
  if (!length(out)) return(tibble())
  dplyr::bind_rows(out)
}

# Plot helper: paired subject lines and condition means with SEM per expt/event/signal
save_paired_plots_pdf <- function(means_df, tests_df, pdf_path) {
  if (nrow(means_df) == 0) return(invisible(NULL))
  df <- means_df %>% dplyr::mutate(condition = factor(condition, levels = c("sham", "taVNS")))
  metrics <- unique(df$metric)
  grDevices::pdf(pdf_path, width = 11/3, height = 8.5/2)
  on.exit(grDevices::dev.off(), add = TRUE)
  for (m in metrics) {
    df_m <- df %>% dplyr::filter(.data$metric == m)
    cond_summary <- df_m %>% dplyr::group_by(.data$condition) %>% dplyr::summarise(
      mean = mean(.data$mean_value, na.rm = TRUE),
      sem = safe_sem(.data$mean_value),
      .groups = "drop"
    )
    # Pick significance label based on normality
    sig_label <- NULL
    if (!is.null(tests_df) && nrow(tests_df)) {
      row <- tests_df %>% dplyr::filter(.data$metric == m) %>% dplyr::slice(1)
      if (nrow(row)) {
        use_wilcox <- is.finite(row$normal) && row$normal < 0.05
        pval <- if (use_wilcox) row$w_p else row$t_p
        if (is.finite(pval)) {
          sig_label <- if (pval < 0.001) "***" else if (pval < 0.01) "**" else if (pval < 0.05) "*" else "ns"
          sig_label <- paste0("p = ", signif(pval, 3), " ", sig_label)
        }
      }
    }
    p <- ggplot2::ggplot(df_m, ggplot2::aes(x = condition, y = mean_value, group = id, color = condition)) +
      ggplot2::geom_line(alpha = 0.25, position = ggplot2::position_dodge(width = 0.15), color = "grey40") +
      ggplot2::geom_point(alpha = 0.8, position = ggplot2::position_dodge(width = 0.15)) +
      ggplot2::geom_point(data = cond_summary, ggplot2::aes(x = condition, y = mean), inherit.aes = FALSE) +
      ggplot2::geom_errorbar(data = cond_summary,
        ggplot2::aes(x = condition, ymin = mean - sem, ymax = mean + sem),
        inherit.aes = FALSE, width = 0.15, linewidth = 1
      ) +
      ggplot2::scale_color_manual(values = CONDITION_COLORS) +
      ggplot2::labs(y = m, x = NULL)
    if (!is.null(sig_label)) {
      y_data_max <- suppressWarnings(max(c(df_m$mean_value, cond_summary$mean + cond_summary$sem), na.rm = TRUE))
      if (is.finite(y_data_max)) {
        p <- p + ggplot2::annotate("text", x = 1.5, y = y_data_max * 1.025, label = sig_label)
      }
    }
    print(p)
  }
}

main <- function() {
  day_dir <- find_latest_day_dir(DATA_DIR)
  if (is.na(day_dir)) stop("No dated output directory found in ", DATA_DIR)
  message("Using outputs from ", day_dir)
  df <- load_timeseries_metrics(day_dir)
  if (nrow(df) == 0) stop("No timeseries_metrics.csv found under ", day_dir)

  # Apply subject exclusions
  df <- df %>% dplyr::filter(!.data$id %in% EXCLUDE_SUBJECTS)

  out_dir <- file.path(day_dir, "TimeseriesMetrics")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  # Loop by experiment/event/signal, compute means and paired tests, save plots and CSVs
  for (exp in unique(df$expt)) {
    for (ev in unique(df %>% dplyr::filter(.data$expt == exp) %>% dplyr::pull(.data$event))) {
      for (sig in unique(df %>% dplyr::filter(.data$expt == exp, .data$event == ev) %>% dplyr::pull(.data$signal))) {
        sub <- df %>% dplyr::filter(.data$expt == exp, .data$event == ev, .data$signal == sig)
        means <- subject_condition_means_for_timeseries(sub)
        tests <- run_paired_tests(means)

        # Save condition summary and tests
        cond_summary <- means %>% dplyr::group_by(.data$metric, .data$condition) %>% dplyr::summarise(
          n_subjects = dplyr::n_distinct(.data$id),
          mean = mean(.data$mean_value, na.rm = TRUE),
          sem = safe_sem(.data$mean_value),
          .groups = "drop"
        )
        readr::write_csv(cond_summary, file.path(out_dir, sprintf("%s_%s_%s_Metrics_PairedMeans.csv", exp, ev, sig)))
        if (nrow(tests)) readr::write_csv(tests, file.path(out_dir, sprintf("%s_%s_%s_Metrics_PairedTests.csv", exp, ev, sig)))

        # Save paired plots PDF
        pdf_path <- file.path(out_dir, sprintf("%s_%s_%s_Metrics_PairedPlots.pdf", exp, ev, sig))
        save_paired_plots_pdf(means, tests, pdf_path)
      }
    }
  }
}

main()
