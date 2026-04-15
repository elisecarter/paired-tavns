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
EXPERIMENTS <- c("SCWT", "StroopSquared")
CONDITION_COLORS <- c(sham = "#E3671C", taVNS = "#6666ff")
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

# Transition-labeled deltas: T->S vs S->T based on order
compute_transition_deltas_metrics <- function(block_summary) {
  df <- block_summary %>%
    dplyr::filter(.data$congruent == "incongruent") %>%
    dplyr::mutate(order = factor(.data$order, levels = c("STTS", "TSST")))
  metrics <- c("block_rt_mean", "block_acc_mean", "block_ies_mean", "block_score")
  out <- lapply(metrics, function(metric) {
    wide <- df %>%
      dplyr::select(subject, order, block, !!rlang::sym(metric)) %>%
      tidyr::pivot_wider(names_from = block, values_from = !!rlang::sym(metric))
    if (!all(c("1", "2", "3", "4") %in% names(wide))) return(NULL)
    # Adjacent deltas
    d12 <- wide[["2"]] - wide[["1"]]
    d34 <- wide[["4"]] - wide[["3"]]
    # Assign transitions by order
    # STTS: 2-1 is S->T; 4-3 is T->S
    # TSST: 2-1 is T->S; 4-3 is S->T
    t_to_s <- ifelse(wide$order == "TSST", d12, d34)
    s_to_t <- ifelse(wide$order == "TSST", d34, d12)
    tibble(
      metric = dplyr::recode(metric,
        block_rt_mean  = "RT (s)",
        block_acc_mean = "Accuracy (%)",
        block_ies_mean = "IES (s)",
        block_score    = "Score"
      ),
      subject = wide$subject,
      order = wide$order,
      delta_T_to_S = t_to_s,
      delta_S_to_T = s_to_t
    )
  })
  out <- Filter(Negate(is.null), out)
  if (!length(out)) return(tibble())
  dplyr::bind_rows(out)
}

# Prepare per-subject means for transitions so we can reuse paired tests/plots
subject_transition_means_for_metrics <- function(trans_deltas) {
  if (nrow(trans_deltas) == 0) return(tibble())
  trans_long <- trans_deltas %>%
    tidyr::pivot_longer(
      cols = c(delta_T_to_S, delta_S_to_T),
      names_to = "condition",
      values_to = "value"
    ) %>%
    dplyr::mutate(
      condition = dplyr::recode(condition,
        delta_T_to_S = "T->S",
        delta_S_to_T = "S->T"
      ),
      # Match factor levels ordering used elsewhere
      condition = factor(condition, levels = c("T->S", "S->T")),
      subject = as.character(subject)
    )
  trans_long %>%
    dplyr::group_by(metric, subject, condition) %>%
    dplyr::summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop")
}

# Transition-labeled deltas based on percent-change contrast metrics per block
compute_transition_deltas_contrasts <- function(diffs) {
  if (nrow(diffs) == 0) return(tibble())
  # Average percent-change contrasts across experimental condition per subject/block/metric
  agg <- diffs %>%
    dplyr::filter(!is.na(.data$block)) %>%
    dplyr::group_by(.data$subject, .data$order, .data$block, .data$metric) %>%
    dplyr::summarise(pct_change = mean(.data$difference, na.rm = TRUE), .groups = "drop")
  out <- lapply(unique(agg$metric), function(m) {
    wide <- agg %>% dplyr::filter(.data$metric == m) %>%
      tidyr::pivot_wider(names_from = .data$block, values_from = .data$pct_change)
    if (!all(c("1", "2", "3", "4") %in% names(wide))) return(NULL)
    d12 <- wide[["2"]] - wide[["1"]]
    d34 <- wide[["4"]] - wide[["3"]]
    t_to_s <- ifelse(wide$order == "TSST", d12, d34)
    s_to_t <- ifelse(wide$order == "TSST", d34, d12)
    tibble(
      metric = m, # already labeled as Percent Change ... (%)
      subject = wide$subject,
      order = wide$order,
      delta_T_to_S = t_to_s,
      delta_S_to_T = s_to_t
    )
  })
  out <- Filter(Negate(is.null), out)
  if (!length(out)) return(tibble())
  dplyr::bind_rows(out)
}

# Plot transitions per subject: x-axis has S->T and T->S, grouped by subject
save_transition_plots_pdf <- function(trans_means, title, pdf_path, tests_df = NULL) {
  if (nrow(trans_means) == 0) return(invisible(NULL))
  df <- trans_means %>%
    dplyr::mutate(condition = factor(condition, levels = c("S->T", "T->S")))
  metrics <- unique(df$metric)
  grDevices::pdf(pdf_path, width = 11/3, height = 8.5/2)
  on.exit(grDevices::dev.off(), add = TRUE)
  for (m in metrics) {
    df_m <- df %>% dplyr::filter(.data$metric == m)
    # Compute condition means and SEM for overlay
    cond_summary <- df_m %>% dplyr::group_by(condition) %>% dplyr::summarise(
      mean = mean(mean_value, na.rm = TRUE),
      sem = safe_sem(mean_value),
      .groups = "drop"
    )
    # Determine p-value label based on normality using tests_df
    sig_label <- NULL
    if (!is.null(tests_df) && nrow(tests_df)) {
      row <- tests_df %>% dplyr::filter(.data$metric == m) %>% dplyr::slice(1)
      if (nrow(row)) {
        test <- if (!is.na(row$shapiro_p) && row$shapiro_p >= 0.05) "paired t-test" else "wilcoxon"
        pval <- if (!is.na(row$normality) && row$normality == "not-rejected") row$t_p else row$w_p
        if (!is.na(pval)) sig_label <- paste0("p=", round(pval, 3), " (", test, ")")
      }
    }
    p <- ggplot(df_m, aes(x = condition, y = mean_value, group = subject)) +
      geom_line(alpha = 0.25, color = "grey40", position = position_dodge(width = 0.15)) +
      geom_point(alpha = 0.8, position = position_dodge(width = 0.15)) +
      geom_point(data = cond_summary, aes(x = condition, y = mean), inherit.aes = FALSE) +
      geom_errorbar(data = cond_summary,
                    aes(x = condition, ymin = mean - sem, ymax = mean + sem),
                    inherit.aes = FALSE, width = 0.15, linewidth = 1) +
      labs(title = paste0(title, " – ", m), x = "Transition", y = m)
    if (!is.null(sig_label)) {
      y_data_max <- suppressWarnings(max(c(df_m$mean_value, cond_summary$mean + cond_summary$sem), na.rm = TRUE))
      if (is.finite(y_data_max)) {
        p <- p +
          coord_cartesian(clip = "off") +
          theme(plot.margin = margin(t = 16, r = 8, b = 8, l = 8)) +
          annotate("text", x = 1.5, y = y_data_max, vjust = -0.6, label = sig_label, size = 5)
      }
    }
    print(p)
  }
}

# Find most recent dated folder containing summary CSVs
find_latest_day_dir <- function(root) {
  dirs <- list.dirs(root, recursive = FALSE, full.names = TRUE)
  # Filter to directories whose basename is exactly 8 digits (YYYYMMDD)
  dated <- dirs[grepl("^\\d{8}$", basename(dirs))]
  if (!length(dated)) return(NA_character_)
  dates <- basename(dated)
  dated[order(dates, decreasing = TRUE)][1]
}

# Robustly load block summary for an experiment (supports two naming patterns)
load_block_summary <- function(day_dir, experiment) {
  candidates <- c(
    file.path(day_dir, sprintf("%s_BlockSummary.csv", experiment)),
    file.path(day_dir, sprintf("Stroop_%s_BlockSummary.csv", experiment))
  )
  file <- candidates[file.exists(candidates)][1]
  if (is.na(file)) return(tibble())
  readr::read_csv(file, show_col_types = FALSE)
}

# Compute within-block Stroop effect (incongruent − congruent) per metric
compute_metric_differences <- function(block_summary) {
  if (nrow(block_summary) == 0) return(tibble())
  metrics <- c("block_rt_mean", "block_acc_mean", "block_ies_mean")
  wide <- block_summary %>%
    select(subject, order, condition, block, congruent, dplyr::all_of(metrics)) %>%
    tidyr::pivot_wider(names_from = congruent, values_from = dplyr::all_of(metrics), names_sep = "__")
  map_metric <- function(metric) {
    cong <- paste0(metric, "__congruent")
    incong <- paste0(metric, "__incongruent")
    if (!all(c(cong, incong) %in% names(wide))) return(NULL)
    base <- wide[[cong]]
    inc  <- wide[[incong]]
    pct_change <- ifelse(is.finite(base) & base != 0, (inc - base) / base * 100, NA_real_)
    tibble(
      subject   = wide$subject,
      order     = wide$order,
      condition = wide$condition,
      block     = wide$block,
      metric    = dplyr::recode(metric,
            block_rt_mean  = "Percent Change RT (%)",
            block_acc_mean = "Percent Change Accuracy (%)",
            block_ies_mean = "Percent Change IES (%)"),
      difference = pct_change
    ) %>% dplyr::filter(!is.na(difference))
  }
  rows <- lapply(metrics, map_metric)
  rows <- Filter(Negate(is.null), rows)
  if (!length(rows)) return(tibble())
  dplyr::bind_rows(rows)
}

safe_sem <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x) / sqrt(length(x))
}

# Subject-level means per condition for block metrics
subject_condition_means_for_metrics <- function(block_summary) {
  metrics <- c("block_rt_mean", "block_acc_mean", "block_ies_mean", "block_score")
  long <- block_summary %>%
    filter(congruent == "incongruent") %>%
    tidyr::pivot_longer(cols = dplyr::all_of(metrics), names_to = "metric", values_to = "value") %>%
    dplyr::mutate(metric = dplyr::recode(metric,
      block_rt_mean  = "RT (s)",
      block_acc_mean = "Accuracy (%)",
      block_ies_mean = "IES (s)",
      block_score    = "Score"
    ))
  long %>%
    dplyr::group_by(metric, subject, condition) %>%
    dplyr::summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop")
}

# Subject-level means per condition for contrasts (incongruent − congruent)
subject_condition_means_for_contrasts <- function(diffs) {
  if (nrow(diffs) == 0) return(tibble())
  diffs %>%
    dplyr::group_by(metric, subject, condition) %>%
    dplyr::summarise(mean_value = mean(difference, na.rm = TRUE), .groups = "drop")
}

# Plot helper: paired subject lines and condition means with SEM
save_paired_plots_pdf <- function(means_df, title, pdf_path, tests_df = NULL) {
  if (nrow(means_df) == 0) return(invisible(NULL))
  df <- means_df %>% mutate(condition = factor(condition, levels = c("sham", "taVNS")))
  metrics <- unique(df$metric)
  grDevices::pdf(pdf_path, width =11/3, height = 8.5/2)
  on.exit(grDevices::dev.off(), add = TRUE)
  for (m in metrics) {
    df_m <- df %>% dplyr::filter(.data$metric == m)
    cond_summary <- df_m %>% group_by(condition) %>% summarise(
      mean = mean(mean_value, na.rm = TRUE),
      sem = safe_sem(mean_value),
      .groups = "drop"
    )
    # Determine significance star and bar based on normality and appropriate p-value
    sig_label <- NULL
    if (!is.null(tests_df) && nrow(tests_df)) {
      row <- tests_df %>% dplyr::filter(.data$metric == m) %>% dplyr::slice(1)
      if (nrow(row)) {
        if (row$normality == "not-rejected") {
          # if normality not rejects, use test that produces a lower p-value
          if (row$t_p <= row$w_p) {
            test <- "paired t-test"
            pval <- row$t_p
          } else {
            test <- "wilcoxon"
            pval <- row$w_p
          }
        } else {
          test <- "wilcoxon"
          pval <- row$w_p
        }
        test <- if (!is.na(row$shapiro_p) && row$shapiro_p >= 0.05) "paired t-test" else "wilcoxon"
        pval <- if (!is.na(row$normality) && row$normality == "not-rejected") row$t_p else row$w_p
        if (!is.na(pval)) {
          #sig_label <- if (pval < 0.001) "***" else if (pval < 0.01) "**" else if (pval < 0.05) "*" else "n.s."
          sig_label <- paste0("p=", round(pval, 3), " (", test, ")")
        }
      }
    }
    p <- ggplot(df_m, aes(x = condition, y = mean_value, group = subject, color = condition)) +
      geom_line(alpha = 0.2, position = position_dodge(width = 0.15), color = "grey40") +
      geom_point(alpha = 0.75, position = position_dodge(width = 0.15)) +
      geom_point(data = cond_summary, aes(x = condition, y = mean, color = condition), 
                    inherit.aes = FALSE, stroke = 1) +
      geom_errorbar(data = cond_summary,
                    aes(x = condition, ymin = mean - sem, ymax = mean + sem, color = condition),
                     inherit.aes = FALSE, width = 0.15, linewidth=1) +  
      scale_color_manual(values = CONDITION_COLORS) +
      labs(y = m, x = NULL)
    # Add significance bar without altering y-scale; place star visually outside
    if (!is.null(sig_label)) {
      y_data_max <- suppressWarnings(max(c(df_m$mean_value, cond_summary$mean + cond_summary$sem), na.rm = TRUE))
      if (is.finite(y_data_max)) {
        p <- p +
          coord_cartesian(clip = "off") +
          theme(plot.margin = margin(t = 16, r = 8, b = 8, l = 8)) +
          #geom_segment(aes(x = 1, xend = 2, y = y_data_max, yend = y_data_max), inherit.aes = FALSE) +
          annotate("text", x = 1.5, y = y_data_max, vjust = -0.6, label = sig_label, size = 5)
      }
    }
    print(p)
  }
}

# Block trajectories by order: mean paths + individual traces
save_block_trajectory_plots_pdf <- function(block_summary, exp, pdf_path) {
  if (nrow(block_summary) == 0) return(invisible(NULL))
  df <- block_summary %>%
    dplyr::filter(!is.na(.data$block), !is.na(.data$congruent)) %>%
    dplyr::mutate(
      order = factor(.data$order, levels = c("STTS", "TSST")),
      condition = factor(.data$condition, levels = c("sham", "taVNS")),
      block = factor(as.character(.data$block), ordered = TRUE)
    ) %>%
    dplyr::filter(.data$congruent == "incongruent") %>%
    tidyr::pivot_longer(
      cols = c("block_rt_mean", "block_acc_mean", "block_ies_mean", "block_score"),
      names_to = "metric", values_to = "value"
    ) %>%
    dplyr::mutate(metric = dplyr::recode(.data$metric,
      block_rt_mean  = "RT (s)",
      block_acc_mean = "Accuracy (%)",
      block_ies_mean = "IES (s)",
      block_score    = "Score"
    ))

  metrics <- unique(df$metric)
  grDevices::pdf(pdf_path, width = 11/2, height = 8.5/2)
  on.exit(grDevices::dev.off(), add = TRUE)
  for (m in metrics) {
    d_m <- df %>% dplyr::filter(.data$metric == m)
    means <- d_m %>% dplyr::group_by(.data$order, .data$block, .data$condition) %>%
      dplyr::summarise(mean_value = mean(.data$value, na.rm = TRUE),
                       sem_value = safe_sem(.data$value), .groups = "drop")
    p <- ggplot(d_m,aes(x = .data$block, y = .data$value, group = .data$subject, linetype = .data$order)) +
      geom_line(alpha = 0.2, linewidth = 0.5) +
      geom_point(alpha = 0.75, aes(color = .data$condition)) +
      geom_line(data = means, aes(x = .data$block, y = .data$mean_value, linetype = .data$order,
                    group = .data$order), linewidth = 1, inherit.aes = FALSE) +
      geom_errorbar(data = means,
                    aes(x = .data$block, ymin = .data$mean_value - .data$sem_value,
                        ymax = .data$mean_value + .data$sem_value, color = .data$condition),
                    inherit.aes = FALSE, width = 0.15, linewidth = 1) +
      scale_color_manual(values = CONDITION_COLORS) +
      scale_linetype_manual(values = c(STTS = "dashed", TSST = "solid")) +
      labs(x = "Block", y = m, color = "Condition", linetype = "Order")
    print(p)
  }
}

# Contrast trajectories by block: percent-change differences across blocks
save_contrast_block_trajectory_plots_pdf <- function(diffs, exp, pdf_path) {
  if (nrow(diffs) == 0) return(invisible(NULL))
  df <- diffs %>%
    dplyr::filter(!is.na(.data$block)) %>%
    dplyr::mutate(
      order = factor(.data$order, levels = c("STTS", "TSST")),
      condition = factor(.data$condition, levels = c("sham", "taVNS")),
      block = factor(as.character(.data$block), ordered = TRUE)
    ) %>%
    dplyr::rename(value = .data$difference)

  metrics <- unique(df$metric)
  grDevices::pdf(pdf_path, width = 11/2, height = 8.5/2)
  on.exit(grDevices::dev.off(), add = TRUE)
  for (m in metrics) {
    d_m <- df %>% dplyr::filter(.data$metric == m)
    means <- d_m %>% dplyr::group_by(.data$order, .data$block, .data$condition) %>%
      dplyr::summarise(mean_value = mean(.data$value, na.rm = TRUE),
                       sem_value = safe_sem(.data$value), .groups = "drop")
    p <- ggplot(d_m, aes(x = .data$block, y = .data$value, group = .data$subject, linetype = .data$order)) +
      geom_line(alpha = 0.2, linewidth = 0.5) +
      geom_point(alpha = 0.75, aes(color = .data$condition)) +
      geom_line(data = means, aes(x = .data$block, y = .data$mean_value, linetype = .data$order,
                    group = .data$order), linewidth = 1, inherit.aes = FALSE) +
      geom_errorbar(data = means,
                    aes(x = .data$block, ymin = .data$mean_value - .data$sem_value,
                        ymax = .data$mean_value + .data$sem_value, color = .data$condition),
                    inherit.aes = FALSE, width = 0.15, linewidth = 1) +
      scale_color_manual(values = CONDITION_COLORS) +
      scale_linetype_manual(values = c(STTS = "dashed", TSST = "solid")) +
      labs(x = "Block", y = m, color = "Condition", linetype = "Order")
    print(p)
  }
}

# Paired tests (t-test and Wilcoxon) for sham vs taVNS within subjects
run_paired_tests <- function(means_df) { 
  # Determine condition pair dynamically to support metrics, contrasts, and transitions
  conds <- sort(unique(as.character(means_df$condition)))
  if (length(conds) != 2) return(tibble())
  if (all(c("sham", "taVNS") %in% conds)) {
    x_name <- "sham"; y_name <- "taVNS"
  } else if (all(c("S->T", "T->S") %in% conds)) {
    x_name <- "S->T"; y_name <- "T->S"
  } else {
    x_name <- conds[1]; y_name <- conds[2]
  }
  # Wide by condition, keep only complete pairs
  wide <- means_df %>% tidyr::pivot_wider(names_from = condition, values_from = mean_value)
  if (!all(c(x_name, y_name) %in% names(wide))) return(tibble())
  wide <- wide %>% dplyr::filter(!is.na(.data[[x_name]]), !is.na(.data[[y_name]]))
  results <- wide %>% dplyr::group_by(metric) %>% dplyr::group_split()
  out <- lapply(results, function(dfm) {
    x <- dfm[[x_name]]
    y <- dfm[[y_name]]
    if (length(x) < 2 || length(y) < 2) return(NULL)
    # Differences for effect size
    d <- y - x
    # Normality test (Shapiro-Wilk) on paired differences
    sh <- tryCatch(stats::shapiro.test(d), error = function(e) NULL)
    t_res <- tryCatch(stats::t.test(y, x, paired = TRUE), error = function(e) NULL)
    w_res <- tryCatch(stats::wilcox.test(y, x, paired = TRUE, exact = FALSE), error = function(e) NULL)
    # 95% CI for mean difference (from paired t-test)
    ci_low <- if (!is.null(t_res) && !is.null(t_res$conf.int)) as.numeric(t_res$conf.int[1]) else NA_real_
    ci_high <- if (!is.null(t_res) && !is.null(t_res$conf.int)) as.numeric(t_res$conf.int[2]) else NA_real_
    tibble(
      metric = unique(dfm$metric),
      n_pairs = length(d),
      mean_sham = mean(x, na.rm = TRUE),
      sem_sham = safe_sem(x),
      mean_taVNS = mean(y, na.rm = TRUE),
      sem_taVNS = safe_sem(y),
      mean_diff = mean(d, na.rm = TRUE),
      sd_diff = stats::sd(d, na.rm = TRUE),
      shapiro_W = if (!is.null(sh)) unname(sh$statistic) else NA_real_,
      shapiro_p = if (!is.null(sh)) sh$p.value else NA_real_,
      normality = if (!is.null(sh)) ifelse(sh$p.value >= 0.05, "not-rejected", "rejected") else NA_character_,
      ci_95_low = ci_low,
      ci_95_high = ci_high,
      t_stat = if (!is.null(t_res)) unname(t_res$statistic) else NA_real_,
      t_p = if (!is.null(t_res)) t_res$p.value else NA_real_,
      w_stat = if (!is.null(w_res)) unname(w_res$statistic) else NA_real_,
      w_p = if (!is.null(w_res)) w_res$p.value else NA_real_
    )
  })
  out <- Filter(Negate(is.null), out)
  if (!length(out)) return(tibble())
  dplyr::bind_rows(out)
}

# Main
main <- function() {
  day_dir <- find_latest_day_dir(DATA_DIR)
  if (is.na(day_dir)) {
    stop("No dated output directory found in ", DATA_DIR)
  }
  message("Using outputs from ", day_dir)
  for (exp in EXPERIMENTS) {
    block_summary <- load_block_summary(day_dir, exp)
    if (nrow(block_summary) == 0) {
      message("No block summary for ", exp)
      next
    }
    # Apply subject exclusion (global + per-experiment) and scaling
    exp_exclusions <- EXCLUDE_SUBJECTS_BY_EXPERIMENT[[exp]]
    if (is.null(exp_exclusions)) exp_exclusions <- character(0)
    all_exclusions <- unique(as.character(c(EXCLUDE_SUBJECTS, exp_exclusions)))
    if (length(all_exclusions)) {
      message("Manual exclusions for ", exp, ": ", paste(all_exclusions, collapse = ", "))
    }
    block_summary <- block_summary %>%
      dplyr::filter(!.data$subject %in% all_exclusions) %>%
      dplyr::mutate(block_acc_mean = block_acc_mean * 100)

    # Per-experiment exclusion: drop subjects with avg incongruent accuracy < 60%
    inc_acc <- block_summary %>%
      dplyr::filter(.data$congruent == "incongruent") %>%
      dplyr::group_by(.data$subject) %>%
      dplyr::summarise(avg_inc_acc = mean(.data$block_acc_mean, na.rm = TRUE), .groups = "drop")
    low_acc_subjects <- inc_acc %>%
      dplyr::filter(.data$avg_inc_acc < 60) %>%
      dplyr::pull(.data$subject) %>% as.character()
    if (length(low_acc_subjects)) {
      message("Excluding subjects (<60% incongruent accuracy) for ", exp, ": ", paste(low_acc_subjects, collapse = ", "))
      block_summary <- block_summary %>% dplyr::filter(!.data$subject %in% low_acc_subjects)
    } else {
      message("No subjects excluded for <60% incongruent accuracy in ", exp)
    }

    # Metrics: subject-wise means and paired tests
    met_means <- subject_condition_means_for_metrics(block_summary)
    met_tests <- run_paired_tests(met_means)

    # Contrasts: compute diffs, subject-wise means and paired tests
    diffs <- compute_metric_differences(block_summary)
    con_means <- subject_condition_means_for_contrasts(diffs) %>%
      dplyr::filter(!.data$subject %in% EXCLUDE_SUBJECTS)
    con_tests <- run_paired_tests(con_means)

    # Transition deltas per subject using contrasts: T->S vs S->T
    trans_deltas <- compute_transition_deltas_contrasts(diffs)
    trans_csv <- file.path(day_dir, sprintf("%s_TransitionDeltas_Contrasts.csv", exp))
    trans_means <- subject_transition_means_for_metrics(trans_deltas)
    trans_tests <- run_paired_tests(trans_means)

    metrics_csv <- file.path(day_dir, sprintf("%s_PairedMeans_Metrics.csv", exp))
    contrasts_csv <- file.path(day_dir, sprintf("%s_PairedMeans_Contrasts.csv", exp))
    transitions_csv <- file.path(day_dir, sprintf("%s_TransitionDeltas_Contrasts.csv", exp))

    # Save per-condition means (aggregated across subjects)
    met_condition_summary <- met_means %>% dplyr::group_by(metric, condition) %>% dplyr::summarise(
      n_subjects = dplyr::n_distinct(subject),
      mean = mean(mean_value, na.rm = TRUE),
      sem = safe_sem(mean_value),
      .groups = "drop"
    )
    con_condition_summary <- con_means %>% dplyr::group_by(metric, condition) %>% dplyr::summarise(
      n_subjects = dplyr::n_distinct(subject),
      mean = mean(mean_value, na.rm = TRUE),
      sem = safe_sem(mean_value),
      .groups = "drop"
    )

    trans_condition_summary <- trans_means %>% dplyr::group_by(metric, condition) %>% dplyr::summarise(
      n_subjects = dplyr::n_distinct(subject),
      mean = mean(mean_value, na.rm = TRUE),
      sem = safe_sem(mean_value),
      .groups = "drop"
    )
    readr::write_csv(trans_condition_summary, transitions_csv)
    readr::write_csv(met_condition_summary, metrics_csv)
    readr::write_csv(con_condition_summary, contrasts_csv)

    # Also save the full paired test result tables to CSV to avoid truncation in text reports
    if (nrow(met_tests)) {
      readr::write_csv(met_tests, file.path(day_dir, sprintf("%s_PairedTests_Metrics.csv", exp)))
    }
    if (nrow(con_tests)) {
      readr::write_csv(con_tests, file.path(day_dir, sprintf("%s_PairedTests_Contrasts.csv", exp)))
    }
    if (nrow(trans_tests)) {
      readr::write_csv(trans_tests, file.path(day_dir, sprintf("%s_PairedTests_Transitions.csv", exp)))
    }

    # Save plots for each comparison
    save_paired_plots_pdf(met_means, paste0("Metrics by Condition (", exp, ")"),
          file.path(day_dir, sprintf("%s_Metrics_PairedPlots.pdf", exp)), tests_df = met_tests)
    save_paired_plots_pdf(con_means, paste0("Contrasts by Condition (", exp, ")"),
          file.path(day_dir, sprintf("%s_Contrasts_PairedPlots.pdf", exp)), tests_df = con_tests)
    save_paired_plots_pdf(trans_means, paste0("Transition Deltas by Condition (", exp, ")"),
          file.path(day_dir, sprintf("%s_TransitionDeltas_PairedPlots.pdf", exp)), tests_df = trans_tests)
    # Block trajectory plots by order with individual traces
    save_block_trajectory_plots_pdf(block_summary, exp,
          file.path(day_dir, sprintf("%s_BlockTrajectories.pdf", exp)))
  }
}

main()
