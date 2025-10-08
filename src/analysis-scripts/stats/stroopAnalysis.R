# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggsci)
library(jsonlite)
library(lmerTest)
library(stringr)
library(rlang) # for .data pronoun in dplyr code

# Set global parameters
start_date <- 20250701
end_date <- 20251031
data_dir <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-taVNS"
output_dir <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-taVNS/analyzed-data"

today <- format(Sys.Date(), "%Y%m%d")
output_dir <- file.path(output_dir, today)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Set a global theme
theme_set(
  theme_minimal() +
    theme(
      axis.line = element_line(linewidth = 1),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12),
      axis.ticks.length = unit(0.25, "cm"),
      axis.ticks = element_line(linewidth = 1),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top"
    )
)
update_geom_defaults("point", list(size = 3, shape = 3, stroke = 1))

#------------------------------------------------------------
# Extensible analysis plan (add metrics here)
# - trial_metrics: per-trial analyses broken out by congruency
# - aggregate_metrics: aggregated across congruency
analysis_plan <- list(
  trial_metrics = c("rt", "acc"),
  aggregate_metrics = c("rt", "acc", "score")
)

#------------------------------------------------------------
# Function to load and aggregate the data
load_stroop_data <- function(data_dir, start_date, end_date, experiment_type) {
  data_list <- list()
  subjects <- list.dirs(data_dir, full.names = TRUE, recursive = FALSE)
  for (subject in subjects) {
    if (basename(subject) == "test") next
    dates <- list.dirs(subject, full.names = FALSE, recursive = FALSE)
    for (date in dates) {
      if (as.numeric(date) >= start_date && as.numeric(date) <= end_date) {
        block_no <- 1
        blocks <- list.dirs(file.path(subject, date), full.names = FALSE, recursive = FALSE)
        for (block in blocks) {
          folder_path <- file.path(subject, date, block)
          files <- list.files(folder_path, full.names = TRUE, recursive = FALSE)
          config_file <- files[grepl("config.json", files, ignore.case = TRUE)]
          if (length(config_file) < 1) next
          config <- fromJSON(config_file)
          if (config["experiment"] == experiment_type) {
            stroop_file <- files[grepl("stroopTrials.csv", files, ignore.case = TRUE)]
            if (length(stroop_file) > 0) {
              stroopTable <- read.csv(stroop_file)
              stroopTable$subject <- basename(subject)
              stroopTable$block <- if (config["condition"] == "practice") "0" else as.character(block_no)
              if (config["condition"] != "practice") block_no <- block_no + 1
              stroopTable$condition <- config[["condition"]]
              stroopTable$order <- config[["order"]]
              if ("trial_type" %in% names(stroopTable)) {
                stroopTable$congruent <- factor(stroopTable$trial_type)
              }
              data_list[[length(data_list) + 1]] <- stroopTable
            }
          }
        }
      }
    }
  }
  if (length(data_list) > 0) {
    return(bind_rows(data_list))
  } else {
    return(data.frame())
  }
}

#------------------------------------------------------------
# Function to preprocess data
preprocess_data <- function(df, experiment_type) {
  df %>%
    filter(.data$condition != "practice") %>%
    mutate(
      subject = factor(.data$subject),
      order = factor(.data$order, levels = c("STTS", "TSST")),
      condition = factor(.data$condition),
      block = factor(.data$block, ordered = TRUE),
      correct = factor(.data$correct, levels = c("True", "False")),
      congruent = factor(.data$congruent)
    )
}

#------------------------------------------------------------
# Helper: compute block-level summary (per subject, order, condition, block, congruent)
compute_block_summary <- function(df) {
  df %>%
    group_by(.data$subject, .data$order, .data$condition, .data$block, .data$congruent) %>%
    summarise(
      rt = mean(.data$rt[.data$correct == "True"], na.rm = TRUE),
      accuracy = mean(.data$correct == "True", na.rm = TRUE),
      score = suppressWarnings(last(.data$score)),
      .groups = "drop"
    )
}

# Helper: compute subject-level summary per congruent
compute_subject_summary <- function(block_summary) {
  block_summary %>%
    group_by(.data$subject, .data$condition, .data$congruent) %>%
    summarise(
      rt_mean = mean(.data$rt, na.rm = TRUE),
      rt_sem = sd(.data$rt, na.rm = TRUE) / sqrt(dplyr::n()),
      acc_mean = mean(.data$accuracy, na.rm = TRUE),
      acc_sem = sd(.data$accuracy, na.rm = TRUE) / sqrt(dplyr::n()),
      score_mean = mean(.data$score, na.rm = TRUE),
      score_sem = sd(.data$score, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups = "drop"
    )
}

# Helper: compute group summary per congruent
compute_group_summary <- function(subject_summary) {
  subject_summary %>%
    group_by(.data$condition, .data$congruent) %>%
    summarise(
      rt_mean_group = mean(.data$rt_mean, na.rm = TRUE),
      rt_sem_group = sd(.data$rt_mean, na.rm = TRUE) / sqrt(dplyr::n()),
      acc_mean_group = mean(.data$acc_mean, na.rm = TRUE),
      acc_sem_group = sd(.data$acc_mean, na.rm = TRUE) / sqrt(dplyr::n()),
      score_mean_group = mean(.data$score_mean, na.rm = TRUE),
      score_sem_group = sd(.data$score_mean, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups = "drop"
    )
}

# Helper: compute aggregate summaries across congruent
compute_aggregate_summaries <- function(df) {
  agg_block_summary <- df %>%
    group_by(.data$subject, .data$condition, .data$block, .data$order) %>%
    summarise(
      rt = mean(.data$rt[.data$correct == "True"], na.rm = TRUE),
      accuracy = mean(.data$correct == "True", na.rm = TRUE),
      score = suppressWarnings(last(.data$score)),
      .groups = "drop"
    )

  agg_order_summary <- agg_block_summary %>%
    group_by(.data$block, .data$order, .data$condition) %>%
    summarise(
      rt_mean = mean(.data$rt, na.rm = TRUE),
      rt_sem = sd(.data$rt, na.rm = TRUE) / sqrt(dplyr::n()),
      acc_mean = mean(.data$accuracy, na.rm = TRUE),
      acc_sem = sd(.data$accuracy, na.rm = TRUE) / sqrt(dplyr::n()),
      score_mean = mean(.data$score, na.rm = TRUE),
      score_sem = sd(.data$score, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups = "drop"
    )

  agg_subject_summary <- agg_block_summary %>%
    group_by(.data$subject, .data$condition) %>%
    summarise(
      rt_mean = mean(.data$rt, na.rm = TRUE),
      rt_sem = sd(.data$rt, na.rm = TRUE) / sqrt(dplyr::n()),
      acc_mean = mean(.data$accuracy, na.rm = TRUE),
      acc_sem = sd(.data$accuracy, na.rm = TRUE) / sqrt(dplyr::n()),
      score_mean = mean(.data$score, na.rm = TRUE),
      score_sem = sd(.data$score, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups = "drop"
    )

  agg_group_summary <- agg_subject_summary %>%
    group_by(.data$condition) %>%
    summarise(
      rt_mean_group = mean(.data$rt_mean, na.rm = TRUE),
      rt_sem_group = sd(.data$rt_mean, na.rm = TRUE) / sqrt(dplyr::n()),
      acc_mean_group = mean(.data$acc_mean, na.rm = TRUE),
      acc_sem_group = sd(.data$acc_mean, na.rm = TRUE) / sqrt(dplyr::n()),
      score_mean_group = mean(.data$score_mean, na.rm = TRUE),
      score_sem_group = sd(.data$score_mean, na.rm = TRUE) / sqrt(dplyr::n()),
      .groups = "drop"
    )

  list(
    agg_block_summary = agg_block_summary,
    agg_order_summary = agg_order_summary,
    agg_subject_summary = agg_subject_summary,
    agg_group_summary = agg_group_summary
  )
}

#------------------------------------------------------------
# Function to define color palette
condition_colors <- c("#68689a", "#6666ff")
names(condition_colors) <- c("sham", "taVNS")


#------------------------------------------------------------
# Function to generate plots for RT by congruency and condition
plot_rt_violin_box <- function(df_correct, condition_colors) {
  p <- ggplot(df_correct, aes(x = .data$congruent, y = .data$rt, fill = .data$condition)) +
    geom_violin(position = position_dodge(1), alpha = 0.5, color = NA) +
    scale_fill_manual(values = condition_colors) +
    geom_boxplot(width = 0.1, position = position_dodge(1)) +
    labs(
      title = "Response Time by Condition and Congruency",
      y = "Response Time (s)"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
    # ylim(0.5, 4)
  print(p)
}

#------------------------------------------------------------
# Function to generate paired subject plots with group summary for RT or accuracy
# type can be "rt" or "acc"
plot_subject_summary <- function(subject_summary, group_summary, trial, type, condition_colors) {
  y_label <- ifelse(type == "rt", "response time (s)", "accuracy (%)")
  title_label <- ifelse(type == "rt",
    paste("Paired Response Times (Congruent:", trial, ")"),
    paste("Paired Accuracy (Congruent:", trial, ")")
  )

  # Column names based on type
  subj_y <- if (type == "rt") "rt_mean" else "acc_mean"
  grp_y  <- if (type == "rt") "rt_mean_group" else "acc_mean_group"
  grp_sem <- if (type == "rt") "rt_sem_group" else "acc_sem_group"

  df_subj <- dplyr::filter(subject_summary, .data$congruent == trial)
  df_grp <- dplyr::filter(group_summary, .data$congruent == trial)

  p <- ggplot(df_subj, aes(x = .data$condition)) +
    geom_line(aes(y = .data[[subj_y]], group = .data$subject),
      alpha = 0.4, color = "gray40", linewidth = 1, position = position_jitter(width=0.1, height=0)
    ) +
    geom_errorbar(
      data = df_grp,
      aes(x = .data$condition,
          ymin = .data[[grp_y]] - .data[[grp_sem]],
          ymax = .data[[grp_y]] + .data[[grp_sem]],
          color = .data$condition
      ),
      width = 0.1, linewidth = 1, inherit.aes = FALSE
    ) +
    geom_point(
      data = df_grp,
      aes(x = .data$condition, y = .data[[grp_y]], color = .data$condition),
      inherit.aes = FALSE
    ) +
    scale_color_manual(values = condition_colors) +
    labs(title = title_label, y = y_label)

  # Set y-axis limits based on type
  # if (type == "rt") {
  #   p <- p + ylim(0, 3)
  # } else {
  #   p <- p + ylim(0.2, 1)
  # }

  print(p)
}

#------------------------------------------------------------
# Function to plot aggregated paired results across all trial types
plot_aggregate_summary <- function(subject_summary, group_summary, type, condition_colors) {
  y_label <- switch(type,
    rt = "response time (s)",
    acc = "accuracy (%)",
    score = "score",
    "value"
  )
  title_label <- switch(type,
    rt = "Paired Response Times Across All Conditions",
    acc = "Paired Accuracy Across All Conditions",
    score = "Paired Scores Across All Conditions",
    paste("Paired", toupper(type), "Across All Conditions")
  )

  subj_y <- switch(type, rt = "rt_mean", acc = "acc_mean", score = "score_mean", stop("Unknown type"))
  grp_y  <- switch(type, rt = "rt_mean_group", acc = "acc_mean_group", score = "score_mean_group")
  grp_sem <- switch(type, rt = "rt_sem_group", acc = "acc_sem_group", score = "score_sem_group")

  p <- ggplot(subject_summary, aes(x = .data$condition, y = .data[[subj_y]], group = .data$subject)) +
    geom_line(alpha = 0.4, color = "gray40", linewidth = 1) +
    geom_point(
      data = group_summary,
      aes(x = .data$condition, y = .data[[grp_y]], color = .data$condition),
      inherit.aes = FALSE
    ) +
    geom_errorbar(
      data = group_summary,
      aes(
        x = .data$condition,
        ymin = .data[[grp_y]] - .data[[grp_sem]],
        ymax = .data[[grp_y]] + .data[[grp_sem]],
        color = .data$condition
      ),
      width = 0.1, linewidth = 1, inherit.aes = FALSE
    ) +
    scale_color_manual(values = condition_colors) +
    labs(title = title_label, y = y_label)

  # if (type == "rt") {
  #   p <- p + ylim(0, 2)
  # } else if (type == "acc") {
  #   p <- p + ylim(0.5, 1)
  # }

  print(p)
}

#------------------------------------------------------------
# Function to plot block-level summary for "rt", "acc", or "score"
plot_by_block <- function(agg_order_summary, agg_block_summary, type, condition_colors) {
  if (type == "rt") {
    y_col_order <- "rt_mean"
    y_col_block <- "rt"
    sem_col <- "rt_sem"
    title_text <- "Response Time by Block - All trial types"
    y_label <- "Response Time (s)"
  } else if (type == "acc") {
    y_col_order <- "acc_mean"
    y_col_block <- "accuracy"
    sem_col <- "acc_sem"
    title_text <- "Accuracy by Block - All trial types"
    y_label <- "Accuracy (%)"
  } else if (type == "score") {
    y_col_order <- "score_mean"
    y_col_block <- "score"
    sem_col <- "score_sem"
    title_text <- "Score by Block - All trial types"
    y_label <- "Score"
  } else {
    stop("Unknown type. Please choose 'rt', 'acc', or 'score'.")
  }

  p <- ggplot(agg_order_summary, aes(x = .data$block, y = .data[[y_col_order]], linetype = .data$order)) +
    # Faint lines for each subject
    geom_line(
      data = agg_block_summary, aes(x = .data$block, y = .data[[y_col_block]], group = .data$subject),
      alpha = 0.3, linewidth = 0.8
    ) +
    # Bold line for each order (group summary)
    geom_line(data = agg_order_summary, aes(group = .data$order), linewidth = 1.2) +
    geom_errorbar(
      aes(
        ymin = .data[[y_col_order]] - .data[[sem_col]],
        ymax = .data[[y_col_order]] + .data[[sem_col]],
        color = .data$condition
      ),
      width = 0.1, linewidth = 1
    ) +
    scale_color_manual(values = condition_colors) +
    labs(
      title = title_text,
      x = "Block", y = y_label
    ) +
    theme_minimal()

  print(p)
}

#------------------------------------------------------------
# Main execution function for a single experiment
run_analysis <- function(experiment_type) {
  pdf_file <- file.path(output_dir, paste0("ComparisonPlots_", experiment_type, ".pdf"))
  report_file <- file.path(output_dir, paste0("StatisticalReport_", experiment_type, ".txt"))
  pdf(pdf_file)
  df_raw <- load_stroop_data(data_dir, start_date, end_date, experiment_type)
  if (nrow(df_raw) == 0) {
    warning(paste("No data loaded for", experiment_type))
    dev.off()
    return(NULL)
  }
  df <- preprocess_data(df_raw, experiment_type)
  df$congruent <- str_replace_all(string=df$congruent, pattern="cue", replacement="stim")
  df_correct <- df %>% dplyr::filter(.data$correct == "True")
  
  # Example trial-level plot (only for RT); more can be added later via plan
  plot_rt_violin_box(df_correct, condition_colors)

  # Summaries
  block_summary <- compute_block_summary(df)
  subject_summary <- compute_subject_summary(block_summary)
  group_summary <- compute_group_summary(subject_summary)

  # Initialize lists for statistical test outputs
  t_test <- list()
  wilcox <- list()

  # Loop over each level of congruency
  for (trial in unique(subject_summary$congruent)) {
    # Wilcoxon (trial-level RTs)
    if ("rt" %in% analysis_plan$trial_metrics) {
      pivot_wilcox <- df_correct %>%
        dplyr::filter(.data$congruent == trial) %>%
        dplyr::select(.data$subject, .data$condition, .data$block, .data$trial_number, .data$rt) %>%
        tidyr::pivot_wider(names_from = .data$condition, values_from = .data$rt)
      if (all(c("taVNS", "sham") %in% names(pivot_wilcox))) {
        wilcox[[as.character(trial)]] <- wilcox.test(pivot_wilcox$taVNS, pivot_wilcox$sham, exact = TRUE)$p.value
      }
      # Subject-level paired t-test (RT)
      pivot_rt <- subject_summary %>%
        dplyr::filter(.data$congruent == trial) %>%
        dplyr::select(.data$subject, .data$rt_mean, .data$condition) %>%
        tidyr::pivot_wider(names_from = .data$condition, values_from = .data$rt_mean)
      if (all(c("taVNS", "sham") %in% names(pivot_rt))) {
        t_test[[paste0("rt_", trial)]] <- t.test(pivot_rt$taVNS, pivot_rt$sham, paired = TRUE)$p.value
      }
      plot_subject_summary(subject_summary, group_summary, trial, type = "rt", condition_colors = condition_colors)
    }

    if ("acc" %in% analysis_plan$trial_metrics) {
      pivot_acc <- subject_summary %>%
        dplyr::filter(.data$congruent == trial) %>%
        dplyr::select(.data$subject, .data$acc_mean, .data$condition) %>%
        tidyr::pivot_wider(names_from = .data$condition, values_from = .data$acc_mean)
      if (all(c("taVNS", "sham") %in% names(pivot_acc))) {
        t_test[[paste0("acc_", trial)]] <- t.test(pivot_acc$taVNS, pivot_acc$sham, paired = TRUE)$p.value
      }
      plot_subject_summary(subject_summary, group_summary, trial, type = "acc", condition_colors = condition_colors)
    }
  }

  # Aggregate summaries for all trial types
  if (experiment_type == "StroopSquared") {
    df <- df %>% filter(subject != "ERC22")
  } 
  aggs <- compute_aggregate_summaries(df)

  # Plot block-level summaries for RT, Accuracy, and Score
  plot_by_block(aggs$agg_order_summary, aggs$agg_block_summary, "rt", condition_colors)
  plot_by_block(aggs$agg_order_summary, aggs$agg_block_summary, "acc", condition_colors)
  plot_by_block(aggs$agg_order_summary, aggs$agg_block_summary, "score", condition_colors)


  # Paired t-test for overall RT and plot aggregate summary
  pivot_all_rt <- aggs$agg_subject_summary %>%
    dplyr::select(.data$subject, .data$rt_mean, .data$condition) %>%
    tidyr::pivot_wider(names_from = .data$condition, values_from = .data$rt_mean)
  t_test[["rt_all"]] <- t.test(pivot_all_rt$taVNS, pivot_all_rt$sham, paired = TRUE)$p.value

  plot_aggregate_summary(aggs$agg_subject_summary, aggs$agg_group_summary, type = "rt", condition_colors = condition_colors)

  # Paired t-test for overall Accuracy and plot aggregate summary
  pivot_all_acc <- aggs$agg_subject_summary %>%
    dplyr::select(.data$subject, .data$acc_mean, .data$condition) %>%
    tidyr::pivot_wider(names_from = .data$condition, values_from = .data$acc_mean)
  t_test[["acc_allTypes"]] <- t.test(pivot_all_acc$taVNS, pivot_all_acc$sham, paired = TRUE)$p.value

  plot_aggregate_summary(aggs$agg_subject_summary, aggs$agg_group_summary, type = "acc", condition_colors = condition_colors)

  # Paired t-test for score and plot aggregate summary
  pivot_score <- aggs$agg_subject_summary %>%
    dplyr::select(.data$subject, .data$score_mean, .data$condition) %>%
    tidyr::pivot_wider(names_from = .data$condition, values_from = .data$score_mean)
  t_test[["score"]] <- t.test(pivot_score$taVNS, pivot_score$sham, paired = TRUE)$p.value

  plot_aggregate_summary(aggs$agg_subject_summary, aggs$agg_group_summary, type = "score", condition_colors = condition_colors)

  # Count subjects and trials
  trial_counts <- df %>%
    dplyr::group_by(.data$subject, .data$condition, .data$block, .data$congruent, .data$correct) %>%
    dplyr::summarise(n_trials = dplyr::n(), .groups = "drop")

  # Save statistical test results to file
  sink(report_file)

  print("Subject and Trial Counts by Condition:")
  subject_counts <- df_correct %>% dplyr::summarise(N = dplyr::n_distinct(.data$subject))
  print("Number of subjects (N) per condition:")
  print(subject_counts)
  print("Number of trials (n) for each subject:")
  print(trial_counts, n = Inf)

  print("Wilcox test p-values:")
  print(wilcox)

  print("Paired t-test p-values:")
  print(t_test)

  sink()
  dev.off()
}

#------------------------------------------------------------
# Run for both experiment types
experiment_types <- c("SCWT", "StroopSquared")
for (exp_type in experiment_types) {
  run_analysis(exp_type)
}
