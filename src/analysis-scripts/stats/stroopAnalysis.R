# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggsci)
library(jsonlite)
library(lmerTest)

# Set global parameters
start_date <- 20250701
end_date <- 20250930
data_dir <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-taVNS/Data"
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
    filter(condition != "practice") %>%
    mutate(
      subject = factor(subject),
      order = factor(order, levels = c("STTS", "TSST")),
      condition = factor(condition),
      block = factor(block, ordered = TRUE),
      correct = factor(correct, levels = c("True", "False")),
      congruent = factor(congruent)
    )
}

#------------------------------------------------------------
# Function to define color palette
condition_colors <- c("#68689a", "#6666ff")
names(condition_colors) <- c("sham", "taVNS")


#------------------------------------------------------------
# Function to generate plots for RT by congruency and condition
plot_rt_violin_box <- function(df_correct, condition_colors) {
  p <- ggplot(df_correct, aes(x = congruent, y = rt, fill = condition)) +
    geom_violin(position = position_dodge(1), alpha = 0.5, color = NA) +
    scale_fill_manual(values = condition_colors) +
    geom_boxplot(width = 0.1, position = position_dodge(1)) +
    labs(
      title = "Response Time by Condition and Congruency",
      y = "Response Time (s)"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(0.5, 4)
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

  valueAes <- if (type == "rt") {
    aes(y = rt_mean)
  } else {
    aes(y = acc_mean)
  }
  groupAes <- aes(group = subject)

  p <- ggplot(filter(subject_summary, congruent == trial), aes(x = condition)) +
    geom_line(aes(y = if (type == "rt") rt_mean else acc_mean, group = subject),
      alpha = 0.4, color = "gray40", linewidth = 1
    )

  if (type == "rt") {
    p <- p +
      geom_errorbar(
        data = filter(group_summary, congruent == trial),
        aes(x = condition, ymin = rt_mean_group - rt_sem_group, ymax = rt_mean_group + rt_sem_group, color = condition),
        width = 0.1, linewidth = 1, inherit.aes = FALSE
      ) +
      geom_point(
        data = filter(group_summary, congruent == trial),
        aes(x = condition, y = rt_mean_group, color = condition),
        inherit.aes = FALSE
      )
  } else {
    p <- p +
      geom_errorbar(
        data = filter(group_summary, congruent == trial),
        aes(x = condition, ymin = acc_mean_group - acc_sem_group, ymax = acc_mean_group + acc_sem_group, color = condition),
        width = 0.1, linewidth = 1, inherit.aes = FALSE
      ) +
      geom_point(
        data = filter(group_summary, congruent == trial),
        aes(x = condition, y = acc_mean_group, color = condition),
        inherit.aes = FALSE
      )
  }

  p <- p +
    scale_color_manual(values = condition_colors) +
    labs(title = title_label, y = y_label)

  # Set y-axis limits based on type
  if (type == "rt") {
    p <- p + ylim(0, 3)
  } else {
    p <- p + ylim(0.5, 1)
  }

  print(p)
}

#------------------------------------------------------------
# Function to plot aggregated paired results across all trial types
plot_aggregate_summary <- function(subject_summary, group_summary, type, condition_colors) {
  y_label <- if (type == "rt") {
    "response time (s)"
  } else if (type == "acc") {
    "accuracy (%)"
  } else {
    "score"
  }

  title_label <- if (type == "rt") {
    "Paired Response Times Across All Conditions"
  } else if (type == "acc") {
    "Paired Accuracy Across All Conditions"
  } else {
    "Paired Scores Across All Conditions"
  }

  p <- ggplot(subject_summary, aes(x = condition, y = if (type == "rt") rt_mean else if (type == "acc") acc_mean else score_mean, group = subject)) +
    geom_line(alpha = 0.4, color = "gray40", linewidth = 1) +
    geom_point(
      data = group_summary,
      aes(
        x = condition, y = if (type == "rt") rt_mean_group else if (type == "acc") acc_mean_group else score_mean_group,
        color = condition
      ), inherit.aes = FALSE
    ) +
    geom_errorbar(
      data = group_summary,
      aes(
        x = condition,
        ymin = if (type == "rt") {
          rt_mean_group - rt_sem_group
        } else if (type == "acc") {
          acc_mean_group - acc_sem_group
        } else {
          score_mean_group - score_sem_group
        },
        ymax = if (type == "rt") {
          rt_mean_group + rt_sem_group
        } else if (type == "acc") {
          acc_mean_group + acc_sem_group
        } else {
          score_mean_group + score_sem_group
        },
        color = condition
      ),
      width = 0.1, linewidth = 1, inherit.aes = FALSE
    ) +
    scale_color_manual(values = condition_colors) +
    labs(title = title_label, y = y_label)

  if (type == "rt") {
    p <- p + ylim(0, 2)
  } else if (type == "acc") {
    p <- p + ylim(0.5, 1)
  }

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

  p <- ggplot(agg_order_summary, aes(x = block, y = .data[[y_col_order]], linetype = order)) +
    # Faint lines for each subject
    geom_line(
      data = agg_block_summary, aes(x = block, y = .data[[y_col_block]], group = subject),
      alpha = 0.3, linewidth = 0.8
    ) +
    # Bold line for each order (group summary)
    geom_line(aes(group = order), linewidth = 1.2) +
    geom_errorbar(
      aes(
        ymin = .data[[y_col_order]] - .data[[sem_col]],
        ymax = .data[[y_col_order]] + .data[[sem_col]],
        color = condition
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
  df_correct <- df %>% filter(correct == "True")
  plot_rt_violin_box(df_correct, condition_colors)

  # Create block-level summary for RT and Accuracy
  block_summary <- df %>%
    group_by(subject, order, condition, block, congruent) %>%
    summarise(
      rt = mean(rt[correct == "True"], na.rm = TRUE),
      accuracy = sum(correct == "True") / n(),
      score = last(score),
      .groups = "drop"
    )


  # Calculate subject-level summary per trial type (by congruent)
  subject_summary <- block_summary %>%
    group_by(subject, condition, congruent) %>%
    summarise(
      rt_mean = mean(rt, na.rm = TRUE),
      rt_sem = sd(rt, na.rm = TRUE) / sqrt(n()),
      acc_mean = mean(accuracy, na.rm = TRUE),
      acc_sem = sd(accuracy, na.rm = TRUE) / sqrt(n()),
      score_mean = mean(score, na.rm = TRUE),
      score_sem = sd(score, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  # Group summary for each trial type (for plotting error bars)
  group_summary <- subject_summary %>%
    group_by(condition, congruent) %>%
    summarise(
      rt_mean_group = mean(rt_mean, na.rm = TRUE),
      rt_sem_group = sd(rt_mean, na.rm = TRUE) / sqrt(n()),
      acc_mean_group = mean(acc_mean, na.rm = TRUE),
      acc_sem_group = sd(acc_mean, na.rm = TRUE) / sqrt(n()),
      score_mean_group = mean(score_mean, na.rm = TRUE),
      score_sem_group = sd(score_mean, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  # Initialize lists for statistical test outputs
  t_test <- list()
  wilcox <- list()

  # Loop over each level of congruency
  for (trial in unique(subject_summary$congruent)) {
    # Wilcoxon signed rank test on individual trial RTs
    pivot_wilcox <- df_correct %>%
      filter(congruent == trial) %>%
      select(subject, condition, block, trial_number, rt) %>%
      pivot_wider(names_from = condition, values_from = rt)
    wilcox[[as.character(trial)]] <- wilcox.test(pivot_wilcox$taVNS, pivot_wilcox$sham, exact = TRUE)$p.value

    # Paired t-test for RT using subject-level summary
    pivot_rt <- subject_summary %>%
      filter(congruent == trial) %>%
      select(subject, rt_mean, condition) %>%
      pivot_wider(names_from = condition, values_from = rt_mean)
    t_test[[paste0("rt_", trial)]] <- t.test(pivot_rt$taVNS, pivot_rt$sham, paired = TRUE)$p.value

    # Plot subject RT with group summary
    plot_subject_summary(subject_summary, group_summary, trial, type = "rt", condition_colors = condition_colors)

    # Paired t-test for Accuracy
    pivot_acc <- subject_summary %>%
      filter(congruent == trial) %>%
      select(subject, acc_mean, condition) %>%
      pivot_wider(names_from = condition, values_from = acc_mean)
    t_test[[paste0("acc_", trial)]] <- t.test(pivot_acc$taVNS, pivot_acc$sham, paired = TRUE)$p.value

    # Plot subject Accuracy with group summary
    plot_subject_summary(subject_summary, group_summary, trial, type = "acc", condition_colors = condition_colors)
  }

  # Aggregate summaries for all trial types
  agg_block_summary <- df %>%
    group_by(subject, condition, block, order) %>%
    summarise(
      rt = mean(rt[correct == "True"], na.rm = TRUE),
      accuracy = sum(correct == "True") / n(),
      score = last(score),
      .groups = "drop"
    )

  agg_order_summary <- agg_block_summary %>%
    group_by(block, order, condition) %>%
    summarise(
      rt_mean = mean(rt, na.rm = TRUE),
      rt_sem = sd(rt, na.rm = TRUE) / sqrt(n()),
      acc_mean = mean(accuracy, na.rm = TRUE),
      acc_sem = sd(accuracy, na.rm = TRUE) / sqrt(n()),
      score_mean = mean(score, na.rm = TRUE),
      score_sem = sd(score, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  agg_subject_summary <- agg_block_summary %>%
    group_by(subject, condition) %>%
    summarise(
      rt_mean = mean(rt, na.rm = TRUE),
      rt_sem = sd(rt, na.rm = TRUE) / sqrt(n()),
      acc_mean = mean(accuracy, na.rm = TRUE),
      acc_sem = sd(accuracy, na.rm = TRUE) / sqrt(n()),
      score_mean = mean(score, na.rm = TRUE),
      score_sem = sd(score, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  agg_group_summary <- agg_subject_summary %>%
    group_by(condition) %>%
    summarise(
      rt_mean_group = mean(rt_mean, na.rm = TRUE),
      rt_sem_group = sd(rt_mean, na.rm = TRUE) / sqrt(n()),
      acc_mean_group = mean(acc_mean, na.rm = TRUE),
      acc_sem_group = sd(acc_mean, na.rm = TRUE) / sqrt(n()),
      score_mean_group = mean(score_mean, na.rm = TRUE),
      score_sem_group = sd(score_mean, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  # Plot block-level summaries for RT, Accuracy, and Score
  plot_by_block(agg_order_summary, agg_block_summary, "rt", condition_colors)
  plot_by_block(agg_order_summary, agg_block_summary, "acc", condition_colors)
  plot_by_block(agg_order_summary, agg_block_summary, "score", condition_colors)


  # Paired t-test for overall RT and plot aggregate summary
  pivot_all_rt <- agg_subject_summary %>%
    select(subject, rt_mean, condition) %>%
    pivot_wider(names_from = condition, values_from = rt_mean)
  t_test[["rt_all"]] <- t.test(pivot_all_rt$taVNS, pivot_all_rt$sham, paired = TRUE)$p.value

  plot_aggregate_summary(agg_subject_summary, agg_group_summary, type = "rt", condition_colors = condition_colors)

  # Paired t-test for overall Accuracy and plot aggregate summary
  pivot_all_acc <- agg_subject_summary %>%
    select(subject, acc_mean, condition) %>%
    pivot_wider(names_from = condition, values_from = acc_mean)
  t_test[["acc_allTypes"]] <- t.test(pivot_all_acc$taVNS, pivot_all_acc$sham, paired = TRUE)$p.value

  plot_aggregate_summary(agg_subject_summary, agg_group_summary, type = "acc", condition_colors = condition_colors)

  # Paired t-test for score and plot aggregate summary
  pivot_score <- agg_subject_summary %>%
    select(subject, score_mean, condition) %>%
    pivot_wider(names_from = condition, values_from = score_mean)
  t_test[["score"]] <- t.test(pivot_score$taVNS, pivot_score$sham, paired = TRUE)$p.value

  plot_aggregate_summary(agg_subject_summary, agg_group_summary, type = "score", condition_colors = condition_colors)

  # Count subjects and trials
  trial_counts <- df %>%
    group_by(subject, condition, block, congruent, correct) %>%
    summarise(n_trials = n(), .groups = "drop")

  # Save statistical test results to file
  sink(report_file)

  print("Subject and Trial Counts by Condition:")
  subject_counts <- df_correct %>% summarise(N = n_distinct(subject))
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
