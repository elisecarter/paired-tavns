# Stroop Task Exploratory Data Analysis
#
# Summarize condition effects (taVNS vs sham) on RT, accuracy, and scores at the
# trial, block, and session levels. Outputs CSV summaries and plot PDFs to
# support hypothesis generation.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(jsonlite)
  library(ggplot2)
  library(patchwork)
  library(readr)
})

DATA_DIR <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/Data"
OUTPUT_DIR <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-taVNS-analysis/analyzed-data"
START_DATE <- 20250701
END_DATE <- Inf
CONDITION_LEVELS <- c("sham", "taVNS")
CONGRUENCY_LEVELS <- c("congruent", "incongruent", "cue incongruent", "responses incongruent")
CONDITION_COLORS <- c(sham = "#1C98E3", taVNS = "#E3671C")
invisible(utils::globalVariables(c(
  "metric", "value", "blocks", "sessions",
  "subject", "condition", "session_rt_mean", "session_acc_mean",
  "session_score_mean", "category", "n_subjects", "difference"
)))


safe_sem <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x) / sqrt(length(x))
}

labelize <- function(x) {
  cleaned <- gsub("_", " ", x, fixed = TRUE)
  cleaned <- trimws(cleaned)
  tools::toTitleCase(cleaned)
}

is_integerish <- function(x) {
  is.numeric(x) && all(is.na(x) | abs(x - round(x)) < .Machine$double.eps^0.5)
}

classify_var_type <- function(vec) {
  if (inherits(vec, c("factor", "ordered", "character", "logical"))) return("discrete")
  if (is_integerish(vec)) return("discrete")
  "continuous"
}

bin_continuous_var <- function(vec, bins = 4) {
  vals <- vec[is.finite(vec)]
  if (length(unique(vals)) <= bins) {
    return(factor(vec))
  }
  suppressWarnings(ggplot2::cut_number(vec, n = bins, dig.lab = 6))
}

`%||%` <- function(x, y) if (is.null(x) || is.na(x)) y else x

prep_output_paths <- function(experiment_type) {
  today <- format(Sys.Date(), "%Y%m%d")
  day_dir <- file.path(OUTPUT_DIR, today)
  if (!dir.exists(day_dir)) dir.create(day_dir, recursive = TRUE)
  list(
    summary_csv = file.path(day_dir, sprintf("%s_EDA_Summary.csv", experiment_type)),
    block_csv = file.path(day_dir, sprintf("%s_BlockSummary.csv", experiment_type)),
    trial_csv = file.path(day_dir, sprintf("%s_Trials.csv", experiment_type)),
    plots_pdf = file.path(day_dir, sprintf("%s_EDA_Plots.pdf", experiment_type)),
    subject_csv = file.path(day_dir, sprintf("%s_SubjectOverview.csv", experiment_type)),
    session_change_csv = file.path(day_dir, sprintf("%s_SessionChangeSummary.csv", experiment_type))
  )
}   

# --- 3. Data Loading Function ---

load_stroop_data <- function(data_dir, start_date, end_date, experiment_type) {
  subjects <- list.dirs(data_dir, recursive = FALSE, full.names = TRUE)
  rows <- vector("list", length(subjects))
  row_id <- 0L
  for (subject_path in subjects) {
    subject_id <- basename(subject_path)
    if (subject_id %in% c("test", "analyzed-data")) next
    date_dirs <- list.dirs(subject_path, recursive = FALSE, full.names = TRUE)
    for (date_path in date_dirs) {
      date_label <- basename(date_path)
      date_num <- suppressWarnings(as.numeric(date_label))
      if (is.na(date_num) || date_num < start_date || date_num > end_date) next
      block_no <- 1L
      block_dirs <- list.dirs(date_path, recursive = FALSE, full.names = TRUE)
      for (block_path in block_dirs) {
        files <- list.files(block_path, full.names = TRUE)
        config_file <- files[grepl("config.json", files, ignore.case = TRUE)]
        if (length(config_file) == 0) next
        config <- jsonlite::fromJSON(config_file)
        if (is.null(config$experiment) || config$experiment != experiment_type) next
        stroop_file <- files[grepl("stroopTrials\\.csv", files, ignore.case = TRUE)]
        if (length(stroop_file) == 0) next
        block_df <- readr::read_csv(stroop_file, show_col_types = FALSE)
        block_df$subject <- subject_id
        block_df$order <- config$order 
        block_df$condition <- config$condition
        block_df$block <- if (identical(config$condition, "practice")) "0" else as.character(block_no)
        block_df$amplitude <- config$current_mA
        block_df$percent_threshold <- config$percent_PT
        block_df$freq <- config$stim_f
        block_df$rating <- config$stim_rating
        block_df$score <- last(block_df$score)
        if (!identical(config$condition, "practice")) block_no <- block_no + 1L
        if ("trial_type" %in% names(block_df)) {
          if ("congruent" %in% names(block_df)) {
            # drop the redundant trial_type if congruent already exists
            block_df$trial_type <- NULL
          } else {
            # rename trial_type to congruent (removes trial_type)
            names(block_df)[names(block_df) == "trial_type"] <- "congruent"
          }
        }
        row_id <- row_id + 1L
        rows[[row_id]] <- block_df
      }
    }
  }
  if (row_id == 0L) return(tibble())
  dplyr::bind_rows(rows[seq_len(row_id)])
}


# --- 4. Preprocessing and Feature Engineering ---

prepare_data <- function(df, experiment_type) {
  processed <- df %>%
    filter(.data$condition != "practice") %>%
    mutate(
      subject = factor(.data$subject),
      order = factor(.data$order, levels = c("STTS", "TSST")),
      condition = factor(.data$condition, levels = CONDITION_LEVELS),
      block = suppressWarnings(as.integer(.data$block)),
      block = factor(.data$block, ordered = TRUE),
      trial_number = suppressWarnings(as.integer(.data$trial_number)),
      correct_numeric = ifelse(.data$correct %in% c("True", "TRUE", "true", "1", 1), 1L, 0L),
      prev_correct = dplyr::lag(.data$correct_numeric),
      prev_rt = dplyr::lag(.data$rt),
      amplitude = suppressWarnings(as.numeric(.data$amplitude)),
      percent_threshold = suppressWarnings(as.numeric(.data$percent_threshold)),
      freq = suppressWarnings(as.numeric(.data$freq)),
      rating = suppressWarnings(as.numeric(.data$rating)),
      score = suppressWarnings(as.numeric(.data$score)),
      rt = suppressWarnings(as.numeric(.data$rt))
    ) %>%
    mutate(
      congruent = dplyr::case_when(
        experiment_type == "StroopSquared" ~ dplyr::case_when(
          .data$congruent %in% c("cueCongruent_responsesCongruent", "stimCongruent_responsesCongruent") ~ "congruent",
          .data$congruent %in% c("cueIncongruent_responsesIncongruent", "stimIncongruent_responsesIncongruent") ~ "incongruent",
          .data$congruent %in% c("cueCongruent_responsesIncongruent", "stimCongruent_responsesIncongruent") ~ "responses incongruent",
          .data$congruent %in% c("cueIncongruent_responsesCongruent", "stimIncongruent_responsesCongruent") ~ "cue incongruent"
        ),
        .data$congruent %in% c("TRUE", "True", "true", "1", "T") ~ "congruent",
        .data$congruent %in% c("FALSE", "False", "false", "0", "F") ~ "incongruent",
        TRUE ~ as.character(.data$congruent)
      )
    )
  processed
}

summarise_trial_level <- function(df) {
  df %>%
    group_by(.data$condition) %>%
    summarise(
      n_trials = dplyr::n(),
      rt_mean = mean(.data$rt, na.rm = TRUE),
      rt_sd = sd(.data$rt, na.rm = TRUE),
      rt_sem = safe_sem(.data$rt),
      acc_mean = mean(.data$correct_numeric, na.rm = TRUE),
      acc_sem = safe_sem(.data$correct_numeric),
      .groups = "drop"
    )
}

summarise_block_level <- function(df) {
  df %>%
    group_by(.data$subject, .data$order, .data$condition, .data$block, .data$congruent) %>%
    summarise(
      block_rt_mean = mean(.data$rt[.data$correct_numeric == 1], na.rm = TRUE),
      block_acc_mean = mean(.data$correct_numeric, na.rm = TRUE),
      block_ies_mean = ifelse(
        mean(.data$correct_numeric, na.rm = TRUE) > 0,
        mean(.data$rt[.data$correct_numeric == 1], na.rm = TRUE) /
          mean(.data$correct_numeric, na.rm = TRUE),
        NA_real_
      ),
      block_score = {
        block_scores <- .data$score[!is.na(.data$score)]
        if (length(block_scores) == 0) NA_real_ else dplyr::last(block_scores)
      },
      .groups = "drop"
    )
}

summarise_session_level <- function(block_summary) {
  block_summary %>%
    group_by(.data$subject, .data$order, .data$condition, .data$congruent) %>%
    summarise(
      session_rt_mean = mean(.data$block_rt_mean, na.rm = TRUE),
      session_acc_mean = mean(.data$block_acc_mean, na.rm = TRUE),
      session_score_mean = mean(.data$block_score, na.rm = TRUE),
      session_ies_mean = mean(.data$block_ies_mean, na.rm = TRUE),
      blocks = dplyr::n(),
      .groups = "drop"
    )
}

# summarise_worse_better <- function(session_summary) {
#   if (nrow(session_summary) == 0) return(tibble())

#   classify_change <- function(sham, tavns, higher_is_better, tol = 1e-6) {
#     if (is.na(sham) || is.na(tavns)) return(NA_character_)
#     delta <- tavns - sham
#     if (abs(delta) <= tol) return("same")
#     improved <- if (higher_is_better) delta > 0 else delta < 0
#     ifelse(improved, "improved", "worsened")
#   }

#   wide <- session_summary %>%
#     select(dplyr::all_of(c("subject", "condition", "session_rt_mean", "session_acc_mean", "session_score_mean", "session_ies_mean"))) %>%
#     tidyr::pivot_wider(
#       names_from = .data$condition,
#       values_from = c(.data$session_rt_mean, .data$session_acc_mean, .data$session_score_mean, .data$session_ies_mean),
#       names_sep = "__"
#     )

#   metric_specs <- list(
#     list(name = "session_acc_mean", label = "Session Accuracy", higher = TRUE),
#     list(name = "session_rt_mean", label = "Session RT", higher = FALSE),
#     list(name = "session_score_mean", label = "Session Score", higher = TRUE)
#   )

#   change_rows <- lapply(metric_specs, function(spec) {
#     sham_col <- paste0(spec$name, "__", CONDITION_LEVELS[1])
#     tavns_col <- paste0(spec$name, "__", CONDITION_LEVELS[2])
#     if (!all(c(sham_col, tavns_col) %in% names(wide))) return(NULL)

#     cats <- mapply(
#       classify_change,
#       sham = wide[[sham_col]],
#       tavns = wide[[tavns_col]],
#       MoreArgs = list(higher_is_better = spec$higher),
#       SIMPLIFY = TRUE
#     )
#     cats <- cats[!is.na(cats)]
#     if (!length(cats)) return(NULL)

#     tab <- table(factor(cats, levels = c("improved", "same", "worsened")))
#     total <- sum(tab)

#     tibble(
#       metric = spec$label,
#       category = names(tab),
#       n_subjects = as.integer(tab),
#       percent = if (total > 0) as.numeric(tab) / total * 100 else NA_real_
#     )
#   })

#   change_rows <- Filter(Negate(is.null), change_rows)
#   if (!length(change_rows)) return(tibble())

#   dplyr::bind_rows(change_rows)
# }
compute_metric_differences <- function(block_summary) {
  # Compute within-block Stroop effect (incongruent − congruent) for each metric
  # Returns one row per subject × condition × block × metric with the difference
  if (nrow(block_summary) == 0) return(tibble())

  # Ensure expected columns exist (block_score is the name in block_summary)
  cols_available <- c("subject", "condition", "block", "congruent",
                      "block_rt_mean", "block_acc_mean", "block_ies_mean", "block_score")
  missing <- setdiff(cols_available, names(block_summary))
  if (length(missing)) {
    warning("compute_metric_differences: missing columns: ", paste(missing, collapse = ", "))
    return(tibble())
  }

  metrics <- c("block_rt_mean", "block_acc_mean", "block_ies_mean", "block_score")

  # Pivot wider across congruency so we can compute incongruent − congruent per block
  wide <- block_summary %>%
    select(dplyr::all_of(c("subject","order", "condition", "block", "congruent")), dplyr::all_of(metrics)) %>%
    tidyr::pivot_wider(
      names_from = .data$congruent,
      values_from = dplyr::all_of(metrics),
      names_sep = "__"
    )

  # Helper to safely fetch columns for a metric under each congruency
  get_cols <- function(metric) {
    list(
      cong = paste0(metric, "__congruent"),
      incong = paste0(metric, "__incongruent")
    )
  }

  out_rows <- lapply(metrics, function(metric) {
    cols <- get_cols(metric)
    if (!all(c(cols$cong, cols$incong) %in% names(wide))) return(NULL)

    tibble(
      subject   = wide$subject,
      order     = wide$order,
      condition = wide$condition,
      block     = wide$block,
      metric    = dplyr::recode(metric,
                                block_rt_mean  = "Block RT",
                                block_acc_mean = "Block Accuracy",
                                block_ies_mean = "Block IES",
                                block_score    = "Block Score"),
      congruent     = wide[[cols$cong]],
      incongruent   = wide[[cols$incong]],
      difference    = incongruent - congruent
    ) %>%
      dplyr::filter(stats::complete.cases(.data$congruent, .data$incongruent))
  })

  out_rows <- Filter(Negate(is.null), out_rows)
  if (!length(out_rows)) return(tibble())
  dplyr::bind_rows(out_rows)
}

summarise_subject_overview <- function(df) {
  df %>%
    group_by(.data$subject, .data$order, .data$condition) %>%
    summarise(
      blocks_list = paste(sort(unique(as.integer(as.character(.data$block)))), collapse = ","),
      n_trials = dplyr::n(),
      .groups = "drop"
    ) %>%
    arrange(.data$order, .data$subject)
}

create_trial_plots <- function(df) {
  p_rt <- ggplot(df %>% filter(.data$correct_numeric == 1),
                 aes(x = .data$condition, y = .data$rt, fill = .data$condition)) +
    geom_violin(alpha = 0.35, color = NA) +
    geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.6) +
    scale_fill_manual(values = CONDITION_COLORS) +
    labs(title = "Trial-level RT (correct trials)", x = "Condition", y = "Reaction time (s)")

  p_acc <- df %>%
    mutate(correct_percent = .data$correct_numeric * 100) %>%
    ggplot(aes(x = .data$condition, y = .data$correct_percent, fill = .data$condition)) +
    geom_bar(stat = "summary", fun = mean, position = "dodge", alpha = 0.6) +
    stat_summary(fun.data = mean_se, geom = "errorbar", width = 0.2, position = position_dodge(0.9), color = "black") +
    scale_fill_manual(values = CONDITION_COLORS) +
    labs(title = "Trial-level Accuracy", x = "Condition", y = "Percent correct")

  p_rt + p_acc + patchwork::plot_layout(guides = "collect") & theme(legend.position = "bottom")
}

create_trial_relationship_plots <- function(df) {
  metrics <- list(
    rt = list(label = "Reaction Time (s)", transform = function(x) x, kind = "rt"),
    correct_numeric = list(label = "Percent Correct", transform = function(x) x * 100, kind = "percent")
  )
  candidate_vars <- setdiff(names(df), names(metrics))
  if (length(candidate_vars) == 0) return(list())

  var_types <- vapply(candidate_vars, function(var) classify_var_type(df[[var]]), character(1))
  plots <- list()

  for (var in candidate_vars) {
    var_type <- var_types[[var]] %||% "discrete"
    for (metric_name in names(metrics)) {
      metric_info <- metrics[[metric_name]]
      metric_label <- metric_info$label
      var_label <- labelize(var)
      data_plot <- df %>% filter(!is.na(.data[[var]]))
      if (nrow(data_plot) == 0) next

      if (var_type == "continuous") {
        var_bins <- bin_continuous_var(data_plot[[var]])
        if (all(is.na(var_bins))) next
        data_plot <- data_plot %>% mutate(var_group = var_bins)
        x_label <- sprintf("%s (binned)", var_label)
      } else {
        data_plot <- data_plot %>% mutate(var_group = factor(.data[[var]]))
        x_label <- var_label
      }

      if (metric_info$kind == "rt") {
        plot_data <- data_plot %>%
          filter(!is.na(.data$rt), .data$rt > 0, !is.na(.data$var_group)) %>%
          mutate(metric_value = metric_info$transform(.data$rt))
        if (nrow(plot_data) == 0) next
        p <- ggplot(plot_data,
                    aes(x = .data$var_group, y = .data$metric_value, fill = .data$condition,
                        group = interaction(.data$var_group, .data$condition, drop = TRUE))) +
          geom_boxplot(alpha = 0.6, outlier.size = 0.5, position = position_dodge(width = 0.75)) +
          scale_fill_manual(values = CONDITION_COLORS) +
          labs(
            title = sprintf("%s by %s", metric_label, x_label),
            x = x_label,
            y = metric_label,
            fill = "Condition"
          ) +
          theme(axis.text.x = element_text(angle = 35, hjust = 1))
      } else {
        summary_data <- data_plot %>%
          mutate(metric_value = metric_info$transform(.data$correct_numeric)) %>%
          filter(!is.na(.data$metric_value), !is.na(.data$var_group)) %>%
          group_by(.data$var_group, .data$condition) %>%
          summarise(mean_value = mean(.data$metric_value, na.rm = TRUE),
                    sem_value = safe_sem(.data$metric_value), .groups = "drop")
        if (nrow(summary_data) == 0) next
        p <- ggplot(summary_data,
                    aes(x = .data$var_group, y = .data$mean_value, fill = .data$condition)) +
          geom_col(position = position_dodge(width = 0.75), alpha = 0.8) +
          geom_errorbar(aes(ymin = .data$mean_value - .data$sem_value,
                            ymax = .data$mean_value + .data$sem_value),
                        position = position_dodge(width = 0.75), width = 0.2, linewidth = 0.7) +
          scale_fill_manual(values = CONDITION_COLORS) +
          labs(
            title = sprintf("%s by %s", metric_label, x_label),
            x = x_label,
            y = metric_label,
            fill = "Condition"
          ) +
          theme(axis.text.x = element_text(angle = 35, hjust = 1))
      }

      plots[[length(plots) + 1]] <- p
    }
  }

  plots
}

create_block_plots <- function(block_summary) {
  block_long <- block_summary %>%
    pivot_longer(cols = c("block_rt_mean", "block_acc_mean", "block_score", "block_ies_mean"),
                 names_to = "metric", values_to = "value") %>%
    mutate(metric = dplyr::recode(.data$metric,
                           block_rt_mean = "Block Mean RT",
                           block_acc_mean = "Block Accuracy",
                           block_score = "Block Score",
                           block_ies_mean = "Block Mean IES"))

  block_means <- block_long %>%
    group_by(.data$condition, .data$order, .data$block, .data$metric) %>%
    summarise(mean_value = mean(.data$value, na.rm = TRUE), .groups = "drop",
              sem_value = safe_sem(.data$value))

  order_paths <- block_means %>%
    arrange(.data$order, .data$block, .data$condition)

  ggplot(block_long, aes(x = .data$block, y = .data$value, color = .data$condition, group = .data$subject)) +
    geom_line(alpha = 0.3) +
    geom_point(alpha = 0.5) +
    geom_line(data = order_paths,
        inherit.aes = FALSE,
        aes(x = .data$block, y = .data$mean_value, group = interaction(.data$order, .data$metric), linetype = .data$order),
        color = "grey25", linewidth = 1.1, alpha = 0.9) +
    # geom_point(data = block_means,
    #      inherit.aes = FALSE,
    #      aes(x = .data$block, y = .data$mean_value, color = .data$condition),
    #      size = 2.7) +
    geom_errorbar(data = block_means,
              inherit.aes = FALSE,
              aes(x = .data$block, y = .data$mean_value,
                  ymin = .data$mean_value - .data$sem_value,
                  ymax = .data$mean_value + .data$sem_value,
                  color = .data$condition),
              width = 0.2, linewidth = 0.8) +
    scale_color_manual(values = CONDITION_COLORS) +
    scale_linetype_manual(values = c(STTS = "dashed", TSST = "solid"), name = "Order") +
    facet_wrap(~metric, scales = "free_y") +
    labs(title = "Block trajectories by condition", x = "Block", y = "Value", color = "Condition")
}

summarise_block_congruency_metrics <- function(df) {
  df %>%
    filter(!is.na(.data$block), !is.na(.data$congruent)) %>%
    group_by(.data$subject, .data$condition, .data$block, .data$congruent) %>%
    summarise(
      mean_rt = mean(.data$rt[.data$correct_numeric == 1], na.rm = TRUE),
      accuracy = mean(.data$correct_numeric, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    filter(!is.na(.data$mean_rt), !is.na(.data$accuracy)) %>%
    mutate(
      block = factor(.data$block, levels = sort(unique(.data$block)), ordered = TRUE),
      congruent = factor(.data$congruent),
      accuracy_pct = .data$accuracy * 100
    )
}

create_block_rt_accuracy_plot <- function(df) {
  summary_df <- summarise_block_congruency_metrics(df)

  if (nrow(summary_df) == 0) return(NULL)

  congr_levels <- levels(summary_df$congruent)
  alpha_vals <- if (length(congr_levels)) stats::setNames(seq(0.4, 0.95, length.out = length(congr_levels)), congr_levels) else NULL

  ggplot(summary_df, aes(x = .data$accuracy_pct, y = .data$mean_rt, color = .data$condition, alpha = .data$congruent)) +
    geom_point(size = 2.3) +
    scale_color_manual(values = CONDITION_COLORS) +
    scale_alpha_manual(values = alpha_vals, guide = guide_legend(title = "Congruency")) +
    labs(
      title = "Block-level RT vs accuracy by congruency",
      x = "Accuracy (%)",
      y = "Mean RT (s)",
      color = "Condition"
    ) +
    theme_minimal()
}

create_session_plots <- function(session_summary) {
  long <- session_summary %>%
    pivot_longer(cols = c("session_rt_mean", "session_acc_mean", "session_score_mean", "session_ies_mean"),
                 names_to = "metric", values_to = "value") %>%
  mutate(metric = dplyr::recode(.data$metric,
               session_rt_mean = "Session Mean RT",
               session_acc_mean = "Session Mean Accuracy",
               session_score_mean = "Session Score",
               session_ies_mean = "Session Mean IES"))

  ggplot(long, aes(x = .data$condition, y = .data$value, color = .data$condition, group = .data$subject)) +
    geom_line(alpha = 0.4, position = position_dodge(width = 0.2)) +
    geom_point(size = 2, position = position_dodge(width = 0.2)) +
    stat_summary(aes(group = .data$condition), fun = mean, geom = "point",
                 shape = 23, size = 3, fill = "white", color = "black") +
    stat_summary(aes(group = .data$condition), fun.data = mean_se, geom = "errorbar",
                 width = 0.1, color = "black") +
    scale_color_manual(values = CONDITION_COLORS) +
    facet_wrap(~metric, scales = "free") +
    labs(title = "Subject-level summaries", x = "Condition", y = "Value", color = "Condition")
}

create_block_diff_plots <- function(block_diffs) {
  if (is.null(block_diffs) || nrow(block_diffs) == 0) return(NULL)

  block_diffs <- block_diffs %>%
    mutate(metric = factor(.data$metric, ordered=FALSE))

  block_means <- block_diffs %>%
    group_by(.data$condition, .data$order, .data$block, .data$metric) %>%
    summarise(mean_value = mean(.data$difference, na.rm = TRUE), .groups = "drop",
              sem_value = safe_sem(.data$difference))

  order_paths <- block_means %>%
    arrange(.data$order, .data$block, .data$condition)

  ggplot(block_diffs, aes(x = .data$block, y = .data$difference, color = .data$condition, group = .data$subject)) +
    geom_line(alpha = 0.3) +
    geom_point(alpha = 0.5) +
    geom_line(data = order_paths,
        inherit.aes = FALSE,
        aes(x = .data$block, y = .data$mean_value, group = interaction(.data$order, .data$metric), linetype = .data$order),
        color = "grey25", linewidth = 1.1, alpha = 0.9) +
    # geom_point(data = block_means,
    #      inherit.aes = FALSE,
    #      aes(x = .data$block, y = .data$mean_value, color = .data$condition),
    #      size = 2.7) +
    geom_errorbar(data = block_means,
              inherit.aes = FALSE,
              aes(x = .data$block, y = .data$mean_value,
                  ymin = .data$mean_value - .data$sem_value,
                  ymax = .data$mean_value + .data$sem_value,
                  color = .data$condition),
              width = 0.2, linewidth = 0.8) +
    scale_color_manual(values = CONDITION_COLORS) +
    scale_linetype_manual(values = c(STTS = "dashed", TSST = "solid"), name = "Order") +
    facet_wrap(~metric, scales = "free_y") +
    labs(title = "Block differences by condition", x = "Block", y = "Incongruent - Congruent", color = "Condition")
}

export_summary_csv <- function(trial_summary, block_summary, session_summary, path) {
  summary_block <- block_summary %>% group_by(.data$condition) %>%
    summarise(
      blocks = dplyr::n(),
      block_rt_mean = mean(.data$block_rt_mean, na.rm = TRUE),
      block_acc_mean = mean(.data$block_acc_mean, na.rm = TRUE),
      block_score_mean = mean(.data$block_score, na.rm = TRUE),
      .groups = "drop"
    )

  summary_session <- session_summary %>% group_by(.data$condition) %>%
    summarise(
      sessions = dplyr::n(),
      session_rt_mean = mean(.data$session_rt_mean, na.rm = TRUE),
      session_acc_mean = mean(.data$session_acc_mean, na.rm = TRUE),
      session_score_mean = mean(.data$session_score_mean, na.rm = TRUE),
      .groups = "drop"
    )

  combined <- trial_summary %>% mutate(level = "trial") %>%
    bind_rows(summary_block %>% mutate(level = "block")) %>%
    bind_rows(summary_session %>% mutate(level = "session"))

  readr::write_csv(combined, path)
}

run_eda <- function(experiment_type) {
  paths <- prep_output_paths(experiment_type)
  message("Running EDA for ", experiment_type)
  raw <- load_stroop_data(DATA_DIR, START_DATE, END_DATE, experiment_type)
  if (nrow(raw) == 0) {
    warning("No data for ", experiment_type)
    return(invisible(NULL))
  }
  data <- prepare_data(raw, experiment_type)

  trial_summary <- summarise_trial_level(data)
  block_summary <- summarise_block_level(data)
  session_summary <- summarise_session_level(block_summary)
  # session_change_summary <- summarise_worse_better(session_summary)
  block_differences <- compute_metric_differences(block_summary)
  subject_overview <- summarise_subject_overview(data)

  export_summary_csv(trial_summary, block_summary, session_summary, paths$summary_csv)
  readr::write_csv(block_summary, paths$block_csv)
  readr::write_csv(data, paths$trial_csv)
  readr::write_csv(subject_overview, paths$subject_csv)
  # if (!is.null(paths$session_change_csv) && nrow(session_change_summary)) {
  #   readr::write_csv(session_change_summary, paths$session_change_csv)
  # }

  trial_plot <- create_trial_plots(data)
  relationship_plots <- create_trial_relationship_plots(data)
  block_rt_accuracy_plot <- create_block_rt_accuracy_plot(data)
  block_plot <- create_block_plots(block_summary %>% filter(congruent=="incongruent"))
  session_plot <- create_session_plots(session_summary %>% filter(congruent=="incongruent"))
  block_diff_plots <- create_block_diff_plots(block_differences)

  grDevices::pdf(paths$plots_pdf, width = 11, height = 8.5)
  on.exit(grDevices::dev.off(), add = TRUE)
  print(trial_plot)
  if (length(relationship_plots)) {
    invisible(lapply(relationship_plots, print))
  }
  if (!is.null(block_rt_accuracy_plot)) print(block_rt_accuracy_plot)
  print(block_plot)
  print(session_plot)
  if (!is.null(block_diff_plots)) print(block_diff_plots)

  invisible(list(
    trial_summary = trial_summary,
    block_summary = block_summary,
    session_summary = session_summary,
    block_differences = block_differences,
    subject_overview = subject_overview
  ))
}


experiment_types <- c("SCWT", "StroopSquared")
results <- lapply(experiment_types, run_eda)

message("EDA complete. Outputs saved in ", OUTPUT_DIR)