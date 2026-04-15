#!/usr/bin/env Rscript

# Plot 150% PT current (mA) and perceived rating (0-10) for taVNS vs sham
# with one point per participant.
# Run from repository root: Rscript plotStimIntensity.R

suppressPackageStartupMessages({
    library(jsonlite)
    library(dplyr)
    library(purrr)
    library(ggplot2)
})


# color palette
condition_colors <- c("#6dc8bf", "#f15a22")
names(condition_colors) <- c("sham", "taVNS")


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
update_geom_defaults("point", list(size = 3, shape = 3, stroke = 1))


`%||%` <- function(x, y) if (is.null(x)) y else x

data_dir <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/paired-tavns-analysis/Data"    
target_freq <- "30"             # change to "100" if needed
keep_latest_session <- TRUE     # one point per participant/condition

cfg_files <- list.files(
    path = data_dir,
    pattern = "^session_config\\.json$",
    recursive = TRUE,
    full.names = TRUE
)

if (length(cfg_files) == 0) {
    stop("No session_config.json files found under 'Data/'.")
}

extract_calibration_points <- function(path, freq = "30") {
    cfg <- tryCatch(fromJSON(path, simplifyVector = FALSE), error = function(e) NULL)
    if (is.null(cfg)) return(NULL)

    cal <- cfg[["calibration results"]] %||% cfg[["calibration_results"]]
    if (is.null(cal)) return(NULL)

    # fallback to first available frequency if target not present
    if (is.null(cal[[freq]])) {
        freq <- names(cal)[1]
        if (is.null(freq) || is.na(freq)) return(NULL)
    }

    participant <- cfg[["ID"]] %||% basename(dirname(dirname(path)))
    session_date <- cfg[["date"]] %||% basename(dirname(path))

    map_dfr(c("taVNS", "sham"), function(cond) {
        currents <- cal[[freq]][[cond]][["calculated_currents"]] %||% list()
        perceived <- cal[[freq]][[cond]][["perceived rating"]] %||% cal[[freq]][[cond]][["perceived_rating"]] %||% list()

        percentages <- sort(unique(c(names(currents), names(perceived))))
        if (length(percentages) == 0) return(tibble())

        map_dfr(percentages, function(pct) {
            tibble(
                participant = as.character(participant),
                date = as.character(session_date),
                frequency = as.character(freq),
                condition = cond,
                percentage = as.character(pct),
                current_mA = suppressWarnings(as.numeric(currents[[pct]] %||% NA)),
                perceived_rating = suppressWarnings(as.numeric(perceived[[pct]] %||% NA))
            )
        })
    })
}

df_all <- map_dfr(cfg_files, extract_calibration_points, freq = target_freq) %>%
    filter(!(is.na(current_mA) & is.na(perceived_rating)))

if (nrow(df_all) == 0) {
    stop("No usable calibration values found.")
}

# Keep one data point per participant/condition (latest session)
if (keep_latest_session) {
    df_all <- df_all %>%
        mutate(date_num = suppressWarnings(as.numeric(date))) %>%
        group_by(participant, condition, percentage) %>%
        slice_max(order_by = date_num, n = 1, with_ties = FALSE) %>%
        ungroup() %>%
        select(-date_num)
}

participants_included <- df_all %>%
    distinct(.data$participant, .data$date, .data$condition) %>%
    mutate(condition = factor(.data$condition, levels = c("sham", "taVNS"))) %>%
    arrange(.data$participant, .data$date, .data$condition)

cat("\nParticipants included in analysis (ID, session date, condition):\n")
print(participants_included)

df_150 <- df_all %>% filter(.data$percentage == "150%")

if (nrow(df_150) == 0) {
    stop("No usable 150% PT calibration values found.")
}

rating_150_by_condition <- df_150 %>%
    filter(!is.na(.data$perceived_rating)) %>%
    group_by(.data$condition) %>%
    summarize(
        n = n(),
        mean_rating = mean(.data$perceived_rating, na.rm = TRUE),
        min_rating = min(.data$perceived_rating, na.rm = TRUE),
        max_rating = max(.data$perceived_rating, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    mutate(condition = factor(.data$condition, levels = c("sham", "taVNS"))) %>%
    arrange(.data$condition)

if (nrow(rating_150_by_condition) > 0) {
    cat("\n150% perceived rating summary by condition:\n")
    print(rating_150_by_condition)
} else {
    message("No non-missing 150% perceived_rating values found for summary.")
}

current_150_by_condition <- df_150 %>%
    filter(!is.na(.data$current_mA)) %>%
    group_by(.data$condition) %>%
    summarize(
        n = n(),
        mean_current_mA = mean(.data$current_mA, na.rm = TRUE),
        min_current_mA = min(.data$current_mA, na.rm = TRUE),
        max_current_mA = max(.data$current_mA, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    mutate(condition = factor(.data$condition, levels = c("sham", "taVNS"))) %>%
    arrange(.data$condition)

if (nrow(current_150_by_condition) > 0) {
    cat("\n150% current (mA) summary by condition:\n")
    print(current_150_by_condition)
} else {
    message("No non-missing 150% current_mA values found for summary.")
}

df_125 <- df_all %>% filter(.data$percentage == "125%")

make_plot_df <- function(data, value_col) {
    data_non_na <- data %>% filter(!is.na(.data[[value_col]]))

    paired_ids <- data_non_na %>%
        count(.data$participant) %>%
        filter(n >= 2) %>%
        pull(.data$participant)

    data_non_na %>%
        filter(.data$participant %in% paired_ids) %>%
        mutate(condition = factor(.data$condition, levels = c("sham", "taVNS")))
}

plot_paired <- function(data, y_col, y_label, y_limits, output_file, discrete_x = FALSE) {
    df_plot <- make_plot_df(data, y_col)

    if (nrow(df_plot) == 0) {
        stop(paste0("No participants with both taVNS and sham at 150% PT for ", y_col, "."))
    }

    paired_vals <- df_plot %>%
        select(.data$participant, .data$condition, value = .data[[y_col]]) %>%
        group_by(.data$participant) %>%
        summarize(
            sham = .data$value[.data$condition == "sham"][1],
            taVNS = .data$value[.data$condition == "taVNS"][1],
            .groups = "drop"
        ) %>%
        filter(!is.na(.data$sham) & !is.na(.data$taVNS))

    test_res <- tryCatch(
        wilcox.test(paired_vals$taVNS, paired_vals$sham, paired = TRUE, exact = FALSE),
        error = function(e) NULL
    )
    p_val_txt <- if (is.null(test_res)) "NA" else format.pval(test_res$p.value, digits = 3, eps = 0.001)

    group_means <- df_plot %>%
        group_by(.data$condition) %>%
        summarize(
            n = sum(!is.na(.data[[y_col]])),
            mean_value = mean(.data[[y_col]], na.rm = TRUE),
            .groups = "drop"
        )

    mean_sham <- group_means %>% filter(.data$condition == "sham") %>% pull(.data$mean_value)
    mean_taVNS <- group_means %>% filter(.data$condition == "taVNS") %>% pull(.data$mean_value)
    n_sham <- group_means %>% filter(.data$condition == "sham") %>% pull(.data$n)
    n_taVNS <- group_means %>% filter(.data$condition == "taVNS") %>% pull(.data$n)
    subtitle_txt <- paste0(
        "sham: mean = ", format(round(mean_sham, 2), nsmall = 2), ", n = ", n_sham,
        "\n",
        "taVNS: mean = ", format(round(mean_taVNS, 2), nsmall = 2), ", n = ", n_taVNS
    )

    if (discrete_x) {
        p <- ggplot(df_plot, aes(x = .data[[y_col]], y = after_stat(density), fill = .data$condition)) +
            geom_histogram(alpha = 0.5, binwidth = 1, boundary = -0.5, position = "identity", color = "white") +
            scale_fill_manual(values = condition_colors) +
            scale_color_manual(values = condition_colors) +
            scale_x_continuous(limits = y_limits, breaks = seq(ceiling(y_limits[1]), floor(y_limits[2]), by = 1)) +
            labs(
                title = paste0("p = ", p_val_txt),
                subtitle = subtitle_txt,
                x = y_label,
                y = "Density"
            ) +
            theme_classic(base_size = 12) +
            theme(legend.position = "top")
    } else {
        p <- ggplot(df_plot, aes(x = .data[[y_col]], y = after_stat(density), fill = .data$condition)) +
            geom_histogram(alpha = 0.5, bins = 10, position = "identity", color = "white") +
            scale_fill_manual(values = condition_colors) +
            scale_color_manual(values = condition_colors) +
            labs(
                title = paste0("p = ", p_val_txt),
                subtitle = subtitle_txt,
                x = y_label,
                y = "Density"
            ) +
            xlim(y_limits[1], y_limits[2]) +
            theme_classic(base_size = 12) +
            theme(legend.position = "top")
    }

    print(p)
    ggsave(output_file, p, width = 3, height = 4, dpi = 300)
    cat("Saved plot: ", output_file, "\n", sep = "")
}

plot_paired(
    data = df_150,
    y_col = "current_mA",
    y_label = "Current (mA)",
    y_limits = c(0, 5),
    output_file = "plotStimIntensity_150PT.png",
    discrete_x = FALSE
)

plot_paired(
    data = df_150,
    y_col = "perceived_rating",
    y_label = "Perceived Rating",
    y_limits = c(0, 10),
    output_file = "plotStimPerceivedRating_150PT.png",
    discrete_x = TRUE
)

if (nrow(df_125) > 0) {
    plot_paired(
        data = df_125,
        y_col = "current_mA",
        y_label = "Current (mA)",
        y_limits = c(0, 5),
        output_file = "plotStimIntensity_125PT.png",
        discrete_x = FALSE
    )

    plot_paired(
        data = df_125,
        y_col = "perceived_rating",
        y_label = "Perceived Rating",
        y_limits = c(0, 10),
        output_file = "plotStimPerceivedRating_125PT.png",
        discrete_x = TRUE
    )
} else {
    message("No usable 125% PT calibration values found; skipping 125% histograms.")
}

df_line <- df_all %>%
    filter(!is.na(.data$current_mA), !is.na(.data$perceived_rating)) %>%
    mutate(condition = factor(.data$condition, levels = c("sham", "taVNS")))

if (nrow(df_line) == 0) {
    stop("No usable current-perceived pairs found for line plot.")
}

p_line <- ggplot(df_line, aes(x = .data$current_mA, y = .data$perceived_rating, color = .data$condition)) +
    geom_line(aes(group = interaction(.data$participant, .data$condition)), alpha = 0.25, linewidth = 0.6) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 1.4) +
    scale_color_manual(values = condition_colors) +
    labs(
        x = "Current (mA)",
        y = "Perceived Rating",
        color = "Condition"
    ) +
    ylim(0, 10) +
    geom_hline(yintercept = 7, linetype = "dashed", color = "grey50") +
    theme_classic(base_size = 12)

print(p_line)
ggsave("plotCurrentVsPerceived_byCondition.png", p_line, width = 4, height = 3, dpi = 300)
cat("Saved plot: plotCurrentVsPerceived_byCondition.png\n")

df_percent <- df_all %>%
    filter(!is.na(.data$perceived_rating), !is.na(.data$percentage)) %>%
    mutate(
        condition = factor(.data$condition, levels = c("sham", "taVNS")),
        percentage_num = suppressWarnings(as.numeric(gsub("%", "", .data$percentage)))
    ) %>%
    filter(!is.na(.data$percentage_num))

if (nrow(df_percent) == 0) {
    stop("No usable perceived-rating-by-percent data found.")
}

percent_summary <- df_percent %>%
    group_by(.data$condition, .data$percentage_num) %>%
    summarize(
        n = sum(!is.na(.data$perceived_rating)),
        mean_perceived_rating = mean(.data$perceived_rating, na.rm = TRUE),
        sd_perceived_rating = sd(.data$perceived_rating, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    mutate(
        se = .data$sd_perceived_rating / sqrt(.data$n),
        t_crit = qt(0.975, pmax(.data$n - 1, 1)),
        ci_half = dplyr::if_else(.data$n > 1, .data$t_crit * .data$se, 0),
        ci_low = .data$mean_perceived_rating - .data$ci_half,
        ci_high = .data$mean_perceived_rating + .data$ci_half
    )

p_percent <- ggplot() +
    geom_line(
        data = df_percent,
        aes(
            x = .data$percentage_num,
            y = .data$perceived_rating,
            color = .data$condition,
            group = interaction(.data$participant, .data$condition)
        ),
        alpha = 0.25,
        linewidth = 0.6
    ) +

    geom_line(
        data = percent_summary,
        aes(x = .data$percentage_num, y = .data$mean_perceived_rating, color = .data$condition, group = .data$condition),
        linewidth = 1.2
    ) +
    scale_color_manual(values = condition_colors) +
    scale_fill_manual(values = condition_colors) +
    scale_x_continuous(breaks = sort(unique(percent_summary$percentage_num))) +
    geom_hline(yintercept = 7, linetype = "dashed", color = "grey50") +
    labs(
        x = "Stimulation (% of PT)",
        y = "Perceived Rating",
        color = "Condition"
    ) +
    ylim(0, 10) +
    theme_classic(base_size = 12) +
    guides(fill = "none")

print(p_percent)
ggsave("plotPerceivedRating_byPercent.png", p_percent, width = 4, height = 3, dpi = 300)
cat("Saved plot: plotPerceivedRating_byPercent.png\n")

df_current_percent <- df_all %>%
    filter(!is.na(.data$current_mA), !is.na(.data$percentage)) %>%
    mutate(
        condition = factor(.data$condition, levels = c("sham", "taVNS")),
        percentage_num = suppressWarnings(as.numeric(gsub("%", "", .data$percentage)))
    ) %>%
    filter(!is.na(.data$percentage_num))

if (nrow(df_current_percent) == 0) {
    stop("No usable current-by-percent data found.")
}

current_percent_summary <- df_current_percent %>%
    group_by(.data$condition, .data$percentage_num) %>%
    summarize(
        n = sum(!is.na(.data$current_mA)),
        mean_current_mA = mean(.data$current_mA, na.rm = TRUE),
        sd_current_mA = sd(.data$current_mA, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    mutate(
        se = .data$sd_current_mA / sqrt(.data$n),
        t_crit = qt(0.975, pmax(.data$n - 1, 1)),
        ci_half = dplyr::if_else(.data$n > 1, .data$t_crit * .data$se, 0),
        ci_low = .data$mean_current_mA - .data$ci_half,
        ci_high = .data$mean_current_mA + .data$ci_half
    )

p_current_percent <- ggplot() +
    geom_line(
        data = df_current_percent,
        aes(
            x = .data$percentage_num,
            y = .data$current_mA,
            color = .data$condition,
            group = interaction(.data$participant, .data$condition)
        ),
        alpha = 0.25,
        linewidth = 0.6
    ) +
    geom_line(
        data = current_percent_summary,
        aes(x = .data$percentage_num, y = .data$mean_current_mA, color = .data$condition, group = .data$condition),
        linewidth = 1.2
    ) +
    scale_color_manual(values = condition_colors) +
    scale_fill_manual(values = condition_colors) +
    scale_x_continuous(breaks = sort(unique(current_percent_summary$percentage_num))) +
    labs(
        x = "Stimulation (% of PT)",
        y = "Current (mA)",
        color = "Condition"
    ) +
    ylim(0, 5) +
    theme_classic(base_size = 12) +
    guides(fill = "none")

print(p_current_percent)
ggsave("plotCurrent_byPercent.png", p_current_percent, width = 4, height = 3, dpi = 300)
cat("Saved plot: plotCurrent_byPercent.png\n")