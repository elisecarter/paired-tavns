library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Set global theme
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

csv_file <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/pairedStroop/analyzed-data/20250818/epochs-table.csv"
output_dir <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/pairedStroop/analyzed-data"  # nolint

# Create today's output folder if it doesn't exist
today <- format(Sys.Date(), "%Y%m%d")
output_dir <- file.path(output_dir, today)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

condition_colors <- c("#68689a", "#6666ff")
names(condition_colors) <- c("sham", "taVNS")
condition_colors


#------------------------------------------------------------
# Function to plot time series for each signal/event/condition
plot_signal_timeseries <- function(df) {
  signal <- unique(df$signal_type)
  if (signal == "pupilDiameter") {
    ymin <- -4
    ymax <- 4
  } else if (signal == "EDA") {
    ymin <- -10
    ymax <- 10
  } else {
    ymin <- -10
    ymax = 10
  }
  p <- ggplot(df, aes(x = as.numeric(as.character(time)), y = mean, color = condition, group = condition)) +
    geom_line() +
    geom_ribbon(aes(ymin = mean - sem, ymax = mean + sem, fill = condition), alpha=0.5, color = NA) +
    scale_color_manual(values = condition_colors) +
    scale_fill_manual(values = condition_colors) +
    # ylim(ymin,ymax) +
    labs(title = paste(unique(df$experiment),': ', unique(df$event)),
         x = "Time (s)", y = paste(unique(df$signal_type), '(z)'))
  print(p)
}

#------------------------------------------------------------
# Main execution function
run_signal_analysis <- function(csv_file) {
  df_raw <- read_csv(csv_file, show_col_types = FALSE)
  if (nrow(df_raw) == 0) {
    warning("No data loaded from epochs-table.csv")
    return(NULL)
  }
  df_raw <- df_raw %>%
    mutate(
      experiment = factor(experiment),
      signal_type = factor(signal_type),
      event = factor(event),
      condition = factor(condition),
      block = factor(block, ordered = TRUE),
      trial = factor(trial, ordered = TRUE),
      time = factor(time, ordered=TRUE)
    ) %>%
    filter(condition!="practice")
  
  # Loop through experiments
  for (exp_name in unique(df_raw$experiment)) {
    exp_df <- df_raw %>% filter(experiment == exp_name)
    pdf_file <- file.path(output_dir, paste0("timeseriesPlots_", exp_name, ".pdf"))
    pdf(pdf_file)
    # Loop through signal types
    for (ev in unique(exp_df$event)) {
      ev_df <- exp_df %>% filter(event == ev)
      # Loop through event markers
      for (sig in unique(ev_df$signal_type)) {
        sig_df <- ev_df %>% filter(signal_type == sig) 
        subject_mean <- sig_df %>%
            group_by(experiment, signal_type, event, id, condition, time) %>%
            summarise(
                val = mean(value), 
                .groups = "drop"
            )
        condition_mean <- subject_mean %>% 
          group_by(experiment, signal_type, event, condition, time) %>%
          summarise(mean = mean(val),
                    sem = sd(val)/ sqrt(n()),
                    .groups = "drop")
        
        plot_signal_timeseries(condition_mean)
      }
    }
  dev.off()
  }
}

#------------------------------------------------------------
# Run analysis
report_file <- file.path(output_dir, paste0("StatisticalReport.txt"))
run_signal_analysis(csv_file)