library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

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

csv_file <- "/Users/elise/Library/CloudStorage/OneDrive-TheUniversityofColoradoDenver/Desktop/pairedStroop/analyzed-data/20250806/features-table.csv"
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

update_geom_defaults("point", list(size = 3, shape = 3, stroke = 1))


plot_subject_summary <- function(subject_summary, group_summary, col, condition_colors) {
  p <- ggplot(subject_summary, aes(x = condition)) +
    geom_line(aes(y = val, group = id), 
              alpha = 0.4, color = "gray40", linewidth=1) +
    geom_errorbar(data = group_summary,
                  aes(x = condition, ymin = mean - sem, ymax = mean + sem, color = condition),
                  width = 0.1,linewidth=1, inherit.aes = FALSE) +
    geom_point(data = group_summary,
               aes(x = condition, y = mean, color = condition),
               inherit.aes = FALSE) +
    scale_color_manual(values = condition_colors) +
  labs(title = paste(unique(subject_summary$signal_type),"_",unique(subject_summary$event),": p=", round(t_test_result$p.value, 3)), y = col)
  # Print the plot
  print(p)
}

df <- read_csv(csv_file)
df <- df %>%
  mutate(
    id = factor(id),
    experiment = factor(experiment),
    signal_type = factor(signal_type),
    order = factor(order),
    event = factor(event),
    condition = factor(condition),
    block = factor(block, ordered = TRUE),
    trial = factor(trial, ordered = TRUE),
  ) %>%
  filter(condition!="practice", event != "response_incorrect")

cols <- names(df)[!names(df) %in% c("id", "order","experiment","datetime", "signal_type", "event", "condition", "block", "trial")]
# Loop through experiments
for (exp_name in unique(df$experiment)) {
  exp_df <- df %>% filter(experiment == exp_name)
  pdf_file <- file.path(output_dir, paste0("FeaturePlots_", exp_name, ".pdf"))
  pdf(pdf_file)
  # Loop through signal types
  for (ev in unique(exp_df$event)) {
    ev_df <- exp_df %>% filter(event == ev)
    # Loop through event markers
    for (sig in unique(ev_df$signal_type)) {
      sig_df <- ev_df %>% filter(signal_type == sig) 
      for (col in cols){
        subject_mean <- sig_df %>%
          group_by(experiment, signal_type, event, id, condition) %>%
          summarise(
            val = mean(.data[[col]], na.rm = TRUE),
            .groups = "drop"
          )
        condition_mean <- subject_mean %>% 
          group_by(experiment, signal_type, event, condition) %>%
          summarise(mean = mean(val),
                    sem = sd(val)/ sqrt(n()),
                    .groups = "drop")
        
        # paired t-test on sham vs taVNS
        t_test_result <- t.test(
          subject_mean$val[subject_mean$condition == "sham"],
          subject_mean$val[subject_mean$condition == "taVNS"],
          paired = TRUE
        )

        plot_subject_summary(subject_mean, condition_mean, col, condition_colors)
      }
    }
  }
  dev.off()
}

dev.off()  # Close PDF file
  # for (col in cols) {
  #   if (col == "response_time") {
  #     break()
  #   }
  
  #   metric_df <- plrt %>%
  #     select(subject, condition, col)
  
  #   summ_df <- metric_df %>%
  #     group_by(subject, condition) %>%
  #     summarise(metric_mean = mean(.data[[col]], na.rm = TRUE),
  #               metric_sem = sd(.data[[col]], na.rm = TRUE) / sqrt(n()),
  #               .groups = "drop")
  
  #   # paired t-test on sham vs taVNS
  #   t_test_result <- t.test(
  #     summ_df$metric_mean[summ_df$condition == "sham"],
  #     summ_df$metric_mean[summ_df$condition == "taVNS"],
  #     paired = TRUE
  #   )
  
    # Create the plot