# --- Prerequisites ---
# Install and load necessary packages
# install.packages(c("ggplot2", "dplyr", "showtext", "scales", "ggtext"))

library(ggplot2)
library(dplyr)
library(showtext)
library(scales)
library(ggtext)

# Add modern, professional fonts
font_add_google("Inter", "inter")
font_add_google("JetBrains Mono", "jetbrains")
showtext_auto()

# Modern color palette
colors <- list(
  primary = "#6366f1",      # Indigo
  secondary = "#8b5cf6",    # Purple  
  accent = "#06b6d4",       # Cyan
  success = "#10b981",      # Emerald
  warning = "#f59e0b",      # Amber
  error = "#ef4444",        # Red
  dark = "#1f2937",         # Gray-800
  light = "#f8fafc"         # Gray-50
)

# --- Step 1: Input your CORRECT data from the final Python script ---
# These values are now updated to your final results
standalone_knn_acc <- 0.9492
standalone_svm_acc <- 0.9692
standalone_rf_acc <- 0.9850
standalone_gb_acc <- 0.9850

stacked_svm_acc <- 0.9683
stacked_rf_acc <- 0.9842
stacked_gb_acc <- 0.9833

# --- Step 2: Create enhanced data frames for plotting ---
standalone_df <- data.frame(
  model_name = c("KNN", "SVM", "Random Forest", "Gradient Boosting"),
  accuracy = c(standalone_knn_acc, standalone_svm_acc, standalone_rf_acc, standalone_gb_acc),
  colors = c(colors$primary, colors$secondary, colors$accent, colors$success)
) %>%
  arrange(accuracy) %>%
  mutate(model_name = factor(model_name, levels = model_name))

stacked_df <- data.frame(
  model_name = c("Stacked (KNN + SVM)", "Stacked (KNN + RF)", "Stacked (KNN + GB)"),
  baseline_acc = standalone_knn_acc,
  stacked_acc = c(stacked_svm_acc, stacked_rf_acc, stacked_gb_acc),
  comparison_group = c("SVM", "Random Forest", "Gradient Boosting"),
  improvement = c(stacked_svm_acc - standalone_knn_acc, 
                  stacked_rf_acc - standalone_knn_acc, 
                  stacked_gb_acc - standalone_knn_acc)
) %>%
  arrange(stacked_acc) %>%
  mutate(comparison_group = factor(comparison_group, levels = comparison_group))

# --- Graph 1: Ultra-Modern Lollipop Chart ---
lollipop_plot <- standalone_df %>%
  ggplot(aes(x = model_name, y = accuracy)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, 
           fill = colors$light, alpha = 0.3) +
  geom_segment(aes(xend = model_name, yend = 0), 
               color = colors$dark, size = 1.2, alpha = 0.7) +
  geom_point(aes(color = colors), size = 8, alpha = 0.9) +
  geom_point(aes(color = colors), size = 6, alpha = 1) +
  geom_point(color = "white", size = 3, alpha = 0.8) +
  geom_text(aes(label = percent(accuracy, accuracy = 0.01), color = colors), 
            vjust = -2, size = 5, family = "jetbrains", fontface = "bold") +
  scale_color_identity() +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0, 1.05),
    expand = c(0, 0),
    breaks = seq(0, 1, 0.2)
  ) +
  coord_flip() +
  labs(
    title = "<span style='color:#1f2937; font-size:28px'><b>ML Model Performance</b></span>",
    subtitle = "<span style='color:#6b7280; font-size:16px'>Standalone classifier accuracy comparison</span>",
    x = "",
    y = "**Accuracy**",
    caption = "Higher accuracy indicates better model performance"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    text = element_text(family = "inter", color = colors$dark),
    plot.title = element_markdown(margin = margin(b = 10)),
    plot.subtitle = element_markdown(margin = margin(b = 20)),
    plot.caption = element_text(color = "#6b7280", size = 11),
    axis.title.y = element_markdown(size = 16, margin = margin(r = 15)),
    axis.text.y = element_text(size = 14, face = "bold", color = colors$dark),
    axis.text.x = element_text(size = 12, color = colors$dark),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "white", linewidth = 0.8),
    plot.background = element_rect(fill = colors$light, color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(25, 25, 25, 25)
  )

# --- Graph 2: Ultra-Modern Dumbbell Plot with Gradient Effects ---
dumbbell_plot <- stacked_df %>%
  ggplot(aes(y = comparison_group)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, 
           fill = colors$light, alpha = 0.3) +
  geom_segment(aes(x = baseline_acc, xend = stacked_acc, yend = comparison_group),
               color = colors$primary, linewidth = 3, alpha = 0.6) +
  geom_segment(aes(x = baseline_acc, xend = stacked_acc, yend = comparison_group),
               color = colors$primary, linewidth = 1.5, alpha = 0.9) +
  geom_point(aes(x = baseline_acc), color = colors$warning, size = 8, alpha = 0.7) +
  geom_point(aes(x = baseline_acc), color = colors$warning, size = 6) +
  geom_point(aes(x = baseline_acc), color = "white", size = 3, alpha = 0.9) +
  geom_point(aes(x = stacked_acc), color = colors$success, size = 8, alpha = 0.7) +
  geom_point(aes(x = stacked_acc), color = colors$success, size = 6) +
  geom_point(aes(x = stacked_acc), color = "white", size = 3, alpha = 0.9) +
  geom_text(aes(x = baseline_acc, label = percent(baseline_acc, 0.01)),
            color = colors$warning, vjust = -2, hjust = 0.5, 
            family = "jetbrains", fontface = "bold", size = 4.5) +
  geom_text(aes(x = stacked_acc, label = percent(stacked_acc, 0.01)),
            color = colors$success, vjust = -2, hjust = 0.5,
            family = "jetbrains", fontface = "bold", size = 4.5) +
  geom_text(aes(x = (baseline_acc + stacked_acc) / 2, 
                label = ifelse(improvement > 0, 
                               paste0("+", percent(improvement, 0.01)), 
                               percent(improvement, 0.01))),
            color = colors$primary, vjust = 1.5, hjust = 0.5,
            family = "inter", fontface = "bold", size = 3.8) +
  scale_x_continuous(
    labels = percent_format(accuracy = 0.1),
    limits = c(0.94, 1.0),
    expand = c(0, 0),
    breaks = seq(0.94, 1.0, 0.02)
  ) +
  labs(
    title = "<span style='color:#1f2937; font-size:28px'><b>Stacking Impact Analysis</b></span>",
    subtitle = "<span style='color:#6b7280; font-size:16px'>KNN baseline <span style='color:#f59e0b'>●</span> vs Stacked performance <span style='color:#10b981'>●</span></span>",
    x = "**Model Accuracy**",
    y = "**Stacking Partner**",
    caption = "Numbers show improvement from baseline KNN accuracy"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    text = element_text(family = "inter", color = colors$dark),
    plot.title = element_markdown(margin = margin(b = 10)),
    plot.subtitle = element_markdown(margin = margin(b = 20)),
    plot.caption = element_text(color = "#6b7280", size = 11),
    axis.title.x = element_markdown(size = 16, margin = margin(t = 15)),
    axis.title.y = element_markdown(size = 16, margin = margin(r = 15)),
    axis.text.y = element_text(size = 14, face = "bold", color = colors$dark),
    axis.text.x = element_text(size = 12, color = colors$dark),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "white", linewidth = 0.8),
    plot.background = element_rect(fill = colors$light, color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(25, 25, 25, 25)
  )

# --- Graph 3: Modern Comparison Heatmap (Bonus visualization) ---
comparison_matrix <- expand.grid(
  Model1 = c("KNN", "SVM", "RF", "GB"),
  Model2 = c("KNN", "SVM", "RF", "GB")
) %>%
  mutate(
    Accuracy1 = case_when(
      Model1 == "KNN" ~ standalone_knn_acc,
      Model1 == "SVM" ~ standalone_svm_acc,
      Model1 == "RF" ~ standalone_rf_acc,
      Model1 == "GB" ~ standalone_gb_acc
    ),
    Accuracy2 = case_when(
      Model2 == "KNN" ~ standalone_knn_acc,
      Model2 == "SVM" ~ standalone_svm_acc,
      Model2 == "RF" ~ standalone_rf_acc,
      Model2 == "GB" ~ standalone_gb_acc
    ),
    Difference = Accuracy2 - Accuracy1
  )

heatmap_plot <- comparison_matrix %>%
  ggplot(aes(x = Model1, y = Model2, fill = Difference)) +
  geom_tile(color = "white", linewidth = 2) +
  geom_text(aes(label = percent(abs(Difference), 0.01)), 
            family = "jetbrains", fontface = "bold", size = 4) +
  scale_fill_gradient2(
    low = colors$error, mid = "white", high = colors$success,
    midpoint = 0, name = "Accuracy\nDifference",
    labels = percent_format(accuracy = 0.1)
  ) +
  labs(
    title = "<span style='color:#1f2937; font-size:28px'><b>Model Comparison Matrix</b></span>",
    subtitle = "<span style='color:#6b7280; font-size:16px'>Pairwise accuracy differences between models</span>",
    x = "**Base Model**",
    y = "**Comparison Model**"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    text = element_text(family = "inter", color = colors$dark),
    plot.title = element_markdown(margin = margin(b = 10)),
    plot.subtitle = element_markdown(margin = margin(b = 20)),
    axis.title.x = element_markdown(size = 16, margin = margin(t = 15)),
    axis.title.y = element_markdown(size = 16, margin = margin(r = 15)),
    axis.text = element_text(size = 12, face = "bold"),
    legend.title = element_text(face = "bold"),
    panel.grid = element_blank(),
    plot.background = element_rect(fill = colors$light, color = NA)
  )

# --- Display the plots ---
print("Displaying Modern Lollipop Plot...")
print(lollipop_plot)

print("Displaying Modern Dumbbell Plot...")
print(dumbbell_plot)

print("Displaying Bonus Heatmap...")
print(heatmap_plot)

# --- Save high-quality plots ---
dir.create("results", showWarnings = FALSE)

ggsave("results/ultra_modern_standalone_plot.png", 
       plot = lollipop_plot, width = 12, height = 8, dpi = 300, bg = colors$light)
ggsave("results/ultra_modern_stacked_plot.png", 
       plot = dumbbell_plot, width = 12, height = 8, dpi = 300, bg = colors$light)
ggsave("results/modern_comparison_heatmap.png", 
       plot = heatmap_plot, width = 10, height = 8, dpi = 300, bg = colors$light)

cat("\n✨ All plots saved successfully with the LATEST data! ✨\n")


