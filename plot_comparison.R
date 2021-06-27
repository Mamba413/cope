rm(list = ls()); gc(reset = TRUE)
# plot_path <- "yourpath"; setwd(plot_path)
library(reshape2)
library(ggplot2)
library(ggpubr)

u_method_type <- c("REG", "REG-M",
                   "DRL", "DRL-M", "MIS", "MIS-M", "COPE")
method_type <- factor(rep(u_method_type, 2),
                      levels = u_method_type)

assessment_type <- rep(c("Bias", "MSE"), each = length(u_method_type))

result_summary1 <- function(reg_1, reg_2, dr, drl, mis, trl, trim = 0.00) {
  root_experiment_rep <- sqrt(ncol(reg_1))
  
  bias_tab <- rbind(apply(reg_1, 1, mean, trim = trim),
                    apply(reg_2, 1, mean, trim = trim),
                    apply(dr, 1, mean, trim = trim),
                    apply(drl, 1, mean, trim = trim),
                    apply(mis, 1, mean, trim = trim),
                    apply(trl, 1, mean, trim = trim))
  
  sd_tab <- rbind(apply(reg_1, 1, sd),
                  apply(reg_2, 1, sd),
                  apply(dr, 1, sd),
                  apply(drl, 1, sd),
                  apply(mis, 1, sd),
                  apply(trl, 1, sd))
  sd_tab <- sd_tab / root_experiment_rep
  
  mse <- function(x, trim = 0) {
    if (trim > 0) {
      q1 <- quantile(x, trim)
      q2 <- quantile(x, 1 - trim)
      x <- x[x >= q1 & x <= q2]
    }
    mean(x^2)
  }
  
  mse_tab <- rbind(apply(reg_1, 1, mse, trim = trim),
                   apply(reg_2, 1, mse, trim = trim),
                   apply(dr, 1, mse, trim = trim),
                   apply(drl, 1, mse, trim = trim),
                   apply(mis, 1, mse, trim = trim),
                   apply(trl, 1, mse, trim = trim))
  
  mssd_tab <- rbind(apply(reg_1^2, 1, sd),
                    apply(reg_2^2, 1, sd),
                    apply(dr^2, 1, sd),
                    apply(drl^2, 1, sd),
                    apply(mis^2, 1, sd),
                    apply(trl^2, 1, sd))
  mssd_tab <- mssd_tab / root_experiment_rep
  
  list(bias_tab, sd_tab, mse_tab, mssd_tab)
}

result_summary2 <- function(reg_1, reg_1_wm, dr, drl, mis, mis_wm, trl, trim = 0.00) {
  root_experiment_rep <- sqrt(ncol(reg_1))
  
  bias_tab <- rbind(apply(abs(reg_1), 1, mean, trim = trim),
                    apply(abs(reg_1_wm), 1, mean, trim = trim),
                    apply(abs(dr), 1, mean, trim = trim),
                    apply(abs(drl), 1, mean, trim = trim),
                    apply(abs(mis), 1, mean, trim = trim),
                    apply(abs(mis_wm), 1, mean, trim = trim),
                    apply(abs(trl), 1, mean, trim = trim))
  
  sd_tab <- rbind(apply(abs(reg_1), 1, sd),
                  apply(abs(reg_1_wm), 1, sd),
                  apply(abs(dr), 1, sd),
                  apply(abs(drl), 1, sd),
                  apply(abs(mis), 1, sd),
                  apply(abs(mis_wm), 1, sd),
                  apply(abs(trl), 1, sd))
  sd_tab <- sd_tab / root_experiment_rep
  
  mse <- function(x, trim = 0) {
    if (trim > 0) {
      q1 <- quantile(x, trim)
      q2 <- quantile(x, 1 - trim)
      x <- x[x >= q1 & x <= q2]
    }
    mean(x^2)
  }
  
  mse_tab <- rbind(apply(reg_1, 1, mse, trim = trim),
                   apply(reg_1_wm, 1, mse, trim = trim),
                   apply(dr, 1, mse, trim = trim),
                   apply(drl, 1, mse, trim = trim),
                   apply(mis, 1, mse, trim = trim),
                   apply(mis_wm, 1, mse, trim = trim),
                   apply(trl, 1, mse, trim = trim))
  
  mssd_tab <- rbind(apply(reg_1^2, 1, sd),
                    apply(reg_1_wm^2, 1, sd),
                    apply(dr^2, 1, sd),
                    apply(drl^2, 1, sd),
                    apply(mis^2, 1, sd),
                    apply(mis_wm^2, 1, sd),
                    apply(trl^2, 1, sd))
  mssd_tab <- mssd_tab / root_experiment_rep
  
  list(bias_tab, sd_tab, mse_tab, mssd_tab)
}

# trajectory_num_vec <- c(60, 100, 140, 180, 220)
trajectory_num_vec <- c(20, 40, 80, 160, 320)
trajectory_num_vec_len <- length(trajectory_num_vec)

setwd("result-20210521/")

reg <- read.csv("trajectory-reg-1.csv", header = FALSE)
regwm <- read.csv("trajectory-regwm-1.csv", header = FALSE)
drl <- read.csv("trajectory-drl-1.csv", header = FALSE)
drlwm <- read.csv("trajectory-drlwm-1.csv", header = FALSE)
mis <- read.csv("trajectory-is-1.csv", header = FALSE)
miswm <- read.csv("trajectory-iswm-1.csv", header = FALSE)
trl <- read.csv("trajectory-trl-1.csv", header = FALSE)
res_trajectory <- result_summary2(reg, regwm, drl, drlwm, mis, miswm, trl)

reg <- read.csv("time-reg-1.csv", header = FALSE)
regwm <- read.csv("time-regwm-1.csv", header = FALSE)
drl <- read.csv("time-drl-1.csv", header = FALSE)
drlwm <- read.csv("time-drlwm-1.csv", header = FALSE)
mis <- read.csv("time-is-1.csv", header = FALSE)
miswm <- read.csv("time-iswm-1.csv", header = FALSE)
trl <- read.csv("time-trl-1.csv", header = FALSE)
res_time <- result_summary2(reg, regwm, drl, drlwm, mis, miswm, trl)

process_result <- function(mean_tab, sd_tab, row_name, col_name) {
  factored_row_name <- factor(row_name, levels = row_name)
  
  pdat1 <- as.data.frame(mean_tab)
  colnames(pdat1) <- col_name
  pdat1[["Method"]] <- factored_row_name
  pdat1 <- melt(pdat1, c("Method"))
  
  pdat2 <- as.data.frame(sd_tab)
  colnames(pdat2) <- col_name
  pdat2[["Method"]] <- factored_row_name
  pdat2 <- melt(pdat2, c("Method"))
  
  pdat1[["sd_value"]] <- pdat2[["value"]]
  pdat1[["variable"]] <- as.numeric(as.character(pdat1[["variable"]]))
  pdat1[["ci_min"]] <- pdat1[["value"]] - 2 * pdat1[["sd_value"]]
  pdat1[["ci_max"]] <- pdat1[["value"]] + 2 * pdat1[["sd_value"]]
  
  pdat1
}

pdat_1_1 <- process_result(abs(res_trajectory[[1]]), 
                           res_trajectory[[2]], 
                           row_name = u_method_type, 
                           col_name = trajectory_num_vec)
pdat_1_2 <- process_result(res_trajectory[[3]], 
                           res_trajectory[[4]], 
                           row_name = u_method_type, 
                           col_name = trajectory_num_vec)
pdat_2_1 <- process_result(abs(res_time[[1]]), res_time[[2]], 
                           row_name = u_method_type, 
                           col_name = trajectory_num_vec)
pdat_2_2 <- process_result(res_time[[3]], res_time[[4]], 
                           row_name = u_method_type, 
                           col_name = trajectory_num_vec)

pdat <- rbind.data.frame(cbind.data.frame(pdat_1_1, type = "logBias", class = "Trajectory"), 
                         cbind.data.frame(pdat_1_2, type = "logMSE", class = "Trajectory"), 
                         cbind.data.frame(pdat_2_1, type = "logBias", class = "Time"), 
                         cbind.data.frame(pdat_2_2, type = "logMSE", class = "Time"))

p1 <- ggplot(pdat, aes(x = variable, y = log10(value), color = Method)) + 
  facet_grid(class ~ type, scales = "free_y", switch = "y") + 
  geom_point(aes(group = Method), size = 2.5) +
  geom_line(aes(group = Method), size = 1) +
  geom_errorbar(aes(ymin = log10(ci_min), ymax = log10(ci_max)), size = 1, width = 0.1) + 
  scale_x_continuous(breaks = trajectory_num_vec, trans = "log2") +
  scale_y_continuous(n.breaks = 6) +
  theme_bw() + 
  xlab("") + ylab("") + 
  theme(
    # legend.position = "bottom", 
        legend.box.margin = margin(t = -10, b = 10, r = 0, l = 0), 
        strip.text.y = element_blank(),
        plot.margin = unit(c(5.5, 5.5, 0.0, 0.0), "pt"), 
        # axis.title.x = element_blank(),
        # axis.text.x = element_blank(),
        # axis.ticks.x = element_blank()
        ) + 
  guides(guide_legend(byrow = TRUE))
p1
# ggsave(filename = "comparison.jpg", plot = p1, width = 6, height = 5.6)
# ggsave(filename = "comparison.eps", plot = p1, width = 6, height = 5.6)

##################################################
#################### Coverage ####################
##################################################
result_summary3 <- function(reg_1, reg_1_wm, dr, drl, mis, mis_wm, trl, trim = 0.00) {
  cover_tab <- rbind(apply(reg_1, 1, mean, trim = trim),
                     apply(reg_1_wm, 1, mean, trim = trim),
                     apply(dr, 1, mean, trim = trim),
                     apply(drl, 1, mean, trim = trim),
                     apply(mis, 1, mean, trim = trim),
                     apply(mis_wm, 1, mean, trim = trim),
                     apply(trl, 1, mean, trim = trim))
  cover_tab
}

reg <- read.csv("trajectory-reg-cover-1.csv", header = FALSE)
regwm <- read.csv("trajectory-regwm-cover-1.csv", header = FALSE)
drl <- read.csv("trajectory-drl-cover-1.csv", header = FALSE)
drlwm <- read.csv("trajectory-drlwm-cover-1.csv", header = FALSE)
mis <- read.csv("trajectory-is-cover-1.csv", header = FALSE)
miswm <- read.csv("trajectory-iswm-cover-1.csv", header = FALSE)
trl <- read.csv("trajectory-trl-cover-1.csv", header = FALSE)
# reg <- read.csv("trajectory-reg-cover-5.csv", header = FALSE)
# regwm <- read.csv("trajectory-regwm-cover-5.csv", header = FALSE)
# drl <- read.csv("trajectory-drl-cover-5.csv", header = FALSE)
# drlwm <- read.csv("trajectory-drlwm-cover-5.csv", header = FALSE)
# mis <- read.csv("trajectory-is-cover-5.csv", header = FALSE)
# miswm <- read.csv("trajectory-iswm-cover-5.csv", header = FALSE)
# trl <- read.csv("trajectory-trl-cover-5.csv", header = FALSE)
res_trajectory <- result_summary3(reg, regwm, drl, drlwm, mis, miswm, trl)

reg <- read.csv("time-reg-cover-1.csv", header = FALSE)
regwm <- read.csv("time-regwm-cover-1.csv", header = FALSE)
drl <- read.csv("time-drl-cover-1.csv", header = FALSE)
drlwm <- read.csv("time-drlwm-cover-1.csv", header = FALSE)
mis <- read.csv("time-is-cover-1.csv", header = FALSE)
miswm <- read.csv("time-iswm-cover-1.csv", header = FALSE)
trl <- read.csv("time-trl-cover-1.csv", header = FALSE)
res_time <- result_summary3(reg, regwm, drl, drlwm, mis, miswm, trl)

process_result2 <- function(mean_tab, row_name, col_name) {
  factored_row_name <- factor(row_name, levels = row_name)
  
  pdat1 <- as.data.frame(mean_tab)
  colnames(pdat1) <- col_name
  pdat1[["Method"]] <- factored_row_name
  pdat1 <- melt(pdat1, c("Method"))
  
  pdat1[["variable"]] <- as.numeric(as.character(pdat1[["variable"]]))
  pdat1
}

pdat1 <- process_result2(res_trajectory, u_method_type, col_name = trajectory_num_vec)
pdat2 <- process_result2(res_time, u_method_type, col_name = trajectory_num_vec)
pdat <- rbind.data.frame(cbind.data.frame(pdat1, class = "Trajectory"), 
                         cbind.data.frame(pdat2, class = "Time"))
pdat[["type"]] <- "Coverage Rate"
p2 <- ggplot(pdat, aes(x = variable, y = value, color = Method)) + 
  facet_grid(class ~ type, scales = "free_y") + 
  geom_hline(yintercept = 0.95, linetype = 2) +
  geom_point(aes(group = Method), size = 2.5) +
  geom_line(aes(group = Method), size = 1) +
  scale_x_continuous(breaks = trajectory_num_vec, trans = "log2") +
  scale_y_continuous(n.breaks = 5) + 
  theme_bw() + 
  xlab("") + ylab("") + 
  theme(
    # legend.position = "bottom", 
        legend.box.margin = margin(t = 0, b = 10, r = 0, l = -30), 
        # strip.text.y = element_blank(),
        plot.margin = unit(c(5.5, 5.5, 0.0, 0.0), "pt")
        ) + 
  guides(guide_legend(byrow = TRUE))
p2
# ggsave(filename = "coverage.jpg", plot = p2, width = 6, height = 3.6)
# ggsave(filename = "coverage.eps", plot = p2, width = 6, height = 3.6)

# p <- ggarrange(p1, p2, ncol = 1, heights = c(2, 1), 
#                common.legend = TRUE, legend = "bottom")

p <- ggarrange(p1, p2, nrow = 1, widths = c(2, 1.13), 
               common.legend = TRUE, legend = "right")
p
# ggexport(p, filename = "comparison.pdf", width = 8, height = 10.2)
ggexport(p, filename = "comparison.pdf", width = 10, height = 5)
