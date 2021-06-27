rm(list = ls()); gc(reset = TRUE)
# plot_path <- "yourpath"; setwd(plot_path)
library(reshape2)
library(ggplot2)
library(ggpubr)

u_method_type <- c("Direct", "IS", "COPE")
method_type <- factor(rep(u_method_type, 2),
                      levels = u_method_type)

assessment_type <- rep(c("Bias", "MSE"), each = length(u_method_type))

result_summary2 <- function(direct, is, cope, trim = 0.00) {
  root_experiment_rep <- sqrt(ncol(direct))
  
  bias_tab <- rbind(apply(abs(direct), 1, mean, trim = trim),
                    apply(abs(is), 1, mean, trim = trim),
                    apply(abs(cope), 1, mean, trim = trim))
  
  sd_tab <- rbind(apply(abs(direct), 1, sd),
                  apply(abs(is), 1, sd),
                  apply(abs(cope), 1, sd))
  sd_tab <- sd_tab / root_experiment_rep
  
  mse <- function(x, trim = 0) {
    if (trim > 0) {
      q1 <- quantile(x, trim)
      q2 <- quantile(x, 1 - trim)
      x <- x[x >= q1 & x <= q2]
    }
    mean(x^2)
  }
  
  mse_tab <- rbind(apply(direct, 1, mse, trim = trim),
                   apply(is, 1, mse, trim = trim),
                   apply(cope, 1, mse, trim = trim))
  
  mssd_tab <- rbind(apply(direct^2, 1, sd),
                    apply(is^2, 1, sd),
                    apply(cope^2, 1, sd))
  mssd_tab <- mssd_tab / root_experiment_rep
  
  list(bias_tab, sd_tab, mse_tab, mssd_tab)
}

# trajectory_num_vec <- c(20, 40, 80, 160, 320, 640, 1280)
trajectory_num_vec <- c(20, 40, 80, 160, 320)
# trajectory_num_vec <- c(640, 1280)
trajectory_num_vec_len <- length(trajectory_num_vec)

cope <- read.csv("trajectory-dic-cope-3.csv", header = FALSE)
is <- read.csv("trajectory-dic-is-3.csv", header = FALSE)
direct <- read.csv("trajectory-dic-direct-3.csv", header = FALSE)
res_trajectory <- result_summary2(direct, is, cope)

res_time <- res_trajectory
# cope <- read.csv("time-dic-cope-3.csv", header = FALSE)
# is <- read.csv("time-dic-is-3.csv", header = FALSE)
# direct <- read.csv("time-dic-direct-3.csv", header = FALSE)
# res_time <- result_summary2(direct, is, cope)

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

pdat <- rbind.data.frame(cbind.data.frame(pdat_1_1, type = "logBias", class = "trajectory"), 
                         cbind.data.frame(pdat_1_2, type = "logMSE", class = "trajectory"), 
                         cbind.data.frame(pdat_1_1, type = "logBias", class = "time"), 
                         cbind.data.frame(pdat_1_2, type = "logMSE", class = "time"))

p1 <- ggplot(pdat, aes(x = variable, y = log10(value), color = Method)) + 
  facet_grid(type ~ class, scales = "free_y") + 
  geom_point(aes(group = Method), size = 2.5) +
  geom_line(aes(group = Method), size = 1) +
  geom_errorbar(aes(ymin = log10(ci_min), ymax = log10(ci_max)), size = 1, width = 0.1) + 
  scale_x_continuous(breaks = trajectory_num_vec, trans = "log2") +
  scale_y_continuous(n.breaks = 6) +
  theme_bw() + 
  xlab("") + ylab("") + 
  theme(legend.position = "bottom", 
        legend.box.margin = margin(t = -10, b = 10, r = 0, l = 0), 
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) + 
  guides(guide_legend(byrow = TRUE))
p1
# ggsave(filename = "comparison.jpg", plot = p1, width = 6, height = 5.6)
# ggsave(filename = "comparison.eps", plot = p1, width = 6, height = 5.6)

##################################################
#################### Coverage ####################
##################################################
result_summary3 <- function(direct, is, cope, trim = 0.00) {
  cover_tab <- rbind(apply(direct, 1, mean, trim = trim),
                     apply(is, 1, mean, trim = trim),
                     apply(cope, 1, mean, trim = trim))
  cover_tab
}

direct <- read.csv("trajectory-dic-direct-3.csv", header = FALSE)
is <- read.csv("trajectory-dic-is-cover-3.csv", header = FALSE)
cope <- read.csv("trajectory-dic-cope-cover-3.csv", header = FALSE)
res_trajectory <- result_summary3(direct, is, cope)

res_time <- res_trajectory
# direct <- read.csv("time-dic-direct-3.csv", header = FALSE)
# is <- read.csv("time-dic-is-cover-3.csv", header = FALSE)
# cope <- read.csv("time-dic-cope-cover-3.csv", header = FALSE)
# res_time <- result_summary3(direct, is, cope)

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
pdat <- rbind.data.frame(cbind.data.frame(pdat1, class = "trajectory"), 
                         cbind.data.frame(pdat2, class = "time"))
pdat[["type"]] <- "Coverage Rate"
p2 <- ggplot(pdat, aes(x = variable, y = value, color = Method)) + 
  facet_grid(type ~ class, scales = "free_y") + 
  geom_hline(yintercept = 0.95, linetype = 2) +
  geom_point(aes(group = Method), size = 2.5) +
  geom_line(aes(group = Method), size = 1) +
  scale_x_continuous(breaks = trajectory_num_vec, trans = "log2") +
  scale_y_continuous(n.breaks = 5) + 
  theme_bw() + 
  xlab("") + ylab("") + 
  theme(
    legend.position = "bottom", 
    # legend.box.margin = margin(t = -30, b = 10, r = 0, l = 0),
    strip.text.x = element_blank(),
    # plot.margin = unit(c(0.0, 5.5, 5.5, 5.5), "pt")
    ) + 
  guides(guide_legend(byrow = TRUE))
p2
# ggsave(filename = "coverage.jpg", plot = p2, width = 6, height = 3.6)
# ggsave(filename = "coverage.eps", plot = p2, width = 6, height = 3.6)

p <- ggarrange(p1, p2, ncol = 1, heights = c(2, 1), 
               common.legend = TRUE, legend = "bottom")
p
ggexport(p, filename = "comparison.pdf", width = 8, height = 10.2)
