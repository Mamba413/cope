rm(list = ls()); gc(reset = TRUE)
# plot_path <- "yourpath"; setwd(plot_path)
library(reshape2)
library(ggplot2)
library(latex2exp)

mse <- function(x) {
  mean(x^2)
}

mse_sd <- function(x) {
  sd(x^2)
}

rmse <- function(x) {
  sqrt(mse(x))
}

drl_truth <- read.csv("triple-robust-drl-truth.csv", header = FALSE)
truth <- read.csv("triple-robust-trl-truth.csv", header = FALSE)
correct1 <- read.csv("triple-robust-trl-correct-1.csv", header = FALSE)
correct2 <- read.csv("triple-robust-trl-correct-2.csv", header = FALSE)

root_experiment_rep <- sqrt(ncol(drl_truth))
bias <- rbind(rowMeans(drl_truth), rowMeans(truth), rowMeans(correct1), rowMeans(correct2))
std <- rbind(apply(drl_truth, 1, sd), apply(truth, 1, sd), 
             apply(correct1, 1, sd), apply(correct2, 1, sd)) / root_experiment_rep
mse <- rbind(apply(drl_truth, 1, mse), apply(truth, 1, mse), apply(correct1, 1, mse), apply(correct2, 1, mse))
msesd <- rbind(apply(drl_truth, 1, mse_sd), apply(truth, 1, mse_sd), 
               apply(correct1, 1, mse_sd), apply(correct2, 1, mse_sd)) / root_experiment_rep

########################################################
###################### Bias & MSE ######################
########################################################
pdat <- rbind(t(bias), t(std))
pdat <- as.data.frame(pdat)
colnames(pdat) <- c("drl_truth", "truth", "correct-1", "correct-2")
trajetory_num <- 15
pdat[["size"]] <- rep((1:trajetory_num) * 60, 2)
pdat[["type"]] <- rep(c("bias", "std"), each = trajetory_num)
pdat <- melt(pdat, c("type", "size"))
pdat <- dcast(pdat, size + variable ~ type)
legend_label <- c("DRL", "$\\mathrm{M}_1 & \\mathrm{M}_2$", "$\\mathrm{M}_1$", "$\\mathrm{M}_2$")
legend_label <- lapply(legend_label, TeX)
pdat1 <- pdat

pdat <- rbind(t(mse), t(msesd))
pdat <- as.data.frame(pdat)
colnames(pdat) <- c("drl_truth", "truth", "correct-1", "correct-2")
pdat[["size"]] <- rep(c(1:trajetory_num) * 60, 2)
pdat[["type"]] <- rep(c("mse", "std"), each = trajetory_num)
pdat <- melt(pdat, c("type", "size"))
pdat <- dcast(pdat, size + variable ~ type)
legend_label <- c("DRL", "$\\mathrm{M}_1 & \\mathrm{M}_2$", "$\\mathrm{M}_1$", "$\\mathrm{M}_2$")
legend_label <- lapply(legend_label, TeX)
pdat2 <- pdat

############################################
################### Plot ###################
############################################
pdat_1 <- cbind(pdat1, "bias")
colnames(pdat_1) <- c("size", "variable", "value", "std", "metric")
pdat_2 <- cbind(pdat2, "MSE")
colnames(pdat_2) <- colnames(pdat_1)

pdat <- rbind.data.frame(pdat_1, pdat_2)

dummy1 <- data.frame(size = c(1, trajetory_num) * 60, value = c(-1.1, 1.1), 
                    metric = "bias", stringsAsFactors = FALSE)
dummy1 <- cbind.data.frame(dummy1, variable = unique(pdat[["variable"]]))
dummy2 <- data.frame(size = c(1, trajetory_num) * 60, value = c(0, 1.23), 
                     metric = "MSE", stringsAsFactors = FALSE)
dummy2 <- cbind.data.frame(dummy2, variable = unique(pdat[["variable"]]))
dummy <- rbind(dummy1, dummy2)

p <- ggplot(pdat, aes(x = size, y = value, colour = variable)) + 
  facet_wrap(metric ~ ., scales = "free") + 
  geom_point(aes(group = variable), size = 2.5) +
  geom_line(aes(group = variable), size = 1) + 
  geom_blank(data = dummy, inherit.aes = TRUE) + 
  geom_errorbar(aes(ymin=value - 1.96 * std, ymax = value + 1.96 * std), 
                width = 10, size = 1) + 
  theme_bw() + 
  scale_y_continuous(name = "") +
  xlab("trajectory") + 
  theme(legend.title = element_blank(), 
        legend.position = c(0.13, 0.2),
        legend.box.background = element_rect(color = "black"), 
        legend.margin = margin(r=2.5,l=1.5,t=0.1,b=1)) + 
  scale_colour_discrete(labels = legend_label) + 
  guides(color = guide_legend(nrow = 2, title = "", byrow = FALSE))

ggsave(filename = "triple_robust_combine.jpg", plot = p, 
       width = 6.8, height = 2.8)

