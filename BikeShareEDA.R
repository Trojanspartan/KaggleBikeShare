library(tidyverse)
library(tidymodels)
library(patchwork)
library(DataExplorer)
library(ggplot2)

train = vroom::vroom("train.csv")

test = vroom::vroom("test.csv")

dplyr::glimpse(train)

skimr::skim(train)

eda1 = plot_correlation(train, type = "continuous",
                        ggtheme = theme(axis.text.x = element_text(angle = 45, hjust = 1),
                                        legend.position = "none"))

eda2 = ggplot(data = na.omit(train), aes(x = temp, y = count)) + 
  geom_point() + 
  geom_smooth(se = FALSE)

eda3 = ggplot(data = train, aes(x = weather)) + 
  geom_bar()

eda4 = ggplot(data = na.omit(train), aes(x = humidity, y = count)) + 
  geom_point() + 
  geom_smooth(se = FALSE)

(eda1 + eda2)/(eda3+eda4)
