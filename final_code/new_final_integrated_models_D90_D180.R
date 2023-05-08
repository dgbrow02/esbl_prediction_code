###this code is for supplemental material, using cohorts with follow-up timepoints to test model performance at days 90 or 180

library(tidyverse)
library(furrr)
library(cvAUC)
library(glmnet)
library(caret)
library(ranger)
library(pROC)
library(Rmisc)

setwd("~/Documents/R/ESBL_acq/code/")

joined_data_impute_complete <- read.csv("rational_not_scaled.csv")

joined_data_impute_complete[c(1:7,11:13,16:24,26)] <- lapply(joined_data_impute_complete[c(1:7,11:13,16:24,26)], factor)

#filter data by cohort

d90_results <- read.csv("filtered_d90_participants.csv")
d180_results <- read.csv("filtered_d180_participants.csv")

d90_results$g10_id <- as.factor(d90_results$g10_id)
d180_results$g10_id <- as.factor(d180_results$g10_id)

rational_d90 <- d90_results %>%  inner_join(joined_data_impute_complete) %>% filter(!is.na(esbl_acq))
rational_d180 <- d180_results %>%  inner_join(joined_data_impute_complete) %>% filter(!is.na(esbl_acq))

rational_d90$esbl_acq <- as.numeric(rational_d90$esbl_acq)-1
rational_d180$esbl_acq <- as.numeric(rational_d180$esbl_acq)-1

resp_var <- "esbl_acq"
nvars_opts = c(1:15)

##Scale and center data

num_names = rational_d90 %>% select(!c("g10_id","esbl_acq", "esbl_d90")) %>% names()
preProcessed90 <- preProcess(rational_d90[,num_names], method = c("center","scale"))
rational_d90[,num_names] <- predict(preProcessed90,rational_d90[,num_names])

preProcessed180 <- preProcess(rational_d180[,num_names], method = c("center","scale"))
rational_d180[,num_names] <- predict(preProcessed180,rational_d180[,num_names])

#Most important variables in dataset by LASSO regression
feature_matrix <- rational_d90 %>% select(!c(g10_id, esbl_acq, esbl_d90))  %>% data.matrix()
outcome_matrix <- rational_d90$esbl_acq %>% as.matrix()
  
glm <- glmnet(feature_matrix,outcome_matrix,family = "binomial", alpha = 1, lambda  = runif(1000, min = 0.001, max = 0.15))
glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)
lambda_sel <- glm_df %>% filter(Df == 15) %>% filter(dev == max(.$dev)) %>% {.$Lambda}
vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)

##Most important variables in dataset by LASSO regression d90
feature_matrix_d90 <- rational_d90 %>% select(!c(g10_id, esbl_acq, esbl_d90))  %>% data.matrix()
outcome_matrix_d90 <- rational_d90$esbl_d90 %>% as.matrix()

glm90 <- glmnet(feature_matrix_d90,outcome_matrix_d90,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.001, max = 0.15))
glm_df90 <- data_frame(Df = glm90$df, Lambda = glm90$lambda, dev = glm90$dev.ratio)
lambda_sel90 <- glm_df90 %>% filter(Df == 15) %>% filter(dev == max(.$dev)) %>% {.$Lambda}
vars <- varImp(glm90, lambda = lambda_sel90) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)


##Most important variables in dataset by LASSO regression d180
feature_matrix_d180 <- rational_d180 %>% select(!c(g10_id, esbl_acq, esbl_d180))  %>% data.matrix()
outcome_matrix_d180 <- rational_d180$esbl_d180 %>% as.matrix()

glm180 <- glmnet(feature_matrix_d180,outcome_matrix_d180,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.001, max = 0.15))
glm_df180 <- data_frame(Df = glm180$df, Lambda = glm180$lambda, dev = glm180$dev.ratio)
lambda_sel180 <- glm_df180 %>% filter(Df == 15) %>% filter(dev == max(.$dev)) %>% {.$Lambda}
vars <- varImp(glm180, lambda = lambda_sel180) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)

#Models
##LASSO selection/ranking and glm model; train and test on Day 0 for Day 90 cohort
glm_no_pretest <- function(x,y){
  train=rational_d90 %>% sample_frac(.80,replace=F)
  train_feature_matrix <- train %>% select(!c("g10_id","esbl_acq", "esbl_d90")) %>% data.matrix()
  train_outcome_matrix <- train %>% select(esbl_acq) %>%  as.matrix()
  test=rational_d90[-which(rational_d90$g10_id %in% train$g10_id),]

  each = x
  nvar = y
  
  repeat{
    glm <- glmnet(train_feature_matrix,train_outcome_matrix,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.001, max = 0.15))
    glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)
    lambda_sel <- glm_df %>% filter(Df == y) %>% filter(dev == min(.$dev)) %>% {.$Lambda}
    if(length(lambda_sel) != 0){
      break
    }
  }
  vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)
  
  patient_mod <- glm(as.formula(paste("esbl_acq",'~',paste(vars$Var,collapse="+"),sep="")),data=train,family="binomial")
  
  df=data.frame(iter=each,nvar=nvar,true=test[["esbl_acq"]],pred_glm=as.numeric(predict(patient_mod,newdata=test,type="response")))
  df
}

plan(multiprocess)
glm_no_pre <- cross2(nvars_opts,1:100) %>% future_map(~glm_no_pretest(.[[2]],.[[1]]),.progress=T)

glm_no_pre_bound=bind_rows(glm_no_pre)

glm_no_pre_AUCs=bind_rows(glm_no_pre_bound) %>% split(.$nvar) %>% purrr::map(~ci.cvAUC(.$pred_glm,.$true,folds=.$iter))

glm_no_pre_AUCs_plot <- glm_no_pre_AUCs %>% bind_rows() %>% select(cvAUC,ci)  %>% mutate(nvars_opts = rep(nvars_opts, 2) %>% sort()) %>% group_by(nvars_opts) %>% dplyr::summarize(cvAUC = mean(cvAUC), CI.1 = min(ci), CI.2 = max(ci))
  
ggplot(glm_no_pre_AUCs_plot, aes(x = nvars_opts, y = cvAUC)) + geom_point() + geom_errorbar(aes(ymin=CI.1, ymax=CI.2))

ggplot(glm_no_pre_AUCs_plot, aes(x = nvars_opts, y = cvAUC)) + geom_point() + theme_bw() + xlab("Number of Features") + ylab("Cross-Validated AUC") + ggtitle("LR cvAUC D0") +
  theme(plot.title = element_text(hjust = 0.5)) + geom_errorbar(aes(ymin=CI.1, ymax=CI.2)) + ylim(0.4,0.8)


#LASSO selection/ranking and glm model; train on Day 0, test on day 90 for day 90 cohort
glm_no_pretest_d90 <- function(x,y){
  train=rational_d90 %>% sample_frac(.80,replace=F)
  train_feature_matrix <- train %>% select(!c("g10_id","esbl_acq", "esbl_d90")) %>% data.matrix()
  train_outcome_matrix <- train %>% select(esbl_acq) %>%  as.matrix()
  test=rational_d90[-which(rational_d90$g10_id %in% train$g10_id),]
  test_feature_matrix <- test %>% select(!c("g10_id","esbl_acq","esbl_d90")) %>% data.matrix()
  test_outcomes_matrix <- test %>% select(esbl_d90) %>%  as.matrix()
    
  
  each = x
  nvar = y
  
  repeat{
    glm <- glmnet(train_feature_matrix,train_outcome_matrix,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.001, max = 0.20))
    glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)
    lambda_sel <- glm_df %>% filter(Df == y) %>% filter(dev == min(.$dev)) %>% {.$Lambda}
    if(length(lambda_sel) != 0){
      break
    }
  }
  vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)
  
  patient_mod <- glm(as.formula(paste("esbl_acq",'~',paste(vars$Var,collapse="+"),sep="")),data=train,family="binomial")
  
  df=data.frame(iter=each,nvar=nvar,true=test[["esbl_d90"]],pred_glm=as.numeric(predict(patient_mod,newdata=test,type="response")))
  df
}

plan(multiprocess)
glm_no_pre_d90 <- cross2(nvars_opts,1:100) %>% future_map(~glm_no_pretest_d90(.[[2]],.[[1]]),.progress=T)

glm_no_pre_bound_d90=bind_rows(glm_no_pre_d90)

glm_no_pre_AUCs_d90=bind_rows(glm_no_pre_bound_d90) %>% split(.$nvar) %>% purrr::map(~ci.cvAUC(.$pred_glm,.$true,folds=.$iter))

glm_no_pre_AUCs_plot_d90 <- glm_no_pre_AUCs_d90 %>% bind_rows() %>% select(cvAUC,ci)  %>% mutate(nvars_opts = rep(nvars_opts, 2) %>% sort()) %>% group_by(nvars_opts) %>% dplyr::summarize(cvAUC = mean(cvAUC), CI.1 = min(ci), CI.2 = max(ci))

ggplot(glm_no_pre_AUCs_plot_d90, aes(x = nvars_opts, y = cvAUC)) + geom_point() + geom_errorbar(aes(ymin=CI.1, ymax=CI.2))

ggplot(glm_no_pre_AUCs_plot_d90, aes(x = nvars_opts, y = cvAUC)) + geom_point() + theme_bw() + xlab("Number of Features") + ylab("Cross-Validated AUC") + ggtitle("LR cvAUC D90") +
  theme(plot.title = element_text(hjust = 0.5)) + geom_errorbar(aes(ymin=CI.1, ymax=CI.2)) + ylim(0.4,0.8)

##LASSO selection/ranking and glm model train on Day 0,  test on Day 0, for day 180 cohort
glm_no_pretest_d0_d180_d0 <- function(x,y){
  train=rational_d180 %>% sample_frac(.80,replace=F)
  train_feature_matrix <- train %>% select(!c("g10_id","esbl_acq", "esbl_d180")) %>% data.matrix()
  train_outcome_matrix <- train %>% select(esbl_acq) %>%  as.matrix()
  test=rational_d180[-which(rational_d180$g10_id %in% train$g10_id),]
  test_feature_matrix <- test %>% select(!c("g10_id","esbl_acq","esbl_d180")) %>% data.matrix()
  test_outcomes_matrix <- test %>% select(esbl_acq) %>%  as.matrix()
  
  each = x
  nvar = y
  
  repeat{
    glm <- glmnet(train_feature_matrix,train_outcome_matrix,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.001, max = 0.2))
    glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)
    lambda_sel <- glm_df %>% filter(Df == y) %>% filter(dev == min(.$dev)) %>% {.$Lambda}
    if(length(lambda_sel) != 0){
      break
    }
  }
  vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)
  
  patient_mod <- glm(as.formula(paste("esbl_acq",'~',paste(vars$Var,collapse="+"),sep="")),data=train,family="binomial")
  
  df=data.frame(iter=each,nvar=nvar,true=test[["esbl_acq"]],pred_glm=as.numeric(predict(patient_mod,newdata=test,type="response")))
  df
}

plan(multiprocess)
glm_no_pre_d0d180d0 <- cross2(nvars_opts,1:100) %>% future_map(~glm_no_pretest_d0_d180_d0(.[[2]],.[[1]]),.progress=T)

glm_no_pre_bound_d0d180d0=bind_rows(glm_no_pre_d0d180d0)

glm_no_pre_AUCs_d0d180=bind_rows(glm_no_pre_bound_d0d180d0) %>% split(.$nvar) %>% purrr::map(~ci.cvAUC(.$pred_glm,.$true,folds=.$iter))

glm_no_pre_AUCs_plot_d0d180d0 <- glm_no_pre_AUCs_d0d180 %>% bind_rows() %>% select(cvAUC,ci)  %>% mutate(nvars_opts = rep(nvars_opts, 2) %>% sort()) %>% group_by(nvars_opts) %>% dplyr::summarize(cvAUC = mean(cvAUC), CI.1 = min(ci), CI.2 = max(ci))

ggplot(glm_no_pre_AUCs_plot_d0d180d0, aes(x = nvars_opts, y = cvAUC)) + geom_point() + geom_errorbar(aes(ymin=CI.1, ymax=CI.2))

ggplot(glm_no_pre_AUCs_plot_d0d180d0, aes(x = nvars_opts, y = cvAUC)) + geom_point() + theme_bw() + xlab("Number of Features") + ylab("Cross-Validated AUC") + ggtitle("LR cvAUC D0") +
  theme(plot.title = element_text(hjust = 0.5)) + geom_errorbar(aes(ymin=CI.1, ymax=CI.2)) + ylim(0.4,0.8)

##LASSO selection/ranking and glm model; train on day 0, test on day 180, for day 180 cohort
glm_no_pretest_d0_d180 <- function(x,y){
  train=rational_d180 %>% sample_frac(.80,replace=F)
  train_feature_matrix <- train %>% select(!c("g10_id","esbl_acq", "esbl_d180")) %>% data.matrix()
  train_outcome_matrix <- train %>% select(esbl_acq) %>%  as.matrix()
  test=rational_d180[-which(rational_d180$g10_id %in% train$g10_id),]
  test_feature_matrix <- test %>% select(!c("g10_id","esbl_acq","esbl_d180")) %>% data.matrix()
  test_outcomes_matrix <- test %>% select(esbl_d180) %>%  as.matrix()
  
  each = x
  nvar = y
  
  repeat{
    glm <- glmnet(train_feature_matrix,train_outcome_matrix,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.001, max = 0.2))
    glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)
    lambda_sel <- glm_df %>% filter(Df == y) %>% filter(dev == min(.$dev)) %>% {.$Lambda}
    if(length(lambda_sel) != 0){
      break
    }
  }
  vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)
  
  patient_mod <- glm(as.formula(paste("esbl_acq",'~',paste(vars$Var,collapse="+"),sep="")),data=train,family="binomial")
  
  df=data.frame(iter=each,nvar=nvar,true=test[["esbl_d180"]],pred_glm=as.numeric(predict(patient_mod,newdata=test,type="response")))
  df
}

plan(multiprocess)
glm_no_pre_d0d180 <- cross2(nvars_opts,1:100) %>% future_map(~glm_no_pretest_d0_d180(.[[2]],.[[1]]),.progress=T)

glm_no_pre_bound_d0d180 <- glm_no_pre_d0d180 %>% map(~filter(., max(.$true)==1)) %>% bind_rows()

glm_no_pre_AUCs_d0d180 <- bind_rows(glm_no_pre_bound_d0d180) %>% split(.$nvar) %>% purrr::map(~ci.cvAUC(.$pred_glm,.$true,folds=.$iter))

glm_no_pre_AUCs_plot_d0d180 <- glm_no_pre_AUCs_d0d180 %>% bind_rows() %>% select(cvAUC,ci)  %>% mutate(nvars_opts = rep(nvars_opts, 2) %>% sort()) %>% group_by(nvars_opts) %>% dplyr::summarize(cvAUC = mean(cvAUC), CI.1 = min(ci), CI.2 = max(ci))

ggplot(glm_no_pre_AUCs_plot_d0d180, aes(x = nvars_opts, y = cvAUC)) + geom_point() + geom_errorbar(aes(ymin=CI.1, ymax=CI.2))

ggplot(glm_no_pre_AUCs_plot_d0d180, aes(x = nvars_opts, y = cvAUC)) + geom_point() + theme_bw() + xlab("Number of Features") + ylab("Cross-Validated AUC") + ggtitle("LR cvAUC D180") +
  theme(plot.title = element_text(hjust = 0.5)) + geom_errorbar(aes(ymin=CI.1, ymax=CI.2)) + ylim(0.4,0.8)

