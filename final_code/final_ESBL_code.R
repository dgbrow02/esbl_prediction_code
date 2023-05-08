###########################final code accompanying publication##############################

#Setting up environment
library(tidyverse)
library(furrr)
library(cvAUC)
library(glmnet)
library(caret)
library(ranger)
library(pROC)
library(Rmisc)

#set working directory
setwd("~/Documents/R/ESBL_acq/code/final_code/")

#file contains all data
##Rational data is processed and scaled as explained in methods
rational <- read_rds("rational.rds") 

#Determining most important features on entire dataset
##create matrices from data
feature_matrix <- rational[3:ncol(rational)] %>% data.matrix()
outcome_matrix <- rational$esbl_acq %>% as.matrix()

##run LASSO regression at various lambda values
glm <- glmnet(feature_matrix,outcome_matrix,family = "binomial", alpha = 1, lambda = seq(from = 0.15, to = 0.01, by = -0.0001))
glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)

##set number of features in model after Df
number_features <- 4
##separate by number of features and ID selected variables
lambda_sel <- glm_df %>% filter(Df == number_features) %>% filter(dev == max(.$dev)) %>% {.$Lambda}
vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)

##logistic regression for effect direction
tbl=summary(glm(as.formula(paste("esbl_acq",'~',paste(vars$Var,collapse="+"),sep="")),data=rational ,family="binomial"))

##caculate adjusted odds ratio
adj_odds=data.frame(Adjusted_Odds=exp(tbl$coefficients[,1]), CI=
                      t(bind_rows(map2(tbl$coefficients[,1],tbl$coefficients[,2], function(x,y) exp(x + c(-1,1)*y*qnorm(.975))))),
                    P_value=tbl$coefficients[,4])
adj_odds$CI=paste(round(adj_odds$CI.1,4),"-",round(adj_odds$CI.2,4))

adj_odds <- adj_odds %>% mutate(rank = 1:nrow(adj_odds))

coef=data.frame(Coef=tbl$coefficients[,1], CI=
                  t(bind_rows(map2(tbl$coefficients[,1],tbl$coefficients[,2], function(x,y) (x + c(-1,1)*y*qnorm(.975))))),
                P_value=tbl$coefficients[,4])
coef$CI=paste("(",round(coef$CI.1,4)," - ",round(coef$CI.2,4),")",sep="")
coef %>% select(Coef,CI, P_value)

##Example plot
ggplot(adj_odds) + geom_point(aes(x = as.factor(rank), y = log(Adjusted_Odds)), fill = NA, color = "Black") + theme(axis.title.x = element_blank(), axis.line = element_line(colour = "black"), panel.background = element_blank(), panel.border = element_blank()) + ylab("Log Odds Ratio") + scale_x_discrete(labels=c("Intercept","Antibiotics for Diarrhea","Waste Management Rankings","Regional Probabilities", "Any Diarrhea")) + theme(axis.text.x = element_text(angle = 45,  hjust=1)) +
  geom_errorbar(aes(x=rank, ymin=log(CI.1), ymax=log(CI.2)), width=0.4, colour="Black", alpha=1, size=1) + geom_hline(yintercept = 0) 

#Prediction models

##set number of features
nvars_opts <- c(1:15)

##LASSO selection/ranking and glm model

lr_model <- function(x,y){
  train=rational %>% sample_frac(.80,replace=F)
  train_feature_matrix <- train %>% select(!c("g10_id","esbl_acq")) %>% data.matrix()
  train_outcome_matrix <- train %>% select(esbl_acq) %>%  as.matrix()
  test=rational[-which(rational$g10_id %in% train$g10_id),]
  test_feature_matrix <- test %>% select(!c("g10_id","esbl_acq")) %>% data.matrix()
  test_outcomes_matrix <- test %>% select(esbl_acq) %>%  as.matrix()
  
  each = x
  nvar = y
  
  repeat{
    glm <- glmnet(train_feature_matrix,train_outcome_matrix,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.01, max = 0.15))
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
lr_model_results <- cross2(nvars_opts,1:100) %>% future_map(~lr_model(.[[2]],.[[1]]),.progress=T)

lr_model_results_bound=bind_rows(lr_model_results)

lr_model_AUCs=bind_rows(lr_model_results_bound) %>% split(.$nvar) %>% purrr::map(~ci.cvAUC(.$pred_glm,.$true,folds=.$iter))

lr_model_AUCs_plot <- lr_model_AUCs %>% bind_rows() %>% select(cvAUC,ci)  %>% mutate(nvars_opts = rep(nvars_opts, 2) %>% sort()) %>% group_by(nvars_opts) %>% dplyr::summarize(cvAUC = mean(cvAUC), CI.1 = min(ci), CI.2 = max(ci))

ggplot(lr_model_AUCs_plot, aes(x = nvars_opts, y = cvAUC)) + geom_point() + theme_bw() + xlab("Number of Features") + ylab("Cross-Validated AUC") + ggtitle("LR Cross Validated AUCs") +
  theme(plot.title = element_text(hjust = 0.5)) + geom_errorbar(aes(ymin=CI.1, ymax=CI.2)) + ylim(0.4,0.8)

##LASSO selection/ranking and rf model

rf_model <- function(x,y){
  train=rational %>% sample_frac(.80,replace=F)
  train_feature_matrix <- train %>% select(!c("g10_id","esbl_acq")) %>% data.matrix()
  train_outcome_matrix <- train %>% select(esbl_acq) %>%  as.matrix()
  test=rational[-which(rational$g10_id %in% train$g10_id),]
  test_feature_matrix <- test %>% select(!c("g10_id","esbl_acq")) %>% data.matrix()
  test_outcomes_matrix <- test %>% select(esbl_acq) %>%  as.matrix()
  
  each = x
  nvar = y
  
  repeat{
    glm <- glmnet(train_feature_matrix,train_outcome_matrix,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.01, max = 0.15))
    glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)
    lambda_sel <- glm_df %>% filter(Df == y) %>% filter(dev == min(.$dev)) %>% {.$Lambda}
    if(length(lambda_sel) != 0){
      break
    }
  }
  vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)
  
  patient_mod <- ranger(as.formula(paste("esbl_acq",'~',paste(vars$Var,collapse="+"),sep="")),data=train)
  
  df=data.frame(nvar=nvar,true=test[["esbl_acq"]],pred_RF=as.numeric(predict(patient_mod,data=test,type="response")$predictions))
  df
}

plan(multiprocess)
rf_model_results <- cross2(nvars_opts,1:100) %>% future_map(~rf_model(.[[2]],.[[1]]),.progress=T)

rf_model_results_bound=bind_rows(rf_model_results)

rf_model_AUCs=bind_rows(rf_model_results_bound) %>% split(.$nvar) %>% purrr::map(~ci.cvAUC(.$pred_RF,.$true,folds=.$iter))

rf_model_AUCs_plot <- rf_model_AUCs %>% bind_rows() %>% select(cvAUC,ci)  %>% mutate(nvars_opts = rep(nvars_opts, 2) %>% sort()) %>% group_by(nvars_opts) %>% dplyr::summarize(cvAUC = mean(cvAUC), CI.1 = min(ci), CI.2 = max(ci))

ggplot(rf_model_AUCs_plot, aes(x = nvars_opts, y = cvAUC)) + geom_point() +  theme_bw() + xlab("Number of Features") + ylab("Cross-Validated AUC") + ggtitle("RF Cross Validated AUCs") +
  theme(plot.title = element_text(hjust = 0.5)) + geom_errorbar(aes(ymin=CI.1, ymax=CI.2)) + ylim(0.4,0.8)

#glm model performance

glm_perf <- function(x,y){
  train=rational %>% sample_frac(.80,replace=F)
  train_feature_matrix <- train %>% select(!c("g10_id","esbl_acq")) %>% data.matrix()
  train_outcome_matrix <- train %>% select(esbl_acq) %>%  as.matrix()
  test=rational[-which(rational$g10_id %in% train$g10_id),]
  test_feature_matrix <- test %>% select(!c("g10_id","esbl_acq")) %>% data.matrix()
  test_outcomes_matrix <- test %>% select(esbl_acq) %>%  as.matrix()
  
  each = x
  nvar = y
  
  repeat{
    glm <- glmnet(train_feature_matrix,train_outcome_matrix,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.01, max = 0.15))
    glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)
    lambda_sel <- glm_df %>% filter(Df == y) %>% filter(dev == min(.$dev)) %>% {.$Lambda}
    if(length(lambda_sel) != 0){
      break
    }
  }
  vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)
  
  patient_mod <- glm(as.formula(paste("esbl_acq",'~',paste(vars$Var,collapse="+"),sep="")),data=train,family="binomial")
  
  roc_res=roc(test$esbl_acq,pred=as.numeric(predict(patient_mod,newdata=test,type="response"), ci=T, plot =T))
  roc_smooth <- smooth(roc_res, n = 21)
  df <- coords(roc_smooth, ret=c("threshold", "sens", "spec", "ppv", "npv"))
  df
}

plan(multiprocess)
glm_perf_results <- cross2(4,1:100) %>% future_map(~glm_perf(.[[2]],.[[1]]),.progress=T)

glm_perf_results_bound <- glm_perf_results %>% bind_rows()

glm_stats <- glm_perf_results_bound %>% group_by(sensitivity) %>% dplyr::summarize(specif = mean(specificity), specificity_error = sd(specificity)/sqrt(n())*qt(p=0.05/2, df=n()-1,lower.tail=F), ppv_mean = mean(ppv), ppv_error = sd(ppv)/sqrt(n())*qt(p=0.05/2, df=n()-1,lower.tail=F), npv_mean = mean(npv), npv_error = sd(npv)/sqrt(n())*qt(p=0.05/2, df=n()-1,lower.tail=F))

#rf model performance

rf_perf <- function(x,y){
  train=rational %>% sample_frac(.80,replace=F)
  train_feature_matrix <- train %>% select(!c("g10_id","esbl_acq")) %>% data.matrix()
  train_outcome_matrix <- train %>% select(esbl_acq) %>%  as.matrix()
  test=rational[-which(rational$g10_id %in% train$g10_id),]
  test_feature_matrix <- test %>% select(!c("g10_id","esbl_acq")) %>% data.matrix()
  test_outcomes_matrix <- test %>% select(esbl_acq) %>%  as.matrix()
  
  each = x
  nvar = y
  
  repeat{
    glm <- glmnet(train_feature_matrix,train_outcome_matrix,family = "binomial", alpha = 1, lambda = runif(1000, min = 0.01, max = 0.15))
    glm_df <- data_frame(Df = glm$df, Lambda = glm$lambda, dev = glm$dev.ratio)
    lambda_sel <- glm_df %>% filter(Df == y) %>% filter(dev == min(.$dev)) %>% {.$Lambda}
    if(length(lambda_sel) != 0){
      break
    }
  }
  
  vars <- varImp(glm, lambda = lambda_sel) %>% rownames_to_column("Var") %>% arrange(desc(Overall)) %>% filter(Overall != 0)
  
  patient_mod <- ranger(as.formula(paste("esbl_acq",'~',paste(vars$Var,collapse="+"),sep="")),data=train)
  
  preds <- predict(patient_mod, data = test,type = "response")
  roc_res=roc(test$esbl_acq,pred=preds$predictions)
  roc_smooth <- smooth(roc_res, n = 21)
  df <- coords(roc_smooth, ret=c("threshold", "sens", "spec", "ppv", "npv"))
  df
}

plan(multiprocess)
rf_perf_results <- cross2(4,1:100) %>% future_map(~rf_perf(.[[2]],.[[1]]),.progress=T)

rf_perf_results_bound <- rf_perf_results %>% bind_rows()

rf_stats <- rf_perf_results_bound %>% group_by(sensitivity) %>% dplyr::summarize(specif = mean(specificity), specificity_error = sd(specificity)/sqrt(n())*qt(p=0.05/2, df=n()-1,lower.tail=F), ppv_mean = mean(ppv), ppv_error = sd(ppv)/sqrt(n())*qt(p=0.05/2, df=n()-1,lower.tail=F), npv_mean = mean(npv), npv_error = sd(npv)/sqrt(n())*qt(p=0.05/2, df=n()-1,lower.tail=F))



