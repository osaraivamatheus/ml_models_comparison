# CARREGANDO PACOTES ------------------------------------------------------
library(frbs); library(tidyverse);library(MLmetrics); library(tidymodels)
library(ROCR); 


# FUNCOES
load_df = function(filename, amostras){
  fnames = paste0(filename,'/cd_',amostras,'.csv')
  my_data <- list()
  for (i in seq_along(fnames)) {
    csv = paste0('Dados/',filename,'.zip')
    my_data[[i]] <- read.csv(unz(csv, fnames[i]))
    print(fnames[i])
  }
  names(my_data) = fnames
  return(my_data)
}

label_encorder = function(x){
  return(as.numeric(factor(x)))
}


pre_processing = function(df){
  set.seed(1353)
  #Preprocessing
  df = df %>% select('step':'isFraud')
  categoricals = c("action", "nameOrig", "nameDest")
  numericals = c('step','amount', 'oldBalanceOrig', 
                 'newBalanceOrig', 'oldBalanceDest',
                 'newBalanceDest')
  
  df[, categoricals] = apply(df[, categoricals], 2, label_encorder)
  df[, numericals] = apply(df[, numericals], 2, scale)
  df$isFraud = ifelse(df$isFraud == 0, 1, 2)
  
  df_split <- initial_split(df, prop=.8, strata='isFraud')
  train_data <- training(df_split)
  test_data <- testing(df_split)
  
  return(
    list(train_data[, -ncol(train_data)],
         train_data$isFraud,
         test_data)
  )
}

fit_fuzzy_model = function(X_train, y_train, test){
  
  now = Sys.time()
  # df = rbind(X_train, test[, -ncol(test)])
  # amplitudes = apply(df[, -ncol(df)], 2, range)
  amplitudes = apply(X_train, 2, range)
  # remove(df)
  
  # CONTROLADOR FUZZY  ------------------------------------------------------
  controlador = list(num.labels=4,
                     type.mf="SIGMOID",
                     type.tnorm="PRODUCT",
                     type.implication.func="ZADEH",
                     type.defuz='COG',
                     num.class=3,
                     name='Fraudes')
  
  # MODELAGEM ---------------------------------------------------------------
  fuzzy_classificacao <- frbs.learn(data.train=cbind(X_train, y_train), 
                                    range.data=amplitudes,
                                    method.type="FRBCS.CHI",
                                    control=controlador)
  
  y_test = test$isFraud
  
  if((length(unique(y_test)) > 1)){
    preds = predict(fuzzy_classificacao, test[, -ncol(test)])
  }else{
    preds = NA
  }
  
  #Fixing labels
  y_test = ifelse(y_test == 2, 1, 0)
  preds = ifelse(preds == 2, 1, 0)
  
  if((length(unique(preds)) == 1) | (length(unique(y_test)) == 1)){
    #Getting metrics
    RECALL = NA
    aUc = NA
    f_half = NA
    f1_score = NA
    f2_score = NA
    
    #Getting more information
    modelo = 'FRBS'
    tp = NA
    tn = NA
    fp = NA
    fn = NA
    N_NEG = sum(y_test == 0)
    N_POS = sum(y_test == 1)
    
    results = data.frame(aUc, RECALL, f_half,  f1_score, f2_score, 
                         tn, fn, fp, tp, modelo, N_NEG,N_POS)
    
    names(results) = c('AUC', 'RECALL', 'FHALF', 'F1', 'F2','TN','FN','FP','TP',
                       'MODEL', 'N_NEG', 'N_POS')  
    
  }else{
    #Getting metrics
    RECALL = Recall(y_true=y_test, y_pred=c(preds))
    aUc = AUC(y_true=y_test, y_pred=preds)
    f_half = FBeta_Score(y_true=y_test, y_pred=preds, beta = .5)
    f1_score = FBeta_Score(y_true=y_test, y_pred=preds, beta = 1)
    f2_score = FBeta_Score(y_true=y_test, y_pred=preds, beta = 2)
    
    #Getting more information
    CM = ConfusionMatrix(y_pred = preds, y_true = y_test)
    modelo = 'FRBS'
    tp = CM[2,2]
    tn = CM[1,1]
    fp = CM[1,2]
    fn = CM[2,1]
    N_NEG = sum(y_test == 0)
    N_POS = sum(y_test == 1)
    
    results = data.frame(aUc, RECALL, f_half,  f1_score, f2_score, 
                         tn, fn, fp, tp, modelo, N_NEG,N_POS)
    
    names(results) = c('AUC', 'RECALL', 'FHALF', 'F1', 'F2','TN','FN','FP','TP',
                       'MODEL', 'N_NEG', 'N_POS')
  }
  
  cat('Tempo de ajuste e resultados: ',(Sys.time() - now))
  return(results)
}

run_through_samples = function(cenario){
  
  results = matrix(data=NA, nrow = 0, ncol = 12)
  results = as.data.frame(results)
  names(results) = c('AUC', 'RECALL', 'FHALF', 'F1', 'F2','TN','FN','FP','TP',
                     'MODEL', 'N_NEG', 'N_POS')
  name_out = names(cenario)[1]
  name_out = substr(name_out, start = 1, stop = 8)
  
  
  for (cd in names(cenario)) {
    df = cenario[[cd]]
    #dados de treino e de teste
    df = pre_processing(df)

    # Pular amostras que não contém as duas classes
    nclasses_train = length(unique(df[[2]]))
    nclasses_test = length(unique(df[[3]]$isFraud))

    if((nclasses_train < 2) | (nclasses_test < 2)){next}

    #Modelagem e resultados
    df_tmp = fit_fuzzy_model(X_train=df[[1]],
                             y_train=df[[2]],
                             test=df[[3]])
    df_tmp$AMOSTRA = cd

    results = rbind(results, df_tmp)
    print(cd)
    
    output_tmp = paste0('Resultados/frbs_results_tmp_',name_out,'.csv')
    print(output_tmp)
    write.csv(results, file=output_tmp, row.names = F)
    
  }
  
  output = paste0('Resultados/FRBS_results_',name_out,'.csv')
  write.csv(results, file=output, row.names = F)
  return(results)
  
  
}

