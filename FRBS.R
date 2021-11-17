########################## BASE DE REGRAS VIA AI ######################

source("modelling/FRBS_MODELLING.R")

d = load_df(filename='cenario15', amostras=c(1:100))

# write.csv(d[[1]], 'amostra104.csv', row.names=F)
r = run_through_samples(d)
