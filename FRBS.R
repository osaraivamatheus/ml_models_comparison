########################## BASE DE REGRAS VIA AI ######################

source("modelling/FRBS_MODELLING.R")

d = load_df(filename='cenario2', amostras=c(1:1))

run_through_samples(d)
