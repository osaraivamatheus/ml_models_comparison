# Repósitório para o desenvolvimento do código referente a comparação de modelos de classificação para detecção de fraudes financeiras

Neste repositório se encontram todos os códigos, banco de dados e análises realizadas referentes ao desenvolvimento da metodologia e dos resultados de minha dissertação de mestrado.

## Dados

Os dados utilizados neste repositório são dados artificiais gerados pelo simulador AMLSim, [disponível aqui](https://github.com/EdgarLopezPhD/PaySim). A partir de diferentes parametrizações 2 cenários foram gerados, cada um com 1.000 amostras:

|           Parâmetro           	| Cenário 1 	| Cenário 2 	|
|:-----------------------------:	|:---------:	|:---------:	|
| Horizonte temporal (em horas) 	|     1.000 	| 1.000.000 	|
| Número de clientes            	|       600 	|   100.000 	|
| Número de fraudadores         	|         5 	|       150 	|
| Número de bancos              	|         4 	|        50 	|
| Número de transações          	|     10000 	| 5.000.000 	|
| Probabilidade de Fraude       	|     0,001 	|   0,00001 	|
| Limite de transferência       	|    10.000 	|    10.000 	|
| Conjunto de dados gerados     	|      1000 	|       100 	|

Devido ao limite de armazenamento existente em repositórios do GitHub os dados disponibilizados aqui possuem apenas alguns subconjuntos. Para ter acesso a todos os conjuntos de dados gerados entre em contato através do email osaraivamatheus@gmail.com.

Nos scripts de execução *ML_MODELS.py* e *FRBS.R* cada cenário é inserido de maneira compactada em formato .zip.

## Modelos

os modelos utilizados neste projeto são:
- Regressão Logística
- Random Forests
- Redes Neurais Artificiais (MLP)
- Support Vector Machine (SVM)
- Extreme Gradiente Boosting Machine (XGB)
- Sistemas Baseados em Regras Fuzzy (SBRF)


## Estrutura

Por causa de facilidades existêntes em cada linguagens, além de pacotes específicos, as modelagens são realizadas através das linguagens de programação Python e R.

### Modelagens utilizando Python

Com excessão dos Sistemas Baseados em Regras Fuzzy, todos os demais modelos executados em linguagem python. Os ajustes, predições e avaliações de desempenho de todos os modelos, são feitos através da biblioteca [Scikit-learn](https://scikit-learn.org/stable/).  

### Modelagens utilizando R

O modelo SBRF é ajustado e executado sob a liguagem de programação R, através da utilzação dos pacotes [frbs](https://cran.r-project.org/web/packages/frbs/frbs.pdf) e  [Tidyverse](https://www.tidyverse.org/).

### Integração de códigos

A execução deste projeto é dada através dos arquivos *ML_MODELS.py* e *FRBS.R*, que carregam e executam os seguintes scrips em python e R, contidos no diretório "/modelling/":
- **pre_process.py**: Rotina criada para o tratamento de dados amostrais e separação dos mesmos em cojuntos de treino e de teste.
- **fit_models.py**: Rotina criada para ajustar modelos e gerar os resultados referentes aos seus desempenhos em predição, sob as métricas de AUC, Acurácia Balanceada, F1-score e F2-score. 
- **run.py**: Este script carrega e executa os scripts anteriores em um conjunto de amostras. Parte deste script carrega parâmetros de modelos já hiperparametrizados, que se encontram no diretório "/Hiperparametrizados/".
- **FRBS_MODELLING.R**: Script para executar os modelos baseados em regras fuzzy.

Os arquivo *ML_models.py* e *FRBS.R* são responsáveis pela integração dos demais scripts, parametrização e execução dos modelos propostos. Durante sua execução uma tabela em formato .csv é gerada, permintindo análises dos resultados parciais enquanto todo o conjunto de amostras ainda não foi finalizado. Ao fim de sua execução, uma tabela final em formato .csv é gerada com todos os resultados de todos os modelos em todas as amostras; armazeanada no diretório "/Resultados/".
