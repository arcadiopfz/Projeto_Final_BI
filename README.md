# Análise de investimentos para ações por meio da Simulação de Monte Carlo, Séries Temporais e Redes Neurais LSTM

#### Aluno: [Arcadio de Paula Fernandez](https://github.com/arcadiopfz/)
#### Orientadora: [Leonardo Alfredo Forero Mendoza](https://github.com/leofome8)

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

---

### Agradecimentos
Gostaria de agradecer todos os meus familiares, em especial ao meu querido pai (in memoriam) e a minha querida esposa, que sempre me apoiaram nos meus projetos e no meu percurso acadêmico. 

Agradeço aos meus filhos que me inspiraram a não desistir.

Agradeço aos professores do curso de pós-graduação em Business Intelligence: Sistemas Inteligentes de Apoio à Decisão em Negócios da PUC-Rio, principalmente, meu orientador Dr. Leonardo Alfredo Forero Mendoza.

---

### Resumo


No processo de investir em ações, muitos investidores utilizam informações qualitativas sobre economia e política, analisam indicadores financeiros e examinam gráficos para saber qual é o melhor momento para realizar suas transações.

Nesse contexto, os investidores reagem exageradamente ao estresse e tomam decisões pela emoção durante períodos de extrema turbulência no mercado financeiro e consequentemente realizam prejuízos 

Para alcançar resultados mais promissores, investidores estão cada vez mais utilizando Artificial Intelligence (AI), Machine Learning (ML) e Deep Learning (DL) no mercado financeiro. 

A finalidade deste trabalho é o de criar modelos utilizando Simulação Monte Carlo, Séries Temporais e Redes Neurais LSTM, para que seja possível definir o momento mais adequado a se investir em uma ação. 

**Palavras chave:** Mercado financeiro, Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), Simulação Monte Carlo, Séries Temporais e Redes Neurais LSTM


### Abstract

In the process of investing in stocks, many investors use qualitative information about the economy and politics, analyze financial indicators and examine charts to find out what is the best time to carry out their transaction.

In this context, investors overreact to stress and make decisions based on emotion during periods of extreme turbulence in the financial market and, consequently, make losses.

To achieve more promising results, investors are increasingly using Artificial Intelligence (AI), Machine Learning (ML) and Deep Learning (DL) in the financial market.

The purpose of this work is to create models using Monte Carlo Simulation, Time Series and LSTM Neural Networks, so that it is possible to define the most appropriate moment to invest in a stock.

**Keywords:**  Financial market, Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), Monte Carlo Simulation, Time Series and LSTM Neural Networks

---

### 1.	Introdução 

####  1.1.	Motivação

O livro “O Homem que decifrou o mercado”, conta a história de como o matemático Jim Simons, formado pelo Massachusetts Institute of Technology - MIT, montou a Renaissance Technologies, uma das maiores corretoras de Investimentos Quantitativos da América do Norte. Sendo que, desde 1988, o fundo Medallion, exclusivo da Renaissance, tem gerado uma média de ganhos anuais de 66% e a empresa registrou lucros de mais de 100 bilhões de dólares com as negociações. Devido a esse sucesso, o próprio Simons tem uma fortuna estimada em 23 bilhões de dólares.

A Renaissance Technologies, navega muito no universo da Artificial Intelligence (AI), Machine Learning (ML) e Deep Learning (DL) para alcançar tais resultados e é uma referência na revolução Quant que se alastrou por Wall Street, Vale do Silício etc.

Enxergar que AI, ML e DL são muito utilizados no mercado financeiro, conforme mencionado pelo Prof. Leonardo Mendoza durante suas aulas, chamou a atenção. 

Neste sentido, o presente trabalho visa percorrer algumas das ferramentas repassadas no transcorrer do curso de BI, para que seja possível atender a necessidade de analisar investimentos em ações de uma forma mais ampla, ou seja, percorrendo técnicas e conceitos da pesquisa tradicional (Traditional Research) na área de finanças, de Machine Learning e de Deep Learning, conforme pode ser observado na figura 1 abaixo.

<div align="center">
<img src="https://user-images.githubusercontent.com/122412860/215283444-f379c9a6-7929-4378-ab50-288da6baf1f4.jpg" width="400px" />
</div>

#### 1.2.	Identificação do Problema

No processo de investimentos em ações muitos investidores utilizam informações qualitativas sobre economia, política, analisam indicadores financeiros e examinam gráficos para saber qual é o melhor o momento para realizar essa transação. “O problema é que os investidores reagem exageradamente ao estresse e tomam decisões pela emoção”. (ZUCKERMAN, 2020 p. 154). 

Nesse contexto, é provável que não seja coincidência que muitos fundos quantitativos, que utilizam AI, ML e DL para realizar suas transações, tenham obtido maiores lucros durante períodos de extrema turbulência no mercado financeiro. 

#### 1.3.	Objetivo 

Levando em conta que, a pesquisa tradicional (Traditional Research) na área de finanças passa pela Simulação Monte Carlo, que é uma série de cálculos de probabilidade que estimam a chance de um evento futuro acontecer, isto é, são feitas diversas simulações para calcular probabilidades de um acerto ou perda.

Considerando que, AI, ML e DL são técnicas que tem como objetivo modelar o modo de raciocínio do ser humano ao tomar decisões em um ambiente de incerteza e imprecisão. 

Tendo em vista que, dentro do universo de ML a análise de ações/investimentos frequentemente utiliza aprendizado supervisionado recorrendo ao estudo de séries temporais para constatar padrões não aleatórios em uma série e observar se este comportamento passado permiti fazer previsões sobre o futuro, orientando a tomada de decisões.

Considerando que, no contexto de DL existem as redes neurais recorrentes (RNN — Recurrent Neural Networks) que é um grupo de redes neurais voltadas para o processamento de dados sequenciais. Sendo que, uma variante das RNNs convencionais é a LSTM (Long Short-Term Memory), que tem uma capacidade bem conhecida de processar dados sequenciais, como são as séries temporais de preços de ações do mercado de financeiro. 

Considerando que, cabe definir como referencial para a base de dados uma empresa com bastante informações qualitativas e com impacto na economia e política, bem como com indicadores financeiros expressivos, optou-se pela  Petro Rio SA (PRIO3)

O objetivo deste trabalho é o de criar modelos utilizando Simulação Monte Carlo, Séries Temporais e Redes Neurais LSTM, para que seja possível definir o momento mais adequado a se investir em uma ação do mercado financeiro, no presente caso a PRIO3. 


### 2.	Avaliação e análise dos dados disponíveis (Pré-processamento)

No pré-processamento cabe ressaltar as seguintes atividades: 

•	Download market data do Yahoo! Finance API

•	Download de informações do site investing.com

•	Convert interger type to float

•	Verificar se existem valores nulos 

•	Criando um gráfico de histórico do preço da ação PRIO.SA

•	Convertendo Dates and Times

•	Normalizar os dados para análise do crescimento das ações

•	Comparando o histórico do preço da ação PRIO.SA (PRIO3.SA) com outras 6 ações. 

Os datasets utilizados neste trabalho estão disponíveis em formato csv no [diretório dados](https://github.com/arcadiopfz/Projeto-Final/tree/main/Dados%20do%20Projeto).

### 3.	Modelagem 

Todas as etapas do projeto foram feitas utilizando ambiente Colab (Google Colaboratory) e a Linguagem de Programação foi o Python.

Tendo em consideração o objetivo, o trabalho foi organizado em 4 etapas: 

    Etapa 1 - Análise exploratória: Pré-processamento e visualização do conjunto de dados da ação da Petro Rio SA (PRIO3)

    Etapa 2 - Simulação Monte Carlo para previsão de preços da ação da Petro Rio SA (PRIO3)

    Etapa 3 - Séries temporais para previsão de preços da ação Petro Rio SA (PRIO3) utilizando o Facebook Prophet

    Etapa 4 - Análise dos preços da ação Petro Rio SA (PRIO3) por meio de Redes Neurais LSTM

Sendo que a Modelagem, consta nas etapas 2, 3 e 4, como pode ser observado no esquema básico do Projeto na figura 2, a seguir: 

<div align="center">
<img src="https://user-images.githubusercontent.com/122412860/215283921-ba8cab3e-4d6b-4bb2-9c81-4ac54c4fb7b0.png" width="600px" />
</div>

### 4.	Avaliação 

Para a previsão do preço da ação da PRIO.SA nos próximos 30 dias utilizando o Método de Monte Carlo com 100 simulações, foi observado que: 

Com 50% de probabilidade, o preço seria maior que R$ 37.86582123478222.

Com 95% de probabilidade, o preço seria maior que R$ 28.103680220487078.
 
Com 99% de probabilidade, o preço seria maior que R$ 24.276331471103337.

No caso das Séries temporais para previsão de preços da ação Petro Rio SA (PRIO3) utilizando o Facebook Prophet, fica evidente que a previsão do modelo (curva azul) segue a tendência dos preços reais (pontos pretos), conforme pode ser observado na figura, a seguir: 

 <div align="center">
<img src="https://user-images.githubusercontent.com/122412860/215283933-984956c2-20e9-4a06-9a7d-c23715b6aa9b.png" width="400px" />
</div>


O modelo de Séries temporais com previsão de preços da ação Petro Rio SA (PRIO3) para 30 dias utilizando o Facebook Prophet mantem uma tendência entorno de R$ 32, como verificado na figura abaixo: 
 
 <div align="center">
<img src="https://user-images.githubusercontent.com/122412860/215283945-c1f261e0-11d7-4b86-aae7-fa92c360f17d.png" width="400px" />
</div>

No modelo Análise dos preços da ação Petro Rio SA (PRIO3) por meio de Redes Neurais LSTM, fica inequívoco que tanto a curva de previstas de teste e train seguem a tendência do valor da ação, como pode ser observado na figura, a seguir: 

<div align="center">
<img src="https://user-images.githubusercontent.com/122412860/215283954-fb0fe13e-eaba-4851-9282-4bfd0a88b8b1.png" width="400px" />
</div>


### 5. Conclusões 

Os modelos de Simulação Monte Carlo, Séries temporais e Redes Neurais LSTM foram eficientes na previsão do comportamento da ação da Petro Rio SA (PRIO3). 

Considerando que, a Lógica Fuzzy é uma boa técnica para modelar o modo de raciocínio do ser humano ao tomar decisões em um ambiente de incerteza e dúvida. Atentando que, o Reinforcement Learning serve para automação e que é um substituto mais avançado do Fuzzy.

Tendo em vista que, o Deep Reinforcement Learning (Deep RL) é a combinação de Reinforcement Learning (RL) e Deep Learning.

Levando em conta que, o principal desafio do Reinforcement Learning está em como o agente deve ser treinado. Em vez de analisar os dados fornecidos, o modelo interage com o ambiente, buscando formas de maximizar a recompensa. No caso do Deep Reinforcement Learning (Deep RL), uma rede neural se encarrega de armazenar as experiências e, assim, melhorar a forma como a tarefa é executada.

Isto posto, uma possível evolução do trabalho seria a inclusão da análise de compra ou venda de uma ação por meio de Deep Reinforcement Learning (Deep RL).


### 6.	Referências  

CHOLLET, François. Deep Learning with Python. Second Edition. USA, NY: Manning Publications Co., 2021.

GÉRON, Aurélien. Mãos A Obra: Aprendizado De Máquina Com Scikit-Learn, Keras & TensorFlow, 2. ed. Rio de Janeiro: Alta Books, 2021.

Granatyr, Jones. Curso: Python para Finanças: Análise de Dados e Machine Learning. IA Expert Academy, 2022.

HILPISCH, Yves. Python for Finance: Mastering Data Driven Finance. Second Edition. Canada, Sebastopol: O’Reilly Media, December 2018.

Ilango, Jack Praveen Raj. Disponível em: https://github.com/jackpraveenraj/Stock-Prediction-Using-Stacked-LSTM/blob/main/Stock%20Prediction%20Using%20LSTM.ipynb. Acesso em: janeiro de 2023.

JANSEN, Stefan. Machine Learning for Algorithmic Trading: Predictive models to extract signals from market and alternative data for systematic trading strategies with Python. 2nd Edition. Birmingham: Packt Publishing Ltd., July 2020

Mendoza, Leonardo Alfredo Forero. Bitcoin_Tweets_Sentimental__cnn_lstmvf.ipynb. Notas de aula -RN 8. Não paginado.

Oliveira, Ivan Madeira. Disponível em: https://github.com/secretaria-ICA/ANALISE_DE_INVESTIMENTOS_PARA_CRIPTOMOEDAS_CRYPTO_VALUATION_O_PROJETO_CARDANO_ADA. Acesso em: dezembro de 2022. 

Rocha Júnior, Maximiliano Augusto. Disponível em: https://github.com/secretaria-ICA/Predicao_de_Demandas_de_Estoque_com_o_Uso_de_Recurrent_Neural_Network_LSTM. Acesso em: janeiro de 2023.

Rocha, Junior. Disponível em: https://github.com/secretaria-ICA/Previsao_de_Precos_de_Commodities_Agricolas_Atraves_de_Redes-Neurais_LSTM. Acesso em: janeiro de 2023. 

ZUCKERMAN, Gregory. O homem que decifrou o mercado: como Jim Simons criou a revolução quant. 2. ed.- Rio de Janeiro: Alta Books, 2020.

---


Matrícula: 212.100.462

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
