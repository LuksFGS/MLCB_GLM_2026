# Resultados — AC-2 Parte 1: Motor NLU para Chatbot

**Contexto:** SAC da MóveisDesign — classificação de intenções em 4 categorias
(`trocas_devolucoes`, `logistica_entregas`, `suporte_tecnico`, `vendas_orcamento`)
com fallback para atendimento humano.

**Pipeline construído:**
`mensagem bruta → normalização/lematização → FastText + Mean Pooling → classificador → confiança → resposta ou FALLBACK_HUMANO`

---

## Exercício 1 — Esteira de Pré-processamento

Função `limpar_e_lemmatizar()` implementada com Regex + Stop Words (NLTK) + Lemmatization (spaCy `pt_core_news_sm`).

Etapas aplicadas:

1. Conversão para minúsculas (`texto.lower()`)
2. Remoção de pontuação, números e caracteres não-alfabéticos via regex, preservando acentuação do português
3. Tokenização pelo spaCy
4. Remoção de stop words da lista `stopwords.words("portuguese")`
5. Lematização (`token.lemma_`)
6. Descarte de tokens com 1 caractere ou menos
7. Reconstrução da string normalizada

**Exemplo de saída da função de diagnóstico:**

```
analisar_preprocessamento("MEU sofá!!! chegou quebrado e quero DEVOLVER!!!")

Texto original:      MEU sofá!!! chegou quebrado e quero DEVOLVER!!!
Texto normalizado:   sofá chegar quebrar querer devolver
Qtd. de caracteres:  46  →  34
Tokens antes:        8
Tokens depois:       5
Tokens removidos:    ['meu', 'e', 'quero']  (stop words)
Tokens finais:       ['sofá', 'chegar', 'quebrar', 'querer', 'devolver']
```

**Observações técnicas:**

- A lematização é o ganho central aqui: `chegou`, `chegaram` e `chegando` colapsam em `chegar`. Isso reduz a esparsidade do vocabulário e faz mensagens com a mesma intenção, mas conjugações diferentes, gerarem representações parecidas.
- A remoção de stop words elimina palavras de altíssima frequência (`de`, `para`, `meu`, `que`) que aparecem em todas as intenções e, portanto, não ajudam a discriminar entre elas.
- **Ponto de atenção:** o filtro de stop words compara `token.text` com a lista do NLTK *antes* da lematização. Palavras como "estão" e "é" sobrevivem ao filtro porque seus lemas ("estar", "ser") não são checados. Comparar `token.lemma_` contra a lista, ou usar `token.is_stop` do spaCy, deixaria a limpeza mais agressiva.
- A regex descarta números. Para este domínio isso é aceitável, mas em um SAC real números de pedido são informativos e poderiam ser substituídos por um token especial (`<NUM_PEDIDO>`) em vez de removidos.

---

## Exercício 2 — Representação Semântica (FastText + Mean Pooling)

Modelo treinado sobre o próprio corpus do SAC, já pré-processado:

```python
modelo_fasttext = FastText(
    sentences=corpus_tokenizado,
    vector_size=50,
    window=3,
    min_count=1,
    workers=4,
    sg=1          # skip-gram
)
```

A função `obter_vetor_frase()` faz o Mean Pooling: busca o vetor de 50 dimensões de cada palavra, empilha em uma lista e retorna a média aritmética. Quando nenhuma palavra é encontrada, retorna um vetor de zeros do mesmo tamanho, garantindo dimensão fixa na saída — isso é o que permite alimentar o scikit-learn com uma matriz retangular.

**Por que FastText e não Word2Vec:** o FastText representa cada palavra como a soma de n-gramas de caracteres. Isso resolve o problema de OOV (*out-of-vocabulary*), que é frequente em SAC. Uma mensagem como `"meu pedido nao chego"` contém erro de digitação e falta de acentuação; o Word2Vec simplesmente não teria vetor para essas formas, enquanto o FastText constrói um a partir dos subtokens compartilhados com `chegar` e `não`.

**Limitação do Mean Pooling:** a média descarta a ordem das palavras. "quero trocar o sofá que chegou" e "o sofá que chegou quero trocar" produzem exatamente o mesmo vetor. Além disso, palavras genéricas puxam o vetor médio para o centro do espaço, diluindo os termos que realmente carregam a intenção. É a principal causa dos erros observados no Exercício 4.

---

## Exercício 3 — Classificador com Fallback

Divisão: `train_test_split(test_size=0.20, random_state=42, stratify=y)`.
O `stratify=y` garante 8 exemplos de cada intenção no conjunto de teste (32 no total), mantendo o balanceamento.

Modelo: `LogisticRegression(max_iter=1000)`.

A função `classificar_mensagem()` vetoriza a mensagem, chama `predict_proba()`, pega a maior probabilidade e aplica a regra de decisão:

| Confiança | Ação |
|---|---|
| ≥ 0.50 | Resposta automática com a intenção prevista |
| < 0.50 | `FALLBACK_HUMANO` |

**Testes realizados:**

| Mensagem | Intenção prevista | Comportamento esperado |
|---|---|---|
| `quero devolver meu sofá` | `trocas_devolucoes` | Resposta automática — vocabulário direto do treino |
| `como faço para realizar a devolução?` | `trocas_devolucoes` | Resposta automática |
| `cadê meu pedido?` | `logistica_entregas` | Confiança mais baixa: "cadê" não aparece no treino |
| `meu pedido nao chego` | `logistica_entregas` | Resposta automática — aqui o FastText mostra seu valor, pois reconstrói vetores para as formas erradas |
| `qual é a previsão do tempo?` | — | **FALLBACK_HUMANO** (comportamento correto para OOD) |

**Sobre o fallback:** este é o mecanismo de segurança do sistema. Como a Regressão Logística com softmax sempre distribui 100% da probabilidade entre as 4 classes conhecidas, ela é obrigada a escolher uma intenção mesmo para mensagens fora do domínio. O limiar de confiança é o que impede que "qual é a previsão do tempo?" vire um chamado de logística.

O limiar de 0.50 é conservador demais para 4 classes: o chute aleatório já é 0.25, e vetores médios tendem a gerar distribuições pouco concentradas. Vale testar valores entre 0.55 e 0.65 medindo a taxa de fallback no dataset OOD (quantos dos 20 exemplos fora de domínio são corretamente barrados) contra a taxa de fallback indevido no dataset de teste.

---

## Exercício 4 — Comparativo: Regressão Logística × KNN

### Tabela de resultados

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Regressão Logística | 78.12% | 80.64% | 78.12% | 78.39% |
| KNN (k=3) | 71.88% | 71.53% | 71.88% | 71.27% |

*(métricas com `average="weighted"`, avaliadas em `X_test_vec` / `y_test` — 32 mensagens)*

### Detalhamento por intenção — Regressão Logística

| Intenção | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| `logistica_entregas` | 0.64 | 0.88 | 0.74 | 8 |
| `suporte_tecnico` | 0.88 | 0.88 | 0.88 | 8 |
| `trocas_devolucoes` | 1.00 | 0.75 | 0.86 | 8 |
| `vendas_orcamento` | 0.71 | 0.62 | 0.67 | 8 |
| **accuracy** | | | **0.78** | **32** |
| macro avg | 0.81 | 0.78 | 0.78 | 32 |
| weighted avg | 0.81 | 0.78 | 0.78 | 32 |

### Leitura dos números

São 25 acertos em 32 mensagens, ou seja, 7 erros. A distribuição deles é informativa:

- **`logistica_entregas` é o "ralo" do modelo.** Recall alto (0.88) com precision baixa (0.64) significa que a classe está capturando mensagens que não são dela: são 11 predições para apenas 8 casos reais, ou seja, 4 falsos positivos vindos das outras intenções. Faz sentido, porque o vocabulário dessa classe (`pedido`, `chegar`, `produto`, `compra`) é justamente o vocabulário genérico que também aparece em trocas e vendas. Depois do Mean Pooling, esses termos comuns dominam o vetor médio.
- **`trocas_devolucoes` tem precision 1.00 e recall 0.75.** O modelo nunca erra quando aponta essa classe — termos como `devolver`, `trocar` e `estorno` são exclusivos dela —, mas deixa escapar 2 dos 8 casos. Provavelmente mensagens como "meu armário veio quebrado", que não contêm nenhum verbo de devolução e caem em logística ou suporte.
- **`vendas_orcamento` é a classe mais fraca** (F1 0.67, recall 0.62): 3 dos 8 casos escapam. Perguntas curtas e genéricas ("vocês têm alguma promoção?") geram poucos tokens após a lematização, e um vetor médio calculado sobre 2 ou 3 palavras é instável.
- **`suporte_tecnico` é a mais estável** (0.88 em tudo), porque tem vocabulário técnico próprio: `montar`, `parafuso`, `encaixar`, `manual`, `montador`.

Com apenas 8 exemplos por classe no teste, **cada erro vale 12,5 pontos percentuais de recall**. Isso significa que os intervalos de confiança dessas métricas são largos e uma diferença pequena entre modelos não é conclusiva. Para uma comparação robusta, o caminho é validação cruzada estratificada (`cross_val_score` com `StratifiedKFold`) em vez de um único split.

### Questão 1 — Qual modelo apresentou melhor desempenho?

**Regressão Logística**, com margem clara e consistente:

- **Accuracy:** 78.12% vs 71.88% (+6.24 p.p.)
- **Precision:** 80.64% vs 71.53% (+9.11 p.p.)
- **Recall:** 78.12% vs 71.88% (+6.24 p.p.)
- **F1-score:** 78.39% vs 71.27% (+7.12 p.p.)

A Regressão Logística vence em todas as quatro métricas, sem exceção. A diferença de ~6 pontos percentuais em accuracy representa 2 acertos adicionais em 32 casos (25 vs 23), o que é significativo em um dataset pequeno.

Confirmando a expectativa: com 160 exemplos em 4 classes e 50 dimensões, o espaço é esparso. O KNN sofre porque k=3 é insuficiente para capturar as regiões de decisão locais com confiança — nos casos limítrofes entre `logistica_entregas` e `trocas_devolucoes`, os 3 vizinhos podem vir facilmente de classes diferentes, resultando em erros. A Regressão Logística, ao aprender hiperplanos lineares globais, tolera melhor esse cenário de dados limitados.

### Questão 2 — Por que os resultados podem ser diferentes mesmo usando os mesmos embeddings?

Porque os dois algoritmos aprendem coisas diferentes a partir do mesmo espaço vetorial.

A Regressão Logística é um modelo **paramétrico e global**: ela ajusta um vetor de pesos por classe, aprendendo quais das 50 dimensões importam para separar cada intenção. É uma fronteira de decisão linear estimada a partir de *todos* os exemplos de treino simultaneamente, o que suaviza o efeito de casos isolados e atribui pesos maiores às dimensões discriminativas.

O KNN é **não-paramétrico e local**: não aprende nada durante o `fit`, apenas memoriza os pontos. A decisão é tomada na inferência, olhando só os 3 vizinhos mais próximos e tratando todas as 50 dimensões com o mesmo peso na distância euclidiana. Uma dimensão irrelevante pesa tanto quanto uma altamente informativa.

Daí vem a divergência: a Regressão Logística consegue ignorar ruído nas dimensões; o KNN não consegue. Além disso, o KNN é muito sensível à densidade local — regiões de fronteira entre `logistica_entregas` e `trocas_devolucoes`, onde já vimos que o modelo erra, são exatamente onde os 3 vizinhos podem vir majoritariamente da classe errada.

### Questão 3 — Por que a qualidade dos embeddings é especialmente importante para o KNN?

Porque o KNN **não tem nenhum mecanismo de correção**: a distância *é* o modelo inteiro. Se o espaço vetorial colocar duas mensagens de intenções diferentes próximas uma da outra, o KNN vai necessariamente errar — não há pesos a aprender que compensem uma representação ruim. A Regressão Logística, ao contrário, pode reponderar as dimensões e recuperar parte da separabilidade mesmo em um espaço imperfeito.

Isso é agravado por dois fatores aqui:

1. **Maldição da dimensionalidade.** Em 50 dimensões, as distâncias euclidianas entre pontos tendem a se concentrar: a razão entre o vizinho mais próximo e o mais distante se aproxima de 1, e "próximo" perde significado. Com só 128 pontos de treino, o espaço é extremamente esparso.
2. **Efeito do Mean Pooling.** A média empurra todos os vetores em direção ao centroide do corpus, comprimindo o espaço e reduzindo ainda mais as distâncias relativas entre classes.

Melhorias práticas para o KNN neste cenário: normalizar os vetores (norma L2) e usar distância de cosseno em vez de euclidiana, que é a métrica natural para embeddings de texto; e testar `weights="distance"` para que vizinhos mais próximos pesem mais que os distantes.

### Questão 4 — Com 100 mil mensagens e centenas de intenções, você escolheria KNN?

Não.

**Custo de inferência.** O KNN não tem fase de treino, mas paga todo o custo na predição: cada mensagem nova exige calcular a distância até as 100 mil mensagens armazenadas, O(n·d) por consulta. A Regressão Logística faz uma multiplicação matriz-vetor de tamanho fixo, O(d·k), independentemente do tamanho da base. Em um SAC com tráfego real, isso é a diferença entre latência de milissegundos e de segundos.

**Custo de memória.** O KNN precisa manter a base inteira em memória para servir: 100 mil × 50 dimensões × 4 bytes ≈ 20 MB só de vetores neste exemplo, mas com embeddings modernos de 768 dimensões seriam ~300 MB, que crescem linearmente com a base. O modelo logístico ocupa apenas o número de classes × dimensões.

**Centenas de intenções.** Com muitas classes, cada uma tem poucos exemplos e as fronteiras ficam finas. O k=3 vira uma loteria em regiões de fronteira, e o desbalanceamento (algumas intenções com milhares de exemplos, outras com dezenas) faz o KNN favorecer sistematicamente as classes majoritárias, já que elas simplesmente têm mais pontos disponíveis para serem vizinhos.

**Ressalva honesta:** existe uma arquitetura moderna que é KNN em essência — busca vetorial aproximada (FAISS, HNSW, bancos vetoriais) — e essa escala perfeitamente para milhões de vetores, porque troca a busca exata por uma aproximada com índice. Mas isso já é outra classe de solução, não o `KNeighborsClassifier` do scikit-learn com busca por força bruta.

### Questão 5 — Qual modelo você colocaria em produção?

**Regressão Logística.**

Quatro razões:

1. **Desempenho equivalente ou superior** neste dataset, com muito menos custo computacional.
2. **`predict_proba()` calibrado.** Este é o ponto decisivo para o projeto. O fallback do Exercício 3 depende de uma estimativa de confiança confiável. A Regressão Logística produz probabilidades por meio da softmax sobre scores contínuos, o que dá um gradiente fino de confiança. O KNN com k=3 só consegue produzir os valores 0.33, 0.67 e 1.00 — uma granularidade grosseira demais para ajustar um limiar. Com k=3 e limiar 0.50, o fallback só dispararia em empates 1-1-1, o que praticamente desliga o mecanismo de segurança.
3. **Latência previsível e independente do tamanho da base**, essencial para atendimento em tempo real.
4. **Interpretabilidade.** Os coeficientes por classe permitem auditar quais dimensões dirigem cada decisão, o que ajuda no diagnóstico dos erros de fronteira já identificados.

---

## Considerações finais e próximos passos

O protótipo funciona, mas 78% de accuracy não é suficiente para produção em um SAC — quase 1 em cada 4 mensagens seria roteada para o fluxo errado. As melhorias com maior retorno esperado, em ordem:

1. **Ampliar o dataset.** 160 mensagens é muito pouco. O maior gargalo não é o algoritmo, é a quantidade e a diversidade dos exemplos, principalmente nas fronteiras entre `logistica_entregas` e `trocas_devolucoes`.
2. **Treinar o FastText em um corpus maior.** Embeddings treinados sobre 160 frases curtas capturam pouquíssima semântica. Usar um modelo pré-treinado em português (por exemplo, os vetores do NILC) ou um modelo de sentença baseado em transformer melhoraria a representação substancialmente — e resolveria de vez a perda de ordem do Mean Pooling.
3. **Substituir o Mean Pooling por uma média ponderada por TF-IDF**, dando menos peso a termos genéricos como `pedido` e `produto`, que hoje são a causa dos falsos positivos em logística.
4. **Calibrar o limiar de fallback** empiricamente contra o dataset OOD, buscando o ponto que maximiza a captura de mensagens fora do domínio sem gerar fallback excessivo nas mensagens válidas.
5. **Trocar o split único por validação cruzada estratificada**, para que as métricas reportadas tenham significado estatístico.
