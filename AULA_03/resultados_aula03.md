# Lab 01
```
--- RESULTADOS DO LAB 01 (AULA 03) ---
Mensagem: 'Preciso urgente da segunda via da fatura'
Intenção Predita: [segunda_via]
Vocabulário Filtrado (sem stopwords): ['2a', '2a via', 'aberto', 'acordo', 'acordo pagar', 'alterar', 'alterar endereço', 'app', 'atrasada', 'atualizo', 'atualizo dados', 'boleto', 'cadastramento', 'dados', 'dados residenciais', 'débito', 'débito aberto', 'dívida', 'emitir', 'emitir segunda', 'endereço', 'endereço cadastramento', 'fatura', 'fatura atrasada', 'fazer', 'fazer um', 'gostaria', 'gostaria alterar', 'negociar', 'negociar pagamento', 'no', 'no app', 'onde', 'onde atualizo', 'pagamento', 'pagamento dívida', 'pagar', 'pagar débito', 'posso', 'posso emitir', 'residenciais', 'residenciais no', 'segunda', 'segunda via', 'um', 'um acordo', 'via', 'via boleto', 'via fatura']
```

1 - Qual o impacto da remoção de stopwords no tamanho do vocabulário do modelo?

    Com a remoção das stopwords, a mensagem é filtrada, então a acurácia seria menor, pois palavras inúteis seriam incluídas.

2 - O que significa a configuração ngram_range=(1, 2) no TfidfVectorizer?

    Instrui o modelo a analisar palavras individuais (1) e pares de palavras consecutivas (2) ao mesmo tempo.

3 - Como a remoção de palavras genéricas ajuda a evitar classificações incorretas?

    O Modelo focará apenas nas palavras-chave, mantendo ele em uma resposta mais correta.
    
# Lab 02
```
--- RESULTADOS DO LAB 02 (AULA 03) ---

 --- Relatório de Classificação ---
                     precision    recall  f1-score   support

horario_atendimento       0.50      1.00      0.67         1
        localizacao       0.00      0.00      0.00         1
    troca_devolucao       0.00      0.00      0.00         1

           accuracy                           0.33         3
          macro avg       0.17      0.33      0.22         3
       weighted avg       0.17      0.33      0.22         3

    --- Matriz de Confusão ---
    [[1 0 0]
     [1 0 0]
     [0 1 0]]

```
1 - O que representam as métricas Precision, Recall e F1-Score no relatório?

    Precision é a quantidade que o Modelo acertou a resposta
    Recall é a quantidade de todas as mensagens reais da intenção correta, que o bot conseguiu encontrar.
    F1-Score é a média entre precision e recall.
    
2 - Como interpretar a diagonal principal da Matriz de Confusão?

    As colunas são as intenções que o Modelo previu e as linhas são intenções REAIS.

3 - Por que a acurácia isolada pode ser enganosa quando temos classes desbalanceadas?

    

# Lab 03
