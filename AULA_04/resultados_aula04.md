# Atividade 3: Relatório Comparativo de Modelos

*Ao concluir a execução dos dois scripts, consolide as análises no arquivo resultados_aula04.md seguindo o modelo abaixo:*

## Relatório de Avaliação NLU - SAC Móveis Residenciais
1 - Tabela Comparativa de Métricas (Dados de Teste)

```
| Modelo        | Acurácia Geral | F1-Score (Weighted) | Principais Erros na Matriz |
| KNN (K=3)     |      100%      |        100%         | Não errou nenhuma frase de teste. |
| Decision Tree |      80%       |        79,7%        | Confundiu logistica_entregas com vendas e reclamacoes com suporte e trocas_devolucoes. |
```

2 - Análise dos Testes de Entrada ("input()")
```
* KNN: Teve melhores resultados, acertando frases relacionadas ao tema e apresentando diferentes níveis de confiança. Também acionou o fallback em uma frase fora do contexto.
* Decision Tree: Não acionou o fallback nenhuma vez e sempre apresentou *100% de confiança*, mesmo em classificações erradas ou frases sem relação com o dataset.
```

3 - Veredito Final
```
* Melhor modelo: KNN (K=3).
O KNN apresentou *100% de acurácia e F1-score, contra **80% da Decision Tree*, além de lidar melhor com incertezas por meio da variação da confiança e do fallback.
```


