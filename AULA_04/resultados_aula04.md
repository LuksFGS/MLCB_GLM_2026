# Atividade 3: Relatório Comparativo de Modelos

*Ao concluir a execução dos dois scripts, consolide as análises no arquivo resultados_aula04.md seguindo o modelo abaixo:*

# Relatório de Avaliação NLU - SAC Móveis Residenciais
1 - Tabela Comparativa de Métricas (Dados de Teste)

```
| Modelo        | Acurácia Geral | F1-Score (Weighted) | Principais Erros na Matriz |
| KNN (K=3)     |      100%      |        100%         | Não errou nenhuma frase de teste. |
| Decision Tree |      80%       |        79,7%        | Confundiu logistica_entregas com vendas e reclamacoes com suporte e trocas_devolucoes. |
```

2 - Análise dos Testes de Entrada ("input()")
```
* Comportamento do KNN (10 testes): [Como o KNN reagiu às variações das frases digitadas e ao fallback?]
* Comportamento da Decision Tree (8 testes): [Como a Árvore de Decisão se comportou em comparação ao KNN?]
```

3 - Veredito Final
```
* Melhor modelo para este projeto: [KNN ou Decision Tree]
* Justificativa técnica: [Explique a escolha com base nas métricas estatísticas e no comportamento do fallback]
```
