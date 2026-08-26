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

    Precision é a quantidade que o Modelo acertou a resposta.
    Recall é a quantidade de todas as mensagens reais da intenção correta, que o bot conseguiu encontrar.
    F1-Score é a média entre precision e recall.
    
2 - Como interpretar a diagonal principal da Matriz de Confusão?

    As colunas são as intenções que o Modelo previu e as linhas são intenções REAIS.

3 - Por que a acurácia isolada pode ser enganosa quando temos classes desbalanceadas?

    A acurácia é só a proporção de acertos totais, sem separar por classe. Isso é um problema quando as classes estão desbalanceadas (uma classe tem muito mais exemplos que as outras), porque o modelo pode "trapacear" simplesmente chutando sempre a classe majoritária e ainda assim parecer bom. Por isso usamos métricas como precision, recall e f1-score.

# Lab 03

1 - Cole o código corrigido e a acurácia obtida.

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    dados_rh = {
    'mensagem': [
        'Como solicitar minhas ferias?', 'Quero agendar meu periodo de ferias',
        'Onde baixo meu holerite do mes?', 'Preciso do comprovante de rendimentos',
        'Como cadastrar meu atestado medico?', 'Onde envio o atestado de consulta?'
    ],
    'intencao': [
        'solicitar_ferias', 'solicitar_ferias',
        'obter_holerite', 'obter_holerite',
        'enviar_atestado', 'enviar_atestado'
    ]
    }
    df3 = pd.DataFrame(dados_rh)

    #TODO 1: Separando X e y
    X = df3['mensagem']
    y = df3['intencao']

    #TODO 2: Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    
    #TODO 3: Montando o Pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words=['de', 'o', 'meu', 'minhas'])),
        ('classifier', LogisticRegression())
    ])

    #TODO 4: Treinando o pipeline com os dados BRUTOS (texto puro)
    pipeline.fit(X_train, y_train)
    
    #TODO 5: Predição e acurácia
    predicoes = pipeline.predict(X_test)
    print(f"--- RESULTADOS DO LAB 03 (AULA 03) ---")
    print(f"Acuracia via Pipeline: {accuracy_score(y_test, predicoes) * 100:.2f}%")

    --- RESULTADOS DO LAB 03 (AULA 03) ---
    Acuracia via Pipeline: 0.00%

2 - Qual é a grande vantagem de utilizar o objeto Pipeline no Scikit-Learn?

    Ele junta várias etapas (vetorizador + classificador) em um único objeto, então eu só preciso chamar .fit() e .predict() uma vez, sem executar cada etapa na mão.

3 - Por que o Pipeline evita que erros de pré-processamento ocorram entre treino e teste?

    Por que evita erros entre treino e teste, o Pipeline garante que o vetorizador só "aprende" o vocabulário (fit) com os dados de treino, e nos dados de teste ele só aplica (transform) esse vocabulário já aprendido, sem re-treinar. Isso evita o vazamento de dados (data leakage).
