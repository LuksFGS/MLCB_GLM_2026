========== RESULTADOS DO LAB 01 ==============

    Mensagem: 'Quero consultar quanto dinheiro tenho' ==> Intenção Predita: [fazer_pix]

    Mensagem: 'Pode me ajudar a fazer um pix?' ==> Intenção Predita: [fazer_pix]

    Mensagem: 'Gostaria de cancelar meu cartão de crédito' ==> Intenção Predita: [cancelar_conta]

**1 - Avaliem os resultados e verifiquem se os resultados foram corretos ou incorretos.**

      A primeira resposta está **incorreta**, já que verificar quantidade de dinheiro, não é o mesmo que pedir para realizar um pix.
  
      A segunda resposta está **correta**.
 
      A terceira resposta está **incorreta**, o cliente solicitou cancelamento de cartão, não de conta, embora não tenha essa opção para o bot.

**2 - Detectado algum erro, qual seria a maneira mais correta de melhorar o resultado do algoritmo?**

      Aumentar o volume de dados e balancear os exemplos.

**3 - Detalhe a função do LogisticRegression no algoritmo.**

      O LogisticRegression é o modelo de classificação responsável por aprender a relação entre as palavras presentes nas mensagens (representadas como vetores numéricos pelo CountVectorizer) e as intenções (rótulos/classes).
    
========== RESULTADOS DO LAB 02 ==============
    
    Mensagem de Teste: 'Gostaria de devolver o produto que comprei'
    Intenção Predita: troca_devolucao

    --- Distribuição de Probabilidades por Classe ---

    Classe [duvida_frete]: 27.99%

    Classe [rastrear_pedido]: 24.54%

    Classe [troca_devolucao]: 47.46%



**1 - Avaliem os resultados e verifiquem se os resultados foram corretos ou incorretos. Coloque a resposta no arquivo do relatório do laboratório**
  
    A primeira resposta está **incorreta**, pois o cliente quer uma devolução, não o valor do frete.
  
    A segunda resposta está **incorreta**, o cliente não pediu pra rastrear o pedido.

    A terceira resposta está **correta**.

**2 - Detectado algum erro, qual seria a maneira mais correta de melhorar o resultado do algoritmo?**

    Aumentar a quantidade de exemplos para cada classe

 **3 - Detalhe a função do Naive Bayes no algorítmo.**

     O Naive Bayes é um algoritmo que classifica tentando adivinhar a intenção do usuário


========== RESULTADOS DO LAB 03 ==============
```
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dados_tech = {
    'mensagem': [
        'Esqueci minha senha de acesso', 'Não consigo entrar no sistema', 'Como redefinir minha senha?',
        'A internet esta muito lenta', 'Sem conexao de rede no escritorio', 'Minha conexao caindo toda hora',
        'Impressora nao esta funcionando', 'Nao consigo imprimir documentos', 'Impressora travada com papel'
    ],
    'intencao': [
        'reset_senha', 'reset_senha', 'reset_senha',
        'problema_conexao', 'problema_conexao', 'problema_conexao',
        'suporte_impressora', 'suporte_impressora', 'suporte_impressora'
    ]
}
df3 = pd.DataFrame(dados_tech)

# TODO 1: Separação de X e y
X = df3['mensagem']
y = df3['intencao']

# TODO 2: Divisão treino (70%) / teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# TODO 3: Vetorização com CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# TODO 4: Instanciar e treinar a Árvore de Decisão
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_train_vec, y_train)

# TODO 5: Predições e acurácia
predicoes = modelo_arvore.predict(X_test_vec)
acuracia = accuracy_score(y_test, predicoes)
print(f"Acurácia do Modelo: {acuracia * 100:.2f}%")
```
1- Por que a acurácia engana num dataset tão pequeno?

    Porque temos poucos exemplos. São apenas 9 frases, e somente 3 foram usadas para teste.

    Então, cada frase vale 33,33% da acurácia. Se acertasse mais uma, já iria para 66,67%.

    Por isso, esse resultado pode não mostrar realmente se o modelo é bom ou ruim. O ideal seria ter mais frases para testar o modelo.

2- Como a Árvore de Decisão separa as intenções?

    A Árvore de Decisão analisa as palavras das frases e faz algumas perguntas para decidir a intenção.

    Por exemplo:

    Tem a palavra "senha"?
    Tem a palavra "pedido"?
    Tem a palavra "internet"?

    Com essas respostas, ela vai seguindo um caminho na árvore até chegar na intenção que considera correta.

3- Qual o risco de não limitar max_depth?

    O principal risco é o overfitting.

    Isso acontece quando a árvore fica muito grande e começa a decorar os exemplos do treinamento.

    Assim, ela pode acertar muito bem as frases que já conhece, mas errar quando recebe frases novas.

    Por isso, usamos o max_depth para limitar o tamanho da árvore e tentar fazer com que ela aprenda de uma forma mais geral.       

========== RESULTADOS DO LAB 04 ==============


PROPOSTA
Você foi contratado para criar o protótipo do motor de Natural Language Understanding (NLU)
de um Chatbot para uma Agência de Viagens.

REQUISITOS OBRIGATÓRIOS
1. Criar um dataset próprio em um DataFrame Pandas com no mínimo 12 frases distribuídas em
   pelo menos 3 intenções distintas (ex: 'comprar_passagem', 'cancelar_reserva', 'falar_atendente').
2. Realizar a divisão em conjuntos de Treino e Teste usando `train_test_split`.
3. Escolher um vetorizador de texto (CountVectorizer ou TfidfVectorizer).
4. Escolher um algoritmo de classificação (LogisticRegression, MultinomialNB ou DecisionTreeClassifier).
5. Treinar o modelo e realizar predições com no mínimo 3 frases INÉDITAS (que não estavam no treino).
6. Exibir as intenções previstas no console.

ENTREGA
Construa o código em um notebook (.ipynb), execute, salve no repositório e registre os resultados
e a justificativa técnica no arquivo 'resultados_aula02.md'.

    Mensagem: 'Quero marcar uma passagem para Fortaleza' ==> Intenção Predita: [comprar_passagem]
    Mensagem: 'Preciso cancelar minha viagem de fim de semana' ==> Intenção Predita: [cancelar_reserva]
    Mensagem: 'Queria falar com uma pessoa do atendimento' ==> Intenção Predita: [cancelar_reserva]
    Justificativa técnica / minhas observações

    Das 3 frases inéditas, o modelo acertou 2:

    "Quero marcar uma passagem para Fortaleza" → acertou comprar_passagem (acho que porque "passagem" apareceu bastante nos exemplos de treino dessa classe).
    "Preciso cancelar minha viagem de fim de semana" → acertou cancelar_reserva (a palavra "cancelar" tem bastante peso no TF-IDF pra essa classe).
    "Queria falar com uma pessoa do atendimento" → errou, previu cancelar_reserva em vez de falar_atendente. Reparei que essa frase não usa nenhuma das palavras "atendente", "ajuda", "pessoa" ou "suporte" do jeito que apareceram no treino (ela usa "atendimento", que é uma palavra diferente pro modelo, já que o TF-IDF/Naive Bayes não entende que            "atendente" e "atendimento" têm a mesma raiz). Como o meu dataset da classe falar_atendente só tem 3 frases (bem menos que as outras), acho que isso também deixou essa classe mais fraca e mais fácil de errar.

    Isso me fez perceber, na prática, o mesmo problema que já tínhamos visto no LAB01 e no LAB03: com dataset pequeno e desbalanceado (a classe falar_atendente tem só 3 exemplos contra 5 e 4 das outras), o modelo fica muito dependente de repetir exatamente as mesmas palavras vistas no treino, e qualquer variação de vocabulário (como "atendimento" em       vez de "atendente") pode confundir a previsão. Se eu fosse melhorar, tentaria equilibrar o número de frases por intenção e incluir mais variações de palavras pra cada classe, principalmente pra falar_atendente.
