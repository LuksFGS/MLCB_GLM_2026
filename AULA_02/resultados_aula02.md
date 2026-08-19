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

