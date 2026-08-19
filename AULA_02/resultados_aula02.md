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



 **3 - Detalhe a função do Naive Bayes no algorítmo.**

 Todos os resultados devem ser inseridos no arquivo resultados_aula02.md

#========== FIM ==============


========== RESULTADOS DO LAB 03 ==============

========== RESULTADOS DO LAB 04 ==============

