#========== PRODUÇÃO DO RELATÓRIO:==============
# Para a entrega completa deste LAB01 você precisa copiar a saída do código (output) e adicionar as repostas das perguntas abaixo:

--- RESULTADOS DO LAB 01 ---
Mensagem: 'Quero consultar quanto dinheiro tenho' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Pode me ajudar a fazer um pix?' ==> Intenção Predita: [fazer_pix]
Mensagem: 'Gostaria de cancelar meu cartão de crédito' ==> Intenção Predita: [cancelar_conta]

# 1 - Avaliem os resultados e verifiquem se os resultados foram corretos ou incorretos. Coloque a resposta no arquivo do relatório do laboratório
A primeira resposta está incorreta, já que verificar quantidade de dinheiro, não é o mesmo que pedir para realizar um pix.
A segunda resposta está correta.
A terceira resposta está incorreta, o cliente solicitou cancelamento de cartão, não de conta, embora não tenha essa opção para a máquina.

# 2 - Detectado algum erro, qual seria a maneira mais correta de melhorar o resultado do algoritmo?
Aumentar o volume de dados e balancear os exemplos.

# 3 - Detalhe a função do LogisticRegression no algoritmo.
O LogisticRegression é o modelo de classificação responsável por aprender a relação entre as palavras presentes nas mensagens (representadas como vetores numéricos pelo CountVectorizer) e as intenções (rótulos/classes).


