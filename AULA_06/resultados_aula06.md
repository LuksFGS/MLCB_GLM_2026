# Resultados Atividade 2

## Output 1 
<img width="1698" height="804" alt="image" src="https://github.com/user-attachments/assets/af980515-1266-4185-a059-be7433eae5ea" />

## Output 2
<img width="1620" height="724" alt="image" src="https://github.com/user-attachments/assets/03058e4f-0c64-4dee-b347-52dbba86ffd6" />

## Output 3
<img width="1616" height="770" alt="image" src="https://github.com/user-attachments/assets/63875d1a-1267-4251-8d45-8563379c9cac" />

## Output 4
<img width="1642" height="758" alt="image" src="https://github.com/user-attachments/assets/8198ccda-c1d8-4c15-aad9-d4741809e677" />


# Resultados dos Labs 01, 02 e 03

## LAB 01 - Troca do Algoritmo de Classificação

O classificador de Regressão Logística foi substituído pelo
DecisionTreeClassifier.

O restante do pipeline de pré-processamento, geração dos embeddings
GloVe e interface Gradio foi mantido.

O modelo foi treinado novamente e a interface continuou realizando
a previsão das intenções cadastradas no dataset.

**Resultado:** Decision Tree implementada e integrada ao pipeline NLU.

---

## LAB 02 - Ajuste de Governança e Regra de Fallback

O limiar mínimo de confiança foi alterado de 50% para 65%.

Antes:
`LIMIAR_CONFIANCA = 0.50`

Depois:
`LIMIAR_CONFIANCA = 0.65`

A mensagem de status da interface também foi alterada para informar
explicitamente ao operador que o corte utilizado é de 65%.

Quando a confiança é igual ou superior a 65%, a intenção é aceita.

Quando a confiança é inferior a 65%, o sistema aciona o fallback.

**Resultado:** regra de governança e fallback atualizadas para o novo
limiar de 65%.

---

## LAB 03 - Expansão de Classe

Foi adicionada uma quinta intenção ao dataset:

`cancelar_contrato`

Foram adicionadas 5 frases de treinamento relacionadas ao cancelamento
ou rescisão de contratos.

Também foi adicionada uma resposta de negócio no dicionário
`RESPOSTAS_PADRAO`.

Após a inclusão da nova classe, o modelo foi treinado novamente.

**Teste realizado na interface do Gradio:**

Mensagem:
`Quero cancelar meu contrato de aluguel`

Intenção esperada:
`cancelar_contrato`

**Resultado:** a nova classe foi adicionada ao sistema NLU e ficou
disponível para classificação e resposta na interface.

**Resultado final:** sistema expandido de 4 para 5 classes de intenção.
