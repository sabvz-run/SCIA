import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Universos (as faixas de valores que vamos usar)
universo_idade = np.arange(0, 91, 1)
universo_classif = np.arange(0, 101, 1)  # Saída de 0 a 100

# Variáveis Fuzzy de Entrada e Saída
idade = ctrl.Antecedent(universo_idade, 'idade')
classificacao = ctrl.Consequent(universo_classif, 'classificacao')


# 2. Funções de Pertinência (Membership Functions)
# Define o que é "jovem", "adulto" e "idoso"
idade['jovem'] = fuzz.trimf(idade.universe, [0, 0, 35])
idade['adulto'] = fuzz.trimf(idade.universe, [25, 45, 65])
idade['idoso'] = fuzz.trimf(idade.universe, [55, 90, 90])

# Define as categorias de saída correspondentes
classificacao['jovem'] = fuzz.trimf(classificacao.universe, [0, 25, 50])
classificacao['adulto'] = fuzz.trimf(classificacao.universe, [25, 50, 75])
classificacao['idoso'] = fuzz.trimf(classificacao.universe, [50, 75, 100])


# 3. Regras
# Conecta a entrada com a saída
regra1 = ctrl.Rule(idade['jovem'], classificacao['jovem'])
regra2 = ctrl.Rule(idade['adulto'], classificacao['adulto'])
regra3 = ctrl.Rule(idade['idoso'], classificacao['idoso'])


# 4. Montagem e Execução do Sistema
# Junta as regras
sistema_fuzzy = ctrl.ControlSystem([regra1, regra2, regra3])

# Cria o simulador
simulador = ctrl.ControlSystemSimulation(sistema_fuzzy)

# ---- PONTO DE TESTE ----
idade_entrada = 30  # Mude este valor para testar
# -------------------------

# Executa o cálculo
simulador.input['idade'] = idade_entrada
simulador.compute()


# 5. Saída
resultado_numerico = simulador.output['classificacao']
print(f"Idade de Entrada: {idade_entrada}")
print(f"Resultado Numérico (Centroide): {resultado_numerico:.2f}")

# Para pegar a classificação final de forma mais clara
graus_pertinencia = {
    'jovem': fuzz.interp_membership(idade.universe, idade['jovem'].mf, idade_entrada),
    'adulto': fuzz.interp_membership(idade.universe, idade['adulto'].mf, idade_entrada),
    'idoso': fuzz.interp_membership(idade.universe, idade['idoso'].mf, idade_entrada)
}

classificacao_final = max(graus_pertinencia, key=graus_pertinencia.get)

print(f"Graus de pertinência: {graus_pertinencia}")
print(f"Classificação Final: {classificacao_final}")


# 6. Para gerar o gráfico (basta descomentar)
# import matplotlib.pyplot as plt
# idade.view()
# classificacao.view(sim=simulador)
# plt.show()
