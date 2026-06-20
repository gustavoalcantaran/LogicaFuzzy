"""
Trabalho Prático - Lógica Fuzzy
Controlador Fuzzy para ajuste do tempo de verde de um semáforo (via secundária)

Variáveis:
  Entrada -> QueueSec     (0 a 30 veículos)  : Baixa, Média, Alta
  Entrada -> WaitTimeSec  (0 a 120 segundos) : Curto, Médio, Longo
  Saída   -> GreenAdj     (-10 a +30 s)      : Reduzir, Manter, Aumentar

Bibliotecas: numpy, scikit-fuzzy, matplotlib
"""

import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário para projeção 3D)

# Pasta onde os gráficos serão salvos: "output" ao lado deste script,
# criada automaticamente se não existir. Funciona em qualquer computador.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. DEFINIÇÃO DAS VARIÁVEIS (UNIVERSOS DE DISCURSO)
# ============================================================

queue_sec = ctrl.Antecedent(np.arange(0, 30 + 1, 1), 'QueueSec')
wait_time_sec = ctrl.Antecedent(np.arange(0, 120 + 1, 1), 'WaitTimeSec')
green_adj = ctrl.Consequent(np.arange(-10, 30 + 1, 1), 'GreenAdj')


# ============================================================
# 2. FUNÇÕES DE PERTINÊNCIA (FUZZIFICAÇÃO)
#    Trapezoidal nas pontas + triangular no meio
# ============================================================

# --- QueueSec: tamanho da fila na via secundária (0-30 veículos) ---
queue_sec['Baixa'] = fuzz.trapmf(queue_sec.universe, [0, 0, 5, 12])
queue_sec['Media'] = fuzz.trimf(queue_sec.universe, [8, 15, 22])
queue_sec['Alta'] = fuzz.trapmf(queue_sec.universe, [18, 25, 30, 30])

# --- WaitTimeSec: tempo médio de espera dos veículos (0-120 s) ---
wait_time_sec['Curto'] = fuzz.trapmf(wait_time_sec.universe, [0, 0, 20, 45])
wait_time_sec['Medio'] = fuzz.trimf(wait_time_sec.universe, [30, 60, 90])
wait_time_sec['Longo'] = fuzz.trapmf(wait_time_sec.universe, [75, 100, 120, 120])

# --- GreenAdj: ajuste aplicado ao tempo de verde (-10 a +30 s) ---
green_adj['Reduzir'] = fuzz.trapmf(green_adj.universe, [-10, -10, -5, 5])
green_adj['Manter'] = fuzz.trimf(green_adj.universe, [0, 10, 20])
green_adj['Aumentar'] = fuzz.trapmf(green_adj.universe, [15, 22, 30, 30])

# --- Gráficos das funções de pertinência ---
queue_sec.view()
plt.savefig(os.path.join(OUTPUT_DIR, '01_pertinencia_QueueSec.png'), dpi=150, bbox_inches='tight')

wait_time_sec.view()
plt.savefig(os.path.join(OUTPUT_DIR, '02_pertinencia_WaitTimeSec.png'), dpi=150, bbox_inches='tight')

green_adj.view()
plt.savefig(os.path.join(OUTPUT_DIR, '03_pertinencia_GreenAdj.png'), dpi=150, bbox_inches='tight')


# ============================================================
# 3. BASE DE REGRAS FUZZY (INFERÊNCIA)
#    9 regras cobrindo todas as combinações (3 x 3)
# ============================================================

rule1 = ctrl.Rule(queue_sec['Baixa'] & wait_time_sec['Curto'], green_adj['Reduzir'])
rule2 = ctrl.Rule(queue_sec['Baixa'] & wait_time_sec['Medio'], green_adj['Reduzir'])
rule3 = ctrl.Rule(queue_sec['Baixa'] & wait_time_sec['Longo'], green_adj['Manter'])

rule4 = ctrl.Rule(queue_sec['Media'] & wait_time_sec['Curto'], green_adj['Reduzir'])
rule5 = ctrl.Rule(queue_sec['Media'] & wait_time_sec['Medio'], green_adj['Manter'])
rule6 = ctrl.Rule(queue_sec['Media'] & wait_time_sec['Longo'], green_adj['Aumentar'])

rule7 = ctrl.Rule(queue_sec['Alta'] & wait_time_sec['Curto'], green_adj['Manter'])
rule8 = ctrl.Rule(queue_sec['Alta'] & wait_time_sec['Medio'], green_adj['Aumentar'])
rule9 = ctrl.Rule(queue_sec['Alta'] & wait_time_sec['Longo'], green_adj['Aumentar'])

regras = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]

semaforo_ctrl = ctrl.ControlSystem(regras)
semaforo_sim = ctrl.ControlSystemSimulation(semaforo_ctrl)


# ============================================================
# 4. TESTE COM UM EXEMPLO (FUZZIFICAÇÃO -> INFERÊNCIA -> DEFUZZIFICAÇÃO)
# ============================================================

# Exemplo: fila de 20 veículos, espera média de 70 segundos
exemplo_queue = 20
exemplo_wait = 70

semaforo_sim.input['QueueSec'] = exemplo_queue
semaforo_sim.input['WaitTimeSec'] = exemplo_wait
semaforo_sim.compute()

resultado = semaforo_sim.output['GreenAdj']
print("=" * 60)
print("EXEMPLO DE SIMULAÇÃO")
print("=" * 60)
print(f"QueueSec (fila)        = {exemplo_queue} veículos")
print(f"WaitTimeSec (espera)   = {exemplo_wait} s")
print(f"GreenAdj (ajuste verde) = {resultado:.2f} s")
print("=" * 60)

green_adj.view(sim=semaforo_sim)
plt.savefig(os.path.join(OUTPUT_DIR, '04_defuzzificacao_exemplo.png'), dpi=150, bbox_inches='tight')


# ============================================================
# 5. SUPERFÍCIE 3D (QueueSec x WaitTimeSec x GreenAdj)
# ============================================================

n_pontos = 25
queue_range = np.linspace(0, 30, n_pontos)
wait_range = np.linspace(0, 120, n_pontos)

X, Y = np.meshgrid(queue_range, wait_range)
Z = np.zeros_like(X)

for i in range(n_pontos):
    for j in range(n_pontos):
        sim = ctrl.ControlSystemSimulation(semaforo_ctrl)
        sim.input['QueueSec'] = X[i, j]
        sim.input['WaitTimeSec'] = Y[i, j]
        sim.compute()
        Z[i, j] = sim.output['GreenAdj']

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.2, antialiased=True)

ax.set_xlabel('QueueSec (veículos)')
ax.set_ylabel('WaitTimeSec (s)')
ax.set_zlabel('GreenAdj (s)')
ax.set_title('Superfície de Controle Fuzzy - Ajuste do Tempo de Verde')
fig.colorbar(surf, shrink=0.6, aspect=12, label='GreenAdj (s)')

plt.savefig(os.path.join(OUTPUT_DIR, '05_superficie_3D.png'), dpi=150, bbox_inches='tight')

print(f"\nGráficos salvos em {OUTPUT_DIR}")
print("Valor mínimo de GreenAdj na superfície:", round(Z.min(), 2))
print("Valor máximo de GreenAdj na superfície:", round(Z.max(), 2))