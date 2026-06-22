"""Módulo de Lógica Fuzzy para Controle de Tempo de Semáforo.

Este script implementa um sistema de controle fuzzy que ajusta o tempo
de luz verde de uma via secundária com base no tamanho da fila e no
tempo de espera dos veículos, conforme especificações do trabalho prático.
"""

import os
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def criar_sistema_fuzzy():
    """Cria e retorna o sistema de controle fuzzy para o semáforo."""

    # =========================================================================
    # 1. DEFINIÇÃO DAS VARIÁVEIS DO SISTEMA
    # =========================================================================
    
    queue = ctrl.Antecedent(np.arange(0, 31, 1), 'QueueSec')
    wait_time = ctrl.Antecedent(np.arange(0, 121, 1), 'WaitTimeSec')
    green_adj = ctrl.Consequent(np.arange(-10, 31, 1), 'GreenAdj')

    # =========================================================================
    # 2. FUNÇÕES DE PERTINÊNCIA (MEMBERSHIP FUNCTIONS)
    # =========================================================================
    
    # QueueSec: Baixa, Média, Alta
    queue['Baixa'] = fuzz.trapmf(queue.universe, [0, 0, 5, 12])
    queue['Média'] = fuzz.trimf(queue.universe, [8, 15, 22])
    queue['Alta'] = fuzz.trapmf(queue.universe, [18, 25, 30, 30])
    
    # WaitTimeSec: Curto, Médio, Longo
    wait_time['Curto'] = fuzz.trapmf(wait_time.universe, [0, 0, 30, 60])
    wait_time['Médio'] = fuzz.trimf(wait_time.universe, [40, 60, 80])
    wait_time['Longo'] = fuzz.trapmf(wait_time.universe, [60, 90, 120, 120])
    
    # GreenAdj: Reduzir, Manter, Aumentar
    green_adj['Reduzir'] = fuzz.trapmf(green_adj.universe, [-10, -10, -5, 0])
    green_adj['Manter'] = fuzz.trimf(green_adj.universe, [-5, 0, 10])
    green_adj['Aumentar'] = fuzz.trapmf(green_adj.universe, [5, 15, 30, 30])

    # =========================================================================
    # 3. BASE DE REGRAS FUZZY 
    # =========================================================================
    
    rule1 = ctrl.Rule(queue['Baixa'] & wait_time['Curto'], green_adj['Reduzir'])
    rule2 = ctrl.Rule(queue['Alta'], green_adj['Aumentar'])
    rule3 = ctrl.Rule(queue['Média'] & wait_time['Longo'], green_adj['Aumentar'])
    rule4 = ctrl.Rule(queue['Baixa'] & wait_time['Longo'], green_adj['Manter'])
    rule5 = ctrl.Rule(queue['Média'] & wait_time['Médio'], green_adj['Manter'])
    rule6 = ctrl.Rule(queue['Média'] & wait_time['Curto'], green_adj['Reduzir'])
    rule7 = ctrl.Rule(queue['Baixa'] & wait_time['Médio'], green_adj['Manter'])

    # Compilando o sistema
    sistema_controle = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
    sistema_simulacao = ctrl.ControlSystemSimulation(sistema_controle)
    
    return sistema_simulacao, queue, wait_time, green_adj

def salvar_graficos_pertinencia(queue, wait_time, green_adj):
    """Salva os gráficos das funções de pertinência na pasta 'output'."""
    print("\n[INFO] A gerar e salvar os gráficos de pertinência na pasta 'output'...")
    
    # Criar pasta output se não existir
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Salvar QueueSec
    queue.view()
    plt.savefig('output/1_QueueSec.png', bbox_inches='tight', dpi=300)
    plt.close() # Fecha a janela silenciosamente para o código continuar rodando
    
    # Salvar WaitTimeSec
    wait_time.view()
    plt.savefig('output/2_WaitTimeSec.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Salvar GreenAdj
    green_adj.view()
    plt.savefig('output/3_GreenAdj.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("[INFO] Gráficos de pertinência salvos com sucesso!")

def gerar_superficie_3d(sistema_simulacao):
    """Gera, salva na pasta output e exibe a superfície de controle 3D."""
    
    print("\n[INFO] Gerando Superfície 3D...")
    
    upsampled_queue = np.linspace(0, 30, 21)
    upsampled_wait = np.linspace(0, 120, 21)
    x, y = np.meshgrid(upsampled_queue, upsampled_wait)
    z = np.zeros_like(x)

    for i in range(21):
        for j in range(21):
            sistema_simulacao.input['QueueSec'] = x[i, j]
            sistema_simulacao.input['WaitTimeSec'] = y[i, j]
            sistema_simulacao.compute()
            z[i, j] = sistema_simulacao.output['GreenAdj']

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', 
                           linewidth=0.4, antialiased=True)

    ax.set_xlabel('QueueSec (Veículos)')
    ax.set_ylabel('WaitTimeSec (Segundos)')
    ax.set_zlabel('GreenAdj (Segundos de Ajuste)')
    ax.set_title('Superfície de Controle Fuzzy - Semáforo')
    
    fig.colorbar(surf, shrink=0.5, aspect=0.5)
    plt.tight_layout()
    
    # Criar pasta output se não existir
    if not os.path.exists('output'):
        os.makedirs('output')
        
    # Salvar a imagem antes de mostrar na tela
    plt.savefig('output/4_Superficie3D.png', bbox_inches='tight', dpi=300)
    print("[INFO] Superfície 3D salva em 'output/4_Superficie3D.png'!")


def simular_casos_teste(sistema_simulacao):
    """Testa o sistema com alguns cenários do mundo real."""
    print("\n" + "="*50)
    print(" TESTES PRÁTICOS DO SISTEMA ".center(50))
    print("="*50)
    
    cenarios = [
        {"fila": 3, "espera": 15, "desc": "Madrugada (Poucos carros, recém chegados)"},
        {"fila": 15, "espera": 60, "desc": "Trânsito Moderado (Normal)"},
        {"fila": 28, "espera": 110, "desc": "Hora do Rush (Fila gigante, espera longa)"},
        {"fila": 5, "espera": 115, "desc": "Esquecidos (Poucos carros, mas muito tempo parados)"}
    ]
    
    for c in cenarios:
        sistema_simulacao.input['QueueSec'] = c['fila']
        sistema_simulacao.input['WaitTimeSec'] = c['espera']
        sistema_simulacao.compute()
        
        ajuste = sistema_simulacao.output['GreenAdj']
        print(f"\nCenário: {c['desc']}")
        print(f" -> Entradas: Fila = {c['fila']} | Espera = {c['espera']}s")
        print(f" -> Decisão Fuzzy: Ajustar tempo de verde em {ajuste:+.2f} segundos")

def main():
    """Função principal."""
    # 1. Instanciar o sistema
    sistema, var_queue, var_wait, var_adj = criar_sistema_fuzzy()
    
    # 2. Rodar simulações no terminal
    simular_casos_teste(sistema)
    
    # 3. Gerar e SALVAR Gráficos de Pertinência
    salvar_graficos_pertinencia(var_queue, var_wait, var_adj)
    
    # 4. Gerar, SALVAR e Mostrar o gráfico 3D
    gerar_superficie_3d(sistema)

if __name__ == '__main__':
    main()