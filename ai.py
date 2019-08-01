import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Estrutura da arquitetura da rede neural
class Network(nn.Module):
    def __init__(self, input_size, nb_action): # (quantidade de entradas, número de ações)
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        hidden_size = 30
        
        # Quantidade de neurônios na camada de entrada: 5 (input_size)
        # Quantidade de neurônios na camada oculta: 30 (hidden_size)
        # Quantidade de neurônios na camada de saída: 3 (nb_action)
        self.fc1 = nn.Linear(input_size, hidden_size) # Ligação da camada de entrada até a camada oculta
        self.fc2 = nn.Linear(hidden_size, nb_action)
        
    def forward(self, state): # O estado representa as 5 entradas da rede neural
        # Funcão de ativação -> Relu
        x = F.relu(self.fc1(state)) # Aplicação da função de ativação (camada de entrada -> camada oculta)
        q_values = self.fc2(x) # Resultado da aplicação da função de ativação
        return q_values # Retorna os valores finais da rede neural

# Replay de experiência
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity # Capacidade da memória
        self.memory = [] # buffer com os últimos eventos (1000)
        
    def push(self, event):
        self.memory.append(event) # Adiciona cada evento ao buffer
        if len(self.memory) > self.capacity: # Se o buffer estiver cheio, remove-se o primeiro elemento
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) # Escolhe aleatóriamente alguns eventos e transforma en formato para rede neural
        result = map(lambda x: Variable(torch.cat(x, 0)), samples) # Mapeia os dados através da concatenação para não se perderem
        return result
    
# Deep Q-Learning
class Dqn(object):
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma # Fator de desconto
        self.reword_window = []
        self.model = Network(input_size, nb_action) # Modelo da rede neural
        self.memory = ReplayMemory(100000) # Grava os últimos 100.000 eventos 
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) # Otimizador para os pesos da rede neural | "lr" representa a taxa de aprendizagem
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # Adcionando o valor de "Batch" no registro para o Pytorch
        self.last_action = 0
        self.last_reword = 0
        
    def selectAction(self, state):
        temperature = 50 # Quanto maior a temperatura, maior a chance de selecionar a saída com mais probabilidade
        probs = F.softmax(self.model(Variable(state, volatile = True)) * temperature) # Probabilidades para cada saída através da função Softmax
        action = probs.multinomial(1) # Escolhe alguma das ações de acordo com as probabilidades
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) # Valores previstos
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward # Cálculo dos estados target para cada saída da rede
        td_loss = F.smooth_l1_loss(outputs, target) # Cálculo do erro (quanto menor a diferença melhor)
        # Atualização dos pesos através da decida do gradiente
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step() # Atualiza os pesos
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]),
                     torch.Tensor([self.last_reword])))
        action = self.selectAction(new_state)
        if len(self.memory.memory) > 100: # Se lista estiver completa, inicia-se a aprendizagem
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reword = reward
        self.reword_window.append(reward)
        if len(self.reword_window) > 1000:
            del self.reword_window[0]
        return action
    
    def score(self):
        return sum(self.reword_window) / (len(self.reword_window) + 1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimazer': self.optimizer.state_dict()
                    }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimazer'])
            print("Rede carregada com sucesso!")
        else:
            print("Erro ao carregar a rede!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    