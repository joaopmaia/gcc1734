# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from math import inf

from game import Agent

class ReflexAgent(Agent):
	"""
	A reflex agent chooses an action at each choice point by examining
	its alternatives via a state evaluation function.

	The code below is provided as a guide.  You are welcome to change
	it in any way you see fit, so long as you don't touch our method
	headers.
	"""
	#

	def getAction(self, gameState): 
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)  # Gera o novo estado após a ação
		newPos = successorGameState.getPacmanPosition()  # Posição de Pac-Man após a ação
		oldFood = currentGameState.getFood()  # Informações sobre a comida no estado atual
		newFood = successorGameState.getFood()  # Informações sobre a comida no estado sucessor
		newFoodList = newFood.asList()  # Lista de posições de comida no estado sucessor
		ghostPositions = successorGameState.getGhostPositions()  # Posições dos fantasmas
		newGhostStates = successorGameState.getGhostStates()  # Estados dos fantasmas (incluindo o tempo de medo)
		
		minDistanceGhost = float("+inf")

		for ghostPos in ghostPositions:
			minDistanceGhost = min(minDistanceGhost, util.manhattanDistance(newPos, ghostPos))
		
		if minDistanceGhost == 0: 
			return float("-inf")
		
		if successorGameState.isWin():
			return float("+inf") 

		score = successorGameState.getScore()
		score += 2 * minDistanceGhost
		minDistanceFood = float("+inf")

		for foodPos in newFoodList:
			minDistanceFood = min(minDistanceFood, util.manhattanDistance(foodPos, newPos))

		score -= 2 * minDistanceFood
		
		if(successorGameState.getNumFood() < currentGameState.getNumFood()):
			score += 5
		
		if action == Directions.STOP:
			score -= 10 

		return score 



def scoreEvaluationFunction(currentGameState):
	"""
	This default evaluation function just returns the score of the state.
	The score is the same one displayed in the Pacman GUI.

	This evaluation function is meant for use with adversarial search agents
	(not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	This class provides some common elements to all of your
	multi-agent searchers.  Any methods defined here will be available
	to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	You *do not* need to make any changes here, but you can if you want to
	add functionality to all your adversarial search agents.  Please do not
	remove anything, however.

	Note: this is an abstract class: one that should not be instantiated.  It's
	only partially specified, and designed to be extended.  Agent (game.py)
	is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		chama o minimax p saber quais ações tomar

		  Returns the minimax action from the current gameState using self.depth
		  and self.evaluationFunction.

		  Here are some method calls that might be useful when implementing minimax.
		  gameState.getLegalActions(agentIndex):
			Returns a list of legal actions for an agent
			agentIndex=0 means Pacman, ghosts are >= 1

		  Directions.STOP:
			The stop direction, which is always legal

		  gameState.generateSuccessor(agentIndex, action):
			Returns the successor game state after an agent takes an action

		  gameState.getNumAgents():
			Returns the total number of agents in the game
		"""
		minimax = self.minimax(gameState, agentIndex=0, depth=self.depth) 
		
		return minimax['action']

	
	def minimax(self, gameState, agentIndex=0, depth='2', action=Directions.STOP):

		agentIndex = agentIndex % gameState.getNumAgents() 
		if agentIndex == 0: depth = depth-1 
		if gameState.isWin() or gameState.isLose() or depth == -1: 
			return {'value':self.evaluationFunction(gameState), 'action':action} 
		else:
			if agentIndex==0: return self.maxValue(gameState,agentIndex,depth) 
			else: return self.minValue(gameState,agentIndex,depth)

	def maxValue(self, gameState, agentIndex, depth):
		v = {'value':float('-inf'), 'action':Directions.STOP}
		legalMoves = gameState.getLegalActions(agentIndex)        

		for action in legalMoves:
			if action == Directions.STOP: continue
			successorGameState = gameState.generateSuccessor(agentIndex, action) 
			successorMinMax = self.minimax(successorGameState, agentIndex+1, depth, action) 
			
			if v['value'] <= successorMinMax['value']:
				v['value'] = successorMinMax['value']
				v['action'] = action
		return v 


	def minValue(self, gameState, agentIndex, depth):

		v = {'value': float('inf'), 'action': Directions.STOP}
		legalMoves = gameState.getLegalActions(agentIndex)

		for action in legalMoves:
			if action == Directions.STOP: continue
			successorGameState = gameState.generateSuccessor(agentIndex, action)
			successorMinMax = self.minimax(successorGameState, agentIndex+1, depth, action)

			if v['value'] >= successorMinMax['value']: 
				v['value'] = successorMinMax['value']
				v['action'] = action
		return v

class AlphaBetaAgent(MultiAgentSearchAgent):

	def maxValue(self, gameState, agentIndex, depth, alpha, beta):
		bestVal = -inf  # Inicializa a melhor pontuação com um valor muito baixo
		legalMoves = gameState.getLegalActions(agentIndex)  # Obtém as ações legais para Pac-Man

		for action in legalMoves:
			successorGameState = gameState.generateSuccessor(agentIndex, action)  # Gera o estado sucessor
			valOfAction, _ = self.minimax(successorGameState, agentIndex + 1, depth, alpha, beta)  # Chama minimax para o próximo agente
			
			if max(bestVal, valOfAction) == valOfAction:  # Se a nova pontuação for maior, atualiza
				bestVal, bestAction = valOfAction, action

			if bestVal > beta:  # Poda: se a melhor pontuação for maior que beta, não é necessário continuar
				return bestVal, bestAction

			alpha = max(alpha, bestVal)  # Atualiza o valor de alpha, pois Pac-Man quer maximizar
		return bestVal, bestAction

	def minValue(self, gameState, agentIndex, depth, alpha, beta):
		bestVal = inf  # Inicializa a melhor pontuação com um valor muito alto
		legalMoves = gameState.getLegalActions(agentIndex)  # Obtém as ações legais para os fantasmas

		for action in legalMoves:
			successorGameState = gameState.generateSuccessor(agentIndex, action)  # Gera o estado sucessor
			valOfAction, _ = self.minimax(successorGameState, agentIndex + 1, depth, alpha, beta)  # Chama minimax para o próximo agente
			
			if min(bestVal, valOfAction) == valOfAction:  # Se a nova pontuação for menor, atualiza
				bestVal, bestAction = valOfAction, action

			if bestVal < alpha:  # Poda: se a melhor pontuação for menor que alpha, não é necessário continuar
				return bestVal, bestAction
			
			beta = min(beta, bestVal)  # Atualiza o valor de beta, pois os fantasmas querem minimizar
		return bestVal, bestAction

	def minimax(self, gameState, agentIndex, depth, alpha, beta):
		agentIndex = agentIndex % gameState.getNumAgents()

		if agentIndex==0: depth = depth-1

		if gameState.isWin() or gameState.isLose() or depth == -1:
			return self.evaluationFunction(gameState), None

		if agentIndex == 0:
			bestVal, bestAction = self.maxValue(gameState, agentIndex, depth, alpha, beta) 
		else:
			bestVal, bestAction = self.minValue(gameState, agentIndex, depth, alpha, beta)

		return bestVal, bestAction

		"""
		Your minimax agent with alpha-beta pruning (question 3)
		"""

	def getAction(self, gameState):
		"""
		Returns the alpha-beta pruned minimax action using self.depth and self.evaluationFunction
		"""
		_, action = self.minimax(gameState, self.index, self.depth, -inf, inf) 
		return action


class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		Returns the expectimax action using self.depth and self.evaluationFunction

		All ghosts should be modeled as choosing uniformly at random from their
		legal moves.
		"""
		"*** YOUR CODE HERE ***"
		expectimax = self.expectimax(gameState, agentIndex=0, depth=self.depth)
		return expectimax['action']
	
	def expectimax(self, gameState, agentIndex=0, depth='2', action=Directions.STOP):
		agentIndex = agentIndex % gameState.getNumAgents()
		if agentIndex == 0: depth = depth-1
		if gameState.isWin() or gameState.isLose() or depth == -1:
			return {'value':self.evaluationFunction(gameState), 'action':action}
		else:
			if agentIndex==0: return self.maxValue(gameState,agentIndex,depth)
			else: return self.expValue(gameState,agentIndex,depth)
	
	def maxValue(self, gameState, agentIndex, depth):
		v = {'value': float('-inf'), 'action': Directions.STOP}
		legalMoves = gameState.getLegalActions(agentIndex)        
		for action in legalMoves:
			if action == Directions.STOP: continue
			successorGameState = gameState.generateSuccessor(agentIndex, action) 
			successorExpectiMax = self.expectimax(successorGameState, agentIndex+1, depth, action)
			if v['value'] <= successorExpectiMax['value']:
				v['value'] = successorExpectiMax['value']
				v['action'] = action
		return v

	def expValue(self, gameState, agentIndex, depth):
		v = {'value': 0, 'action': Directions.STOP}
		legalMoves = gameState.getLegalActions(agentIndex)        
		for action in legalMoves:
			if action == Directions.STOP: continue
			successorGameState = gameState.generateSuccessor(agentIndex, action)
			'''*** ESCREVER AQUI *** '''
			successorExpectiMax = self.expectimax(successorGameState, agentIndex+1, depth, action)
			p = 1/len(legalMoves)
			v['value'] += p * successorExpectiMax['value']
			v['action'] = action
		return v

def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"
	
	if currentGameState.isWin():
		return float("+inf")

	
	if currentGameState.isLose():
		return float("-inf")

	
	score = scoreEvaluationFunction(currentGameState)
	newFoodList = currentGameState.getFood().asList()
	newPos = currentGameState.getPacmanPosition()
	
	#
	# ATENÇÃO: variáveis não usadas AINDA! 
	# Procure modificar essa função para usar essas variáveis e melhorar a função de avaliação.
	# Descreva em seu relatório de que forma essas variáveis foram usadas.
	#
	
	ghostStates = currentGameState.getGhostStates() # Estado dos fantasmas
	scaredGhostDistances = [] # Lista de dist dos fantasmas vulneraveis
	for ghost in ghostStates:
		if ghost.scaredTimer > 0: # Caso o fantasma esteja vulneravel
			scaredGhostDistances += [util.manhattanDistance(newPos, ghost.getPosition())] # Add na lista
	minDistanceScaredGhost = -1
	if len(scaredGhostDistances) > 0: # Se |lista| > 0, existem vulneraveis
		minDistanceScaredGhost = min(scaredGhostDistances) # Fantasma vulneravel mais prox



	# Calc dist entre agente e pilula mais prox
	minDistanceFood = float("+inf")
	for foodPos in newFoodList:
		minDistanceFood = min(minDistanceFood, util.manhattanDistance(foodPos, newPos))

	# Incentiva o agente a ir pra pilula mais prox
	score -= 2 * minDistanceFood

	# Incentiva o agente a comer pilulas
	score -= 4 * len(newFoodList)
	
	score += 2 / minDistanceScaredGhost # Incentiva o agente a ir pros fantasmas vulneraveis

	# Incentiva a agente a ficar prox das pilulas
	capsulelocations = currentGameState.getCapsules()
	score -= 4 * len(capsulelocations)

	return score

# Abbreviation
better = betterEvaluationFunction
