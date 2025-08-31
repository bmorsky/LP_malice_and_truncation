import numpy as np; import pandas as pd
from typing import Literal, Tuple
from scipy.optimize import linprog
import cdd
import random as rd
from itertools import permutations
from matplotlib import pyplot as plt

class LinearProgrammingError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
    def __str__(self) -> str:
        return super().__str__()

class Game(pd.DataFrame):
    def __init__(self, payoff_df:pd.DataFrame, deep = True) -> None:
        super().__init__()
        for column in payoff_df.columns:
            this_column:pd.Series = payoff_df[column]
            self[column] = this_column.copy(deep = deep)
            
    def copy(self, deep = True):
        return self.__init__(payoff_df = self, deep = deep)    
    
    def outcome(self, strategies:tuple) -> Tuple[float, float]:
        if len(strategies) < 2:
            raise ValueError("Need at least 2 strategies to fix an outcome. What is given has a length of {}".format(
                len(strategies)))
        s1, s2 = strategies[:2]
        if s1 not in self.index:
            raise ValueError("Can't find strategy '{}' for player 1, available strategies are {}".format(s1, self.index))
        if s2 not in self.columns:
            raise ValueError("Can't find strategy '{}' for player 2, available strategies are {}".format(s2, self.columns))
        return self.loc[s1, s2]
    
    def payoffMatrix(self, player:Literal["row", "column"] = "column") -> np.matrix:
        player = player.lower()
        if player not in ["row", "column"]:
            raise ValueError("Player need to be one of ['row', 'column'], what is given is {}".format(player))
        player_index = {"row":0, "column":1}[player]
        result = []
        for s1 in self.index:
            this_row = []
            for s2 in self.columns: this_row += [self.loc[s1, s2][player_index]]
            result += [this_row]
        return np.matrix(result)
    
    def strategies(self, player:Literal["row", "column"] = "column") -> pd.Index:
        player = player.lower()
        if player not in ["row", "column"]:
            raise ValueError("Player need to be one of ['row', 'column'], what is given is {}".format(player))
        player_index = {"row":0, "column":1}[player]
        return [self.index, self.columns][player_index]
    
    def fromMatrices(player_1_payoff_matrix:np.matrix,
                    player_2_payoff_matrix:np.matrix, 
                    transposed = False,
                    player_1_strategies:list[str] = None, 
                    player_2_strategies:list[str] = None):
        if not transposed: player_2_payoff_matrix = player_2_payoff_matrix.T
        if not player_1_payoff_matrix.shape == player_2_payoff_matrix.shape: raise ValueError
        if player_1_strategies is None: player_1_strategies = [f"R_{i}" for i in range(player_1_payoff_matrix.shape[0])]
        if player_2_strategies is None: player_2_strategies = [f"C_{i}" for i in range(player_1_payoff_matrix.shape[1])]
        if not len(player_1_strategies) == player_1_payoff_matrix.shape[0]: raise ValueError
        if not len(player_2_strategies) == player_2_payoff_matrix.shape[1]: raise ValueError
        payoff_dict = {}
        for j, strategy_2 in enumerate(player_2_strategies):
            payoff_dict[strategy_2] = [(player_1_payoff_matrix[i,j], player_2_payoff_matrix[i,j])
                                        for i, _ in enumerate(player_1_strategies)]
        return Game(payoff_df = pd.DataFrame(payoff_dict, index = player_1_strategies))
    

def maximin(payoffMatrix:np.matrix, 
            player:Literal["row", "column"] = "column") -> Tuple[np.float64, np.matrix]:
    '''
    Returns: maximin value: np.float64, and a particular maximin strategy: np.matrix \n
    Note that this method gives a particular maximin strategy, not the entire maximin strategy space.
    '''
    worst:float = payoffMatrix.min()
    shift_value = abs(worst) + 1
    A:np.matrix = payoffMatrix + shift_value
    if player.lower() == "row": A = A.T
    b = np.ones(A.shape[0])
    c = np.ones(A.shape[1])
    bounds = [(0, None)] * A.shape[1]
    result = linprog(c = c, A_ub = -A, b_ub = -b, bounds = bounds)
    if not result.success: print("Note that the linear programming was not successful")
    d = result.fun
    maximin_value = d**-1 - shift_value
    maximin_strategy = np.matrix(result.x).T / d
    return maximin_value, maximin_strategy

def thresholdVertices(payoffMatrix:np.matrix, threshold:np.float64, 
                      player:Literal["row", "column"] = "column") -> pd.DataFrame:
    if player == "row": payoffMatrix = payoffMatrix.T
    if maximin(payoffMatrix = payoffMatrix)[0] < threshold: raise ValueError("Given threshold is not feasible")
    n, m = payoffMatrix.shape
    # Coefficients of the probability hyperplane
    lin_set = {0}
    normalization_A = np.ones((1, m))
    normalization_b = np.array([1])
    nonnegativity_A = np.eye(N = m, M = m)
    nonnegativity_b = np.zeros((m, 1))
    probability_A = np.vstack([normalization_A, nonnegativity_A])
    probability_b = np.vstack([normalization_b, nonnegativity_b])
    # Coefficients of the threshold constraints
    payoff_A = payoffMatrix
    threshold_b = np.ones((n, 1)) * threshold
    # Feed the constraints to the double description module
    A = np.vstack([probability_A, payoff_A])
    b = np.vstack([probability_b, threshold_b])
    b_A = np.array(np.hstack([-b, A])) # 0 \leq -b + Ax
    M_Hrep = cdd.matrix_from_array(b_A, rep_type = cdd.RepType.INEQUALITY, lin_set = lin_set)
    P_Hrep = cdd.polyhedron_from_matrix(M_Hrep)
    M_Vrep = cdd.copy_generators(P_Hrep)
    # Output as a pandas DataFrame
    probabilitySubset = pd.DataFrame(M_Vrep.array)
    if (probabilitySubset.iloc[:, 0] != 1).any(): raise ValueError("Unbounded polyhedron")
    probabilitySubset = probabilitySubset.iloc[:, 1:]
    probabilitySubset.index = [f"Vertex_{i}" for i in range(probabilitySubset.shape[0])]
    probabilitySubset.columns = [f"Probability_{i}" for i in range(probabilitySubset.shape[1])]
    probabilitySubset = probabilitySubset.T
    return probabilitySubset

def possibleSupports(game:Game, player:Literal["row", "column"] = "column") -> np.matrix:
    rows = []
    S = game.strategies(player = player)
    dim_S = S.shape[0]
    for i in range(1, 2**dim_S):
        this_support_as_str = bin(i)[2:]
        while len(this_support_as_str) < dim_S: this_support_as_str = '0' + this_support_as_str
        this_support_as_list = [int(j) for j in this_support_as_str]
        rows.append(this_support_as_list)
    result:np.matrix = np.matrix(rows)
    result_as_list = []
    for row_i in result:
        row_i:np.matrix
        result_as_list.append(row_i.T)
    return result_as_list

def extractSupport(mixed_strategy_vector:np.matrix, epsilon = 1e-9) -> np.matrix:
    if not all(np.isreal(mixed_strategy_vector)):
        raise ValueError("Contains complex entries: \n{}".format(mixed_strategy_vector))
    if any(mixed_strategy_vector < -epsilon):
        raise ValueError("Contains significantly negative entries: \n{}".format(mixed_strategy_vector))
    significant:np.matrix = mixed_strategy_vector > epsilon
    return significant.astype(int)

def nonzeroAt(support:np.matrix) -> list:
    return [i for i, row in enumerate(support) if row != [0]]

def sub(payoff_matrix:np.matrix, support:np.matrix, 
        player:Literal["row", "column"] = "column") -> np.matrix:
    A, s = payoff_matrix, support
    player = player.lower()
    if player == "row": A = A.T
    if not A.shape[1] == s.shape[0]: raise ValueError("The dimensions are A:{}, s:{}".format(A.shape, s.shape))
    if not s.shape[1] == 1: raise ValueError("The s should be a column vector: dim(s):{}".format(s.shape))
    active_columns = nonzeroAt(support = support)
    if player == "row": return A[:, active_columns].T
    return A[:, active_columns]

def maxIndifferent(opponents_payoff_matrix:np.matrix, 
                    opponents_support:np.matrix, 
                    opponents_type:Literal["row", "column"],
                    players_support:np.matrix,
                    ) -> np.matrix:
    A = opponents_payoff_matrix
    opponents_type = opponents_type.lower()
    if opponents_type == "column": A = A.T
    s_opponent = opponents_support
    A_reduced = sub(payoff_matrix = A, 
                    support = opponents_support, 
                    player = "row") # Fix row because we have already transposed
    c = s_opponent.T * A
    A_ub, b_ub = A, np.ones(shape = (A.shape[0], 1))
    A_eq, b_eq = A_reduced, np.ones(shape = (A_reduced.shape[0], 1))
    bounds = [(0, None) if p_i > 0 else (0, 0) for p_i in players_support.T.tolist()[0]]
    result = linprog(c = c, 
                     A_ub = A_ub, b_ub = b_ub, 
                     A_eq = A_eq, b_eq = b_eq, 
                     bounds = bounds)
    if not result.success: raise LinearProgrammingError("The linear programming was not successful...")
    mixed_strategy = np.matrix(result.x).T
    n1_norm = mixed_strategy.sum()
    if not n1_norm > 0: raise ValueError("The result of linear programming returned a non positive vector...")
    mixed_strategy = mixed_strategy / n1_norm
#   print(mixed_strategy, A * mixed_strategy, sep = "\n\n", end = "\n\n")
    return mixed_strategy

def supportEnumeration(game:Game) -> list[tuple[np.matrix]]:
    A = game.payoffMatrix(player = "row")
    B = game.payoffMatrix(player = "column")
    A_shift_scalar = abs(A.min())*2
    B_shift_scalar = abs(B.min())*2
    A += A_shift_scalar
    B += B_shift_scalar
    possible_supports_row = possibleSupports(game = game, player = "row")
    possible_supports_column = possibleSupports(game = game, player = "column")
    good_pairs = []
    for s1 in possible_supports_row[:]:
        for s2 in possible_supports_column[:]:
            s1:np.matrix
            s2:np.matrix
            try:
                p = maxIndifferent(opponents_payoff_matrix = B,
                                    opponents_support = s2,
                                    opponents_type = "column",
                                    players_support = s1)
                q = maxIndifferent(opponents_payoff_matrix = A,
                                    opponents_support = s1,
                                    opponents_type = "row",
                                    players_support = s2)
            except LinearProgrammingError: continue
            good_pairs.append((p.T, q.T))
    return good_pairs

def inPolyhedron(point:np.matrix, polyhedronAsMatrix:np.matrix, display = False) -> bool:
    "Note that the columns of the matrix are interpreted as the vertices"
    n, m = polyhedronAsMatrix.shape
    if not point.shape[1] == 1:
        raise ValueError(f"We need the point to be expressed as a column vector: {point.shape}")
    if not point.shape[0] == n:
        raise ValueError(f"The dimensions are not compatible: {polyhedronAsMatrix.shape}, {point.shape}")
    lin_set = set(range(n+1))
    normalization_A = np.ones((1, m))
    normalization_b = np.array([1])
    A_eq = np.vstack([polyhedronAsMatrix, normalization_A])
    b_eq = np.vstack([point, normalization_b])
    nonnegativity_A = np.eye(N = m, M = m)
    nonnegativity_b = np.zeros((m, 1))
    A = np.vstack([A_eq, nonnegativity_A])
    b = np.vstack([b_eq, nonnegativity_b])
    b_A = np.array(np.hstack([-b, A]))
    M_Hrep = cdd.matrix_from_array(b_A, rep_type = cdd.RepType.INEQUALITY, lin_set = lin_set)
    P_Hrep = cdd.polyhedron_from_matrix(M_Hrep)
    M_Vrep = cdd.copy_generators(P_Hrep)
    convexHullSubset = pd.DataFrame(M_Vrep.array)
    if display: print(convexHullSubset)
    return 0 not in convexHullSubset.shape

def extractSurvivedFrequency(frequency:np.float64, 
                             cumsum:np.float64, 
                             survivalPressure:np.float64) -> np.float64:
    if cumsum <= survivalPressure: return frequency
    if cumsum-frequency > survivalPressure: return np.float64(0)
    return survivalPressure-cumsum+frequency

def evolve(state:pd.DataFrame,
           payoffMatrix:np.matrix, 
           survivalPressure:np.float64) -> pd.DataFrame:
    '''state: the column vector consists of frequcies, stored in a DataFrame, where the index in an ascending order.
       payoffMatrix: the row payoff matrix of a symmetric game. '''
    if not isinstance(state, pd.DataFrame): raise TypeError("We need the state vector to be stored in a pandas DF")
    if not payoffMatrix.shape[0] == payoffMatrix.shape[1]: raise ValueError("Not a symmetric game")
    if not state.shape[0] == payoffMatrix.shape[1]: raise ValueError("The state is not correct in # of types")
    if not state.shape[1] == 1: raise ValueError("The state is not a column vector")
    if not (survivalPressure >= 0 and survivalPressure <= 1): raise ValueError("The pressure should be in [0,1]")
    state.columns = ["Frequencies"]
    state["Frequencies"] /= state["Frequencies"].sum()
    state["Fitnesses"] = payoffMatrix*(state[["Frequencies"]].values)
    state["OrderedFitnessCumsum"] = state.sort_values(by = "Fitnesses", ascending = False)["Frequencies"].cumsum()
    state["SurvivedFrequency"] = [extractSurvivedFrequency(f_i, c_i) 
                                  for f_i, c_i in zip(state["Frequencies"], state["OrderedFitnessCumsum"])]
    state["SurvivedFrequencyNormalized"] = state["SurvivedFrequency"] / state["SurvivedFrequency"].sum()
    state = state.sort_index(ascending = True)
    return state[["SurvivedFrequencyNormalized"]]

def permuteMatrix(matrix:np.matrix, rowOrder:tuple[int]) -> np.matrix:
    df = pd.DataFrame(matrix)
    df.index = rowOrder
    df = df.sort_index(ascending = True)
    return np.matrix(df.values)

def permutationVertices(payoffMatrix:np.matrix, fitnessOrder:tuple[int], lin_set = {0}) -> pd.DataFrame:
    '''Note that the order is not strict, and the fitness order is set to "ascending". 
    The variable "lin_set" denotes the indices of rows in the constraints that are equalities, 
    default to {0} meaning only the first row, the probablility sum constraint is an equality. '''
    n, m = payoffMatrix.shape
    # Coefficients of the probability hyperplane
    lin_set = lin_set
    normalization_A = np.ones((1, m))
    normalization_b = np.array([1])
    nonnegativity_A = np.eye(N = m, M = m)
    nonnegativity_b = np.zeros((m, 1))
    probability_A = np.vstack([normalization_A, nonnegativity_A])
    probability_b = np.vstack([normalization_b, nonnegativity_b])
    # Coefficients of the permutation constraints
    permutedMatrix = permuteMatrix(matrix = payoffMatrix, rowOrder = fitnessOrder)
    permutationConstraint_rows = []
    for i in range(n-1):
        larger, lesser = permutedMatrix[i, :], permutedMatrix[i+1, :]
        permutationConstraint_rows.append(larger-lesser)
    permutationConstraint_A = np.vstack(permutationConstraint_rows)
    permutationConstraint_b = np.zeros((n-1, 1))
    # Feed the constraints to the double description module
    A = np.vstack([probability_A, permutationConstraint_A])
    b = np.vstack([probability_b, permutationConstraint_b])
    b_A = np.array(np.hstack([-b, A]))
    M_Hrep = cdd.matrix_from_array(b_A, rep_type = cdd.RepType.INEQUALITY, lin_set = lin_set)
    P_Hrep = cdd.polyhedron_from_matrix(M_Hrep)
    M_Vrep = cdd.copy_generators(P_Hrep)
    # Output as a pandas DataFrame
    probabilitySubset = pd.DataFrame(M_Vrep.array)
    if 0 in probabilitySubset.shape: raise pd.errors.EmptyDataError("This order is not possible for this matrix")
    if (probabilitySubset.iloc[:, 0] != 1).any(): raise ValueError("Unbounded polyhedron")
    probabilitySubset = probabilitySubset.iloc[:, 1:]
    probabilitySubset.index = [f"Vertex_{i}" for i in range(probabilitySubset.shape[0])]
    probabilitySubset.columns = [f"Probability_{i}" for i in range(probabilitySubset.shape[1])]
    probabilitySubset = probabilitySubset.T
    return probabilitySubset

def permutationPolyhedra(payoffMatrix:np.matrix, display = False) -> dict[tuple[int], pd.DataFrame]:
    result = {}
    for i in permutations(range(payoffMatrix.shape[0])):
        try:
            vertices_i = permutationVertices(payoffMatrix, i)
            if display: print(f"{i} yields: \n{vertices_i}\n")
            result[i] = vertices_i
        except KeyboardInterrupt: raise KeyboardInterrupt
        except pd.errors.EmptyDataError:
            if display: print(f"{i} yields nothing\n")
    return result

def cull(state:np.matrix, fitnesses:np.matrix, surviveRatio:float) -> np.matrix:
    df = pd.DataFrame(np.hstack((state, fitnesses)))
    df.index = [f"PlayerType_{i}" for i in range(df.shape[0])]
    df.columns = ["Frequency", "Fitness"]
    df["FitnessRank"] = df["Fitness"].rank(method = "average", ascending = False)
    fitnessRanks = df["FitnessRank"].drop_duplicates(inplace = False).sort_values(ascending = True)
    result = df["Frequency"].copy()*0
    currentCumulativeFrequency = 0.0
    for rank_i in fitnessRanks:
        if currentCumulativeFrequency >= surviveRatio: break
        frequencies_i = df["Frequency"][df["FitnessRank"] == rank_i]
        sum_i:float = frequencies_i.sum()
        if sum_i == 0: continue # Being 0 means they are all culled
        if currentCumulativeFrequency+sum_i >= surviveRatio:
            spaceLeft = surviveRatio - currentCumulativeFrequency # Expecting a positive probability here
            normalizationCoefficient = spaceLeft/sum_i
            for index_j in frequencies_i.index: result.loc[index_j] = frequencies_i.loc[index_j]*normalizationCoefficient
            currentCumulativeFrequency += spaceLeft
        else:
            for index_j in frequencies_i.index: result.loc[index_j] = frequencies_i.loc[index_j]
        currentCumulativeFrequency += sum_i
    result /= surviveRatio
    return np.matrix(result.to_numpy()).T


