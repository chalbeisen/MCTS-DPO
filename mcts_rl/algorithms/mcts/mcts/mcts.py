# Adapted from: https://github.com/maitrix-org/llm-reasoners/blob/main/reasoners/algorithm/mcts.py

import itertools
from tqdm import trange
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Union
import os
import math
import torch
import numpy as np
from copy import deepcopy
import pickle
import copy
 
from mcts_rl.algorithms.mcts.mcts.base import (
    State, Action, Example, 
    SearchAlgorithm, WorldModel, SearchConfig,
)

from mcts_rl.utils import calculate_diversity_score


class MCTSConfig(NamedTuple):
    output_trace_in_each_iter: bool = False
    w_exp: float = 1.
    depth_limit: int = 5
    breadth_limit: int = 8
    n_iters: int = 10
    simulate_strategy: str | Callable[[list[float]], int] = 'max'
    disable_tqdm: bool = True
    temperature: float = 0.0
    temperature_decay_ratio: float = 0.75
    gamma: float = 1.0
    add_kl: bool = False
    consider_diversity: bool = True
    length_penalty: float = 1.25
    eval_method: str = None
    

class MCTSNode(Generic[State, Action]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(
        self, 
        state: Optional[State], 
        action: Optional[Action], 
        parent: "Optional[MCTSNode]" = None,
        base_rewards: torch.Tensor = None, 
        value: float = 0.0, 
        embeddings: torch.Tensor = None, 
        log_probs: torch.Tensor = None,
        ref_log_probs: torch.Tensor = None,
        is_terminal: bool = False,
        length_penalty: float = 1.25,
    ):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param embeddings: the embeddings of the current state (BERTScore calculation for similar generations filtering)
        :param is_terminal: whether the current state is a terminal state
        
        :param rewards: base rewards
        :param value: advantage of taking the action
        """
        self.id = next(MCTSNode.id_iter)
        self.is_terminal = is_terminal
        self.state = state
        self.action = action
        self.parent = parent
        self.embeddings = embeddings
        self.children: 'Optional[list[MCTSNode]]' = None
        self.depth = 0 if parent is None else parent.depth + 1
        self.length_penalty = length_penalty
        
        self.rewards = base_rewards
        self.log_probs = log_probs
        self.ref_log_probs = ref_log_probs
        self.value = value
        
        self.N = 0
        self.V = 0.0
        ### Q(s,a) value = expected total reard you will get if you take a action a at s and then keep following good actions
        ### it combines the immediate reward r(s,a) and the value V(s)
        ### V(s) is the expected overall quality of being in state s
        ### Q(s,a) guides action choice while V(s) summarizes the state quality
        self.Q = self.parent.V + self.r if self.parent is not None else self.r

    @property
    def r(self) -> float:
        if self.rewards is None:
            ### Parent reward minus current reward -> r(s,a)=R(s')-R(s)
            ### reward = immediate difference in quality when taking action a (so node.action)
            return self.value if self.parent is None else (self.value - self.parent.value)
        # TODO: consider KL divergence in MCTS
        # return self.rewards.mean().detach().item() + (self.value if self.parent is None else (self.value - self.parent.value))
        raise ValueError('Should not consider kl divergence here!')
    
    @property
    def p(self) -> float:
        return (self.log_probs.sum() / self.log_probs.size(-1) ** self.length_penalty).exp().detach().item()

    def cpu_clone(self, seen=None):
        import torch
        import copy

        if seen is None:
            seen = {}

        obj_id = id(self)
        if obj_id in seen:
            return seen[obj_id]

        # Shallow copy first
        clone = copy.copy(self)
        seen[obj_id] = clone

        for attr_name, attr_val in self.__dict__.items():
            clone_val = self._clone_to_cpu(attr_val, seen)
            setattr(clone, attr_name, clone_val)

        return clone

    @staticmethod
    def _clone_to_cpu(obj, seen):
        import torch
        import copy

        if id(obj) in seen:
            return seen[id(obj)]

        if isinstance(obj, torch.Tensor):
            cpu_tensor = obj.detach().cpu()
            seen[id(obj)] = cpu_tensor
            return cpu_tensor
        elif isinstance(obj, dict):
            cpu_dict = {}
            seen[id(obj)] = cpu_dict
            for k, v in obj.items():
                cpu_dict[k] = MCTSNode._clone_to_cpu(v, seen)
            return cpu_dict
        elif isinstance(obj, list):
            cpu_list = []
            seen[id(obj)] = cpu_list
            for item in obj:
                cpu_list.append(MCTSNode._clone_to_cpu(item, seen))
            return cpu_list
        elif isinstance(obj, tuple):
            cpu_tuple = tuple(MCTSNode._clone_to_cpu(i, seen) for i in obj)
            seen[id(obj)] = cpu_tuple
            return cpu_tuple
        elif isinstance(obj, set):
            cpu_set = {MCTSNode._clone_to_cpu(i, seen) for i in obj}
            seen[id(obj)] = cpu_set
            return cpu_set
        elif hasattr(obj, '__dict__'):
            clone_obj = copy.copy(obj)
            seen[id(obj)] = clone_obj
            for k, v in vars(obj).items():
                setattr(clone_obj, k, MCTSNode._clone_to_cpu(v, seen))
            return clone_obj
        else:
            return obj
        
class MCTSResult(NamedTuple):
    tree_state: MCTSNode
    next_action_pi: list[float]
    next_action_V: list[float]
    next_action_Q: list[float]
    trace_in_each_iter: list[list[MCTSNode]] = None
    next_action_idx: int = 0
    trace_of_nodes: list[MCTSNode] = None
    cum_reward: float = None
    search_history: dict = {}


class MCTS(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, args: MCTSConfig):
        """
        MCTS algorithm
        """
        super().__init__()
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = args.output_trace_in_each_iter
        self.w_exp = args.w_exp
        self.depth_limit = args.depth_limit
        self.breadth_limit = args.breadth_limit
        self.n_iters = args.n_iters
        self.gamma = args.gamma
        self.add_kl = args.add_kl
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(args.simulate_strategy,
                                                                                             args.simulate_strategy)
        self.temperature = args.temperature
        self.temperature_decay_ratio = args.temperature_decay_ratio
        self.follow_probability = False
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = args.disable_tqdm
        self.consider_diversity = args.consider_diversity
        self.length_penalty = args.length_penalty
        self.eval_method = args.eval_method
        
        self.policy_model = None
    
    def _get_simulated_pi(self, cur_node: MCTSNode, return_selection=False) -> list[float]:
        """
        Apated from: https://github.com/suragnair/alpha-zero-general/blob/ce020c8eebbabf0e22654279508a6887b4791015/MCTS.py#L28C5-L53C21
        """
        visit_counts = [child.N for child in cur_node.children]
        next_action_V = [child.V for child in cur_node.children]
        next_action_Q = [child.Q for child in cur_node.children]
        next_action_n_children = [len(child.children) if child.children is not None else 0 for child in cur_node.children]
        next_action_variance = [calculate_diversity_score(child.children) for child in cur_node.children]
        
        def _cal_probs(temp):
            if temp > 0:
                try:
                    ## choice 1: to sample based on visit counts
                    # counts = [(x * (nc + 1 if self.consider_diversity else 1)) ** (1. / temp) if x else x \
                    #     for x, nc in zip(visit_counts, next_action_n_children)]
                    ## choice 2: to sample based on Q values
                    counts = [(math.exp(x) * (nc + 1 if self.consider_diversity else 1)) ** (1. / temp) if x else x \
                        for x, nc in zip(next_action_Q, next_action_n_children)]
                    total_count = float(sum(counts))
                    probs = [x / total_count for x in counts]
                    return probs
                except OverflowError as e:
                    print(('Run into {} -- Temperature too small ... Set to zero ...').format(str(e)))
            best_actions = np.array(np.argwhere(visit_counts == np.max(visit_counts))).flatten()
            probs = [0] * len(visit_counts)
            for best_action in best_actions:
                probs[best_action] = 1 / len(best_actions)
            return probs
        
        temperature = self.temperature * (self.temperature_decay_ratio ** cur_node.depth)
        probs = _cal_probs(temperature)
        
        if return_selection:
            if temperature == 0:
                ## choice 1: to sample based on visit counts
                # selected_idx = max(range(len(visit_counts)), key=lambda x: (
                #     (next_action_Q[x] + 2) * (next_action_variance[x] + 1 if self.consider_diversity else 1), 
                #     visit_counts[x], next_action_V[x]
                # ))
                ## choice 2: to sample based on Q values
                selected_idx = max(range(len(visit_counts)), key=lambda x: (
                    visit_counts[x] * (next_action_variance[x] + 1 if self.consider_diversity else 1), 
                    next_action_Q[x], next_action_V[x]
                ))
            else:
                selected_idx = np.random.choice(range(len(visit_counts)), p=probs)
            return probs, selected_idx, next_action_V, next_action_Q
        return probs, next_action_V, next_action_Q

    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        node.N += 1
        path = self._select(node)
        it_cnt = 0
        iter_history = {}
        while not self._is_terminal_with_depth_limit(path[-1]):
            ### EXPANSION
            self._expand_and_evaluate(path[-1])
            if self._is_terminal_with_depth_limit(path[-1]) or len(path[-1].children) == 0:
                break
            ### SELECTION (Exploration with Q(s,a) and Exploitation with N)
            node = self._puct_select(path[-1])
            path.append(node)

            iter_history[f'iter_{it_cnt}'] = {'cur_node': self.root.cpu_clone(), 'path': [node.id for node in path]}

            it_cnt += 1
        self._back_propagate(path)

        iter_history['backprob'] = {'cur_node': self.root.cpu_clone(), 'path': [node.id for node in path], 'reward': path[-1].value}

        return path, iter_history


    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or (node.depth - self.root.depth) >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            node = self._puct_select(node)

    def _puct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * node.p * np.sqrt(node.parent.N) / (1 + node.N)
    
    def _puct_select(self, node: MCTSNode) -> MCTSNode:
        xnode = max(node.children, key=self._puct)
        return xnode

    def _expand_and_evaluate(self, node: MCTSNode):
        """
        add n_actions of child nodes containing a reasoning step for the given prompt
        """
        if node.state is None:
            node.state = self.world_model.step(node.parent.state, node.action, node.log_probs)
            node.is_terminal = self.world_model.is_terminal(node.state)
        
        if node.is_terminal:
            return
        
        actions = self.search_config.get_actions(self.policy_model, node.state, add_kl=self.add_kl)
        
        action_batch, log_probs_batch, ref_log_probs_batch = [], [], []
        
        for action, (log_probs, ref_log_probs), _ in actions:
            action_batch.append(action)
            log_probs_batch.append(log_probs)
            ref_log_probs_batch.append(ref_log_probs)

        ### Compute the reward of adding new nodes
        if self.eval_method == 'log_probs':
            reward_value_batch = self.search_config.get_values_logProbs(node.state,
                                                                        action_batch,
                                                                        log_probs_batch,
                                                                        ref_log_probs_batch,
                                                                        self.add_kl)
        else:
            # R(s) = O(s)+C(s) (outcome correctness + self evaluation confidence score)
            reward_value_batch = self.search_config.get_values(self.policy_model, node.state, action_batch, 
                                                           log_probs_batch, ref_log_probs_batch, 
                                                           add_kl=self.add_kl, parent_depth=node.depth,
                                                           parent_value=node.value)

        children = []
        for (action, (log_probs, ref_log_probs), embs), (value, base_rewards, is_terminal) in zip(actions, reward_value_batch):
            child = MCTSNode(state=None, action=action, parent=node, 
                             base_rewards=base_rewards, value=value, 
                             embeddings=embs, log_probs=log_probs, ref_log_probs=ref_log_probs,
                             is_terminal=is_terminal, length_penalty=self.length_penalty)
            children.append(child)
        node.children = children if node.children is None else node.children + children

    
    def _simulate(self, path: list[MCTSNode]):
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)

    def _back_propagate(self, path: list[MCTSNode]):
        node = path[-1]
        ### update Q value with path based reward + future V
        ### r(s,a) = R(s')-R(s) so the reward (value here) difference between parent and current node 
        ### V(s) = average of Q(s,a) over all actions tried at s, weighted by visit counts
        node.Q = node.r + self.gamma * node.V
        node.N += 1
        for node in reversed(path[:-1]):
            node.V = sum(max(1, child.N) * child.Q for child in node.children) / sum(max(1, child.N) for child in node.children)
            node.N += 1
            if node.action is not None:
                node.Q = node.r + self.gamma * node.V

    def search(self):
        if self.root is None:
            self.root = MCTSNode(state=self.world_model.init_state(), action=None, parent=None, length_penalty=self.length_penalty)
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        n_iters = self.n_iters if self.root.depth else self.n_iters * 4     # iterate more at the starting point
        search_history = {}
        for i in trange(n_iters, disable=self.disable_tqdm, desc='MCTS iteration', leave=False):
            path, iter_history = self.iterate(self.root)
            search_history[f'search_{i}'] = iter_history
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        return search_history

    def __call__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 root_node: Optional[Union[MCTSNode, int]] = None,
                 **kwargs) -> MCTSResult:

        self.root = root_node
        self.world_model = world_model
        self.search_config = search_config
        self.consider_diversity = False if self.search_config.n_actions == 1 else self.consider_diversity

        if root_node is None:
            MCTSNode.reset_id()

        search_history = self.search()
        
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
        else:
            trace_in_each_iter = None
        
        next_action_pi, selected_idx, next_action_V, next_action_Q = self._get_simulated_pi(self.root, return_selection=True)
        
        return MCTSResult(tree_state=self.root,
                          next_action_pi=next_action_pi,
                          next_action_V=next_action_V,
                          next_action_Q=next_action_Q,
                          trace_in_each_iter=trace_in_each_iter,
                          next_action_idx=selected_idx,
                          search_history=search_history)

