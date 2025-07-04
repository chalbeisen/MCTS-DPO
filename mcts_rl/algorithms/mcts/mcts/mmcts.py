# Adapted from: https://github.com/maitrix-org/llm-reasoners/blob/main/reasoners/algorithm/mcts.py

import itertools
from tqdm import trange
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Union

import os
import pickle
import math
import torch
import random
import numpy as np
import copy

from mcts_rl.algorithms.mcts.mcts.base import (
    State, Action, Example, 
    SearchAlgorithm, WorldModel, SearchConfig,
)

from mcts_rl.algorithms.mcts.mcts.mcts import *
from mcts_rl.utils import calculate_diversity_score

from mcts_rl.algorithms.mcts.mcts.sampling.puct_distribution import puct_distribution
from mcts_rl.algorithms.mcts.mcts.sampling.cap_distribution import cap_distribution


class MMCTSResult(NamedTuple):
    tree_state: MCTSNode
    search_history: dict = {}


class MMCTS(MCTS):
    def __init__(self, args: MCTSConfig):
        super().__init__(args)
        self.p_max: float = 0.25
        self.puct_inf_softening: float = 2.0
    

    def iterate(self, path: list[MCTSNode]) -> list[MCTSNode]:
        """
        iterate through selected path of the current tree and try to find a better path
        select next child according to capped puct distribution
        continue until a terminal node was reached and return new path
        """
        node = path[0]
        new_path = [node]
        node.N += 1
        it_cnt = 0
        iter_history = {}
        while not self._is_terminal_with_depth_limit(node):
            if node.children == None:
                self._expand_and_evaluate(node)

            node_puct_values = torch.tensor([float("inf") if child.N == 0 else self._puct(child) for child in node.children])
            distribution = puct_distribution(node_puct_values, self.puct_inf_softening)
            #distribution = cap_distribution(distribution, (1+p_max)/len(node.children))
            p_max = 0.8/len(node.children)
            distribution = cap_distribution(distribution, (1+p_max)/len(node.children))

            candidate_index = torch.multinomial(distribution, 1)
            child = node.children[candidate_index]

            accept_prob = 0.3
            if torch.rand(1) < accept_prob:
                node = child
                new_path.append(node)
            else:
                if it_cnt < len(path)-1 and path[it_cnt+1] in node.children:
                    node = path[it_cnt+1]
                else:
                    node = self._puct_select(node)
                new_path.append(node)
            
            iter_history[f'iter_{it_cnt}'] = {'cur_node': self.root.cpu_clone(), 'path': [node.id for node in new_path]}

            it_cnt += 1
        self._back_propagate(new_path)
        iter_history['backprob'] = {'cur_node': self.root.cpu_clone(), 'path': [node.id for node in new_path], 'reward': new_path[-1].value}
        return new_path, iter_history
    
    def _random_select(self, node: MCTSNode) -> MCTSNode:
        if len(node.children) - 1 == 0:
            idx = 0
        else:
            idx = random.randint(0, len(node.children)-1)

        return node.children[idx]

    def set_initial_path(self) -> list[MCTSNode]:
        if self.root is None:
            self.root = MCTSNode(state=self.world_model.init_state(), action=None, parent=None, length_penalty=self.length_penalty)

        node = self.root
        node.N += 1
        path = [node]

        while not self._is_terminal_with_depth_limit(path[-1]):
            if path[-1].children == None:
                self._expand_and_evaluate(node)
            node = self._random_select(node)
            node.N += 1
            path.append(node)

        return path

    def search(self):
        """
        build an initial path through tree randomly
        then try to adapt that path for n_iters times
        """
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []


        n_iters = self.n_iters if self.root.depth else self.n_iters * 4     # iterate more at the starting point
        search_history = {}
        path = self.initial_path
        search_history['initial_path'] = {'cur_node': self.root.cpu_clone(), 'path': [node.id for node in path]}
        for i in trange(n_iters, disable=self.disable_tqdm, desc='MMCTS iteration', leave=False):
            path, iter_history = self.iterate(path)
            search_history[f'search_{i}'] = iter_history
            if self.output_trace_in_each_iter:
                self.trace_in_each_iter.append(deepcopy(path))

        return search_history

    def __call__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 root_node: Optional[Union[MCTSNode, int]] = None,
                 **kwargs) -> MMCTSResult:

        self.world_model = world_model
        self.search_config = search_config
        self.root = root_node

        if root_node is None:
            MCTSNode.reset_id()
            self.initial_path = self.set_initial_path()

        self.consider_diversity = False if self.search_config.n_actions == 1 else self.consider_diversity

        search_history = self.search()
        
        return MMCTSResult(tree_state=self.root,
                           search_history=search_history)

