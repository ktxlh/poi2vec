#! /usr/bin/env python3

import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config


class POI2VEC(nn.Module):
    def __init__(self, poi_cnt, id2route, id2lr, id2prob):
        """
        Args:
            poi_cnt (int): the number of POIs
            id2route (torch.tensor): the route of each POI
            id2lr (torch.tensor): the left-right choice of each POI
            id2prob (torch.tensor): the probability of each POI
        """
        super(POI2VEC, self).__init__()

        route_cnt = np.power(2, config.route_depth) - 1
        
        self.register_buffer('id2route', id2route)
        self.register_buffer('id2lr', id2lr)
        self.register_buffer('id2prob', id2prob)

        self.feat_dim = config.feat_dim

        self.poi_weight = nn.Embedding(poi_cnt, config.feat_dim, padding_idx=0)
        self.route_weight = nn.Embedding(route_cnt, config.feat_dim, padding_idx=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, context, target):
        """
        Args:
            context (torch.tensor): the context of the target, shape (batch_size, context_size)
            target (torch.tensor): the target POI, shape (batch_size,)
        """
        
        # route := path from root to POI in binary tree
        route = self.id2route[target] # shape (batch_size, route_count, route_depth)
        batch_size, route_count, route_depth = route.shape
        route = route.view(-1, route_count * route_depth).type(config.ltype)
        
        lr = self.id2lr[target]
        lr = torch.concat([lr, torch.zeros(batch_size, route_count, 1, device=lr.device)], dim=2)
        lr = lr.view(-1, route_count * route_depth).type(config.ftype)
        # lr shape: (batch_size, route_count * route_depth)
        
        prob = self.id2prob[target] # shape (batch_size, route_count)
        prob = prob.view(-1, route_count).type(config.ftype)
        
        context = self.poi_weight(context) # shape (batch_size, context_size, feat_dim)
        route = self.route_weight(route) # shape (batch_size, route_count * route_depth, feat_dim)

        phi_context = torch.sum(context, dim=1, keepdim=True).permute(0, 2, 1) # shape (batch_size, feat_dim, 1)
        psi_context = torch.bmm(route, phi_context) # shape (batch_size, route_count * route_depth, 1)
        psi_context = self.sigmoid(psi_context).view(-1, route_count * route_depth)
        psi_context = (torch.pow(torch.mul(psi_context, 2), lr) - psi_context)
        psi_context = psi_context.view(-1, route_count, route_depth) # shape (batch_size, route_count, route_depth)

        pr_path = torch.ones(batch_size, route_count, device=psi_context.device)
        for i in range(route_depth):
            pr_path = torch.mul(psi_context[:, :, i], pr_path)
        pr_path = torch.sum(torch.mul(pr_path, prob), 1) # shape (batch_size,)
    
        loss = -torch.mean(pr_path)

        return loss

class Rec:
    def __init__(self, coords):
        self.top, self.down, self.left, self.right = coords

    def overlap(self, a):
        dx = min(self.top, a.top) - max(self.down, a.down)
        dy = min(self.right, a.right) - max(self.left, a.left)
        return dx * dy if dx >= 0 and dy >= 0 else -1

class Node:
    theta = 0.05
    count = 0 
    leaves = []

    def __init__(self, west, east, north, south, level):
        self.left = None
        self.right = None
        self.west = west
        self.east = east
        self.north = north
        self.south = south
        self.level = level
        Node.count += 1
        self.count = Node.count

    def build(self):
        if self.level % 2 == 0:
            if (self.east - (self.west + self.east) / 2) > 2 * Node.theta:
                self.left = Node(self.west, (self.west + self.east) / 2, self.north, self.south, self.level + 1)
                self.right = Node((self.west + self.east) / 2, self.east, self.north, self.south, self.level + 1)
                self.left.build()
                self.right.build()
            else:
                Node.leaves.append(self)
        else:
            if (self.north - (self.north + self.south) / 2) > 2 * Node.theta:
                self.left = Node(self.west, self.east, self.north, (self.north + self.south) / 2, self.level + 1)
                self.right = Node(self.west, self.east, (self.north + self.south) / 2, self.south, self.level + 1)
                self.left.build()
                self.right.build()
            else:
                Node.leaves.append(self)

    def find_route(self, coords):
        latitude, longitude = coords
        if self.left is None:
            return [self.count], []

        if self.level % 2 == 0:
            if self.left.east < latitude:
                prev_route, prev_lr = self.right.find_route((latitude, longitude))
                prev_lr.append(1)
            else:
                prev_route, prev_lr = self.left.find_route((latitude, longitude))
                prev_lr.append(0)
        else:
            if self.left.south < longitude:
                prev_route, prev_lr = self.left.find_route((latitude, longitude))
                prev_lr.append(0)
            else:
                prev_route, prev_lr = self.right.find_route((latitude, longitude))
                prev_lr.append(1)
        prev_route.append(self.count)
        return prev_route, prev_lr

    def find_idx(self, idx):
        for leaf in Node.leaves:
            if leaf.count == idx:
                return leaf.north, leaf.south, leaf.west, leaf.east