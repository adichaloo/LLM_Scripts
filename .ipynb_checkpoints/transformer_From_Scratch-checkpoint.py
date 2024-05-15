# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *



# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.num_classes=num_classes
        
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.positional = PositionalEncoding(d_model,num_positions)
        self.transformerlayer=TransformerLayer(d_model,d_internal)
        self.output_layer = nn.Linear(d_model,num_classes)
        # self.attention_score_list=[]

        # raise Exception("Implement me")

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # print(f"Indices shape: {indices.shape}") 
        emb = self.embedding(indices)
        # print(f"Embedding shape: {emb.shape}")

        residual1_pos_emb = self.positional(emb)
        # print(f"Residual network {residual1_pos_emb.shape}")
        # for i in range (self.num_layers):
        #     print(f"Layer {i}")
        #     layer_output = self.transformerlayer(residual1_pos_emb)
        # print(f"Layer outputs {layer_output.shape}")

        attention_score_list=[]
        layer_output,attention_scores = self.transformerlayer(residual1_pos_emb)
        # print(f"Layer Output : {layer_output.shape}")
        attention_score_list.append(attention_scores)
        layer_output1,attention_scores1 = self.transformerlayer(layer_output)
        attention_score_list.append(attention_scores1)
        layer_output_reshape = self.output_layer(layer_output1)

        # print(f"Before log softmax : {layer_output_reshape.shape}")
        logits = nn.functional.log_softmax(layer_output_reshape,dim=-1)
        # print(f"After log softmax : {logits.shape}") 
        # print(f"Layer Output Attention: {attention_scores.shape}") 

        return logits,attention_score_list

        # print(layers.shape)
        # raise Exception("Implement me")


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_internal= d_internal
        self.weight_q = nn.Linear(d_model,d_internal)
        self.weight_k = nn.Linear(d_model,d_internal)
        self.weight_v = nn.Linear(d_model,d_model)
        self.linear1 = nn.Linear(d_model,d_internal)
        self.linear2 = nn.Linear(d_internal,d_model)
        # raise Exception("Implement me")


    def forward(self, input_vecs):
        """
        :param input_vecs: an input tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the log probabilities of each position in the input
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """
        query = nn.functional.relu(self.weight_q(input_vecs))
        key = nn.functional.relu(self.weight_k(input_vecs))
        values = nn.functional.relu(self.weight_v(input_vecs))
        # print(f"Shape of query {query.shape}")
        attention_score = torch.matmul(query,key.T)
        attention_score = torch.divide(attention_score,torch.sqrt(torch.tensor(self.d_internal))) # Divide root dk
        # print(f"Shape of attention {attention_score.shape}")
        attention_score_adj = attention_score 
        # attention_score_adj = attention_score.tril()
        # mask = attention_score_adj == 0
        # attention_score_adj = attention_score_adj.masked_fill_(mask,-1*float("inf"))
        norm_attention_score_adj = nn.functional.softmax(attention_score_adj)
        # print(f"Shape of adjusted {attention_score_adj.shape}")
        # print(f"Shape of norm adjusted {norm_attention_score_adj.shape}") 
        hidden_layer = torch.matmul(norm_attention_score_adj,values)
        # print(f"Hidden layer {hidden_layer.shape}")
        # hidden_layer_z = self.weight_z(hidden_layer)
        # print(f"Hidden layer z after Relu {hidden_layer_z.shape}")

        concatenated = input_vecs+hidden_layer

        reshaped_concat = nn.functional.relu(self.linear1(concatenated))

        reshaped_concat_2 = self.linear2(reshaped_concat)

        concatenated_new = concatenated+ reshaped_concat_2

        # print(f"Concatenated {concatenated_new.shape}")

        return concatenated_new,norm_attention_score_adj


        # raise Exception("Implement me")


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # print(len(train))
    # print(vars(dev[0]))
    # raise Exception("Not fully implemented yet")

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    model = Transformer(27,20,50,60,3,2)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            ex = train[ex_idx]
            output, _= model.forward(ex.input_tensor)
            # output = torch.log_softmax(output, dim=-1)
            loss = loss_fcn(output,ex.output_tensor) # TODO: Run forward and compute loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
    model.eval()
    return model


def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            print(f"Attention Map shape {len(attn_maps)}")
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
