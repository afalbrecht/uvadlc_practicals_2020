# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################


def train(config, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print(device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size, config.seq_length, 
        dataset.vocab_size, config.lstm_num_hidden,
        config.lstm_num_layers, config.device
    ).to(device)

    # Setup the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    loss_history = []
    acc_history = []
    # state = model.init_state()

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # print(batch_inputs)
        # print(len(batch_inputs))

        # Only for time measurement of step through network
        t1 = time.time()

        # Move to GPU
        batch_inputs = torch.Tensor(torch.cat([x.float().unsqueeze(dim=0) for x in batch_inputs])).long().to(device)
        batch_targets = torch.Tensor(torch.cat([y.float().unsqueeze(dim=0) for y in batch_targets])).long().to(device)
        # batch_inputs = torch.cat(batch_inputs).long().to(device)
        # batch_targets = torch.cat(batch_targets).long().to(device)

        # print(batch_inputs)
        # print(batch_inputs.size())

        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        # log_probs, state = model(batch_inputs, state)
        log_probs = model(batch_inputs)
        
        # print('log_probs:', log_probs.size())
        # print(batch_targets.size())
        # print('log_probs:', log_probs.transpose(0, 1).size())

        # one_hot = nn.functional.one_hot(batch_targets, dataset.vocab_size)

        # loss_list = torch.Tensor([criterion(char, target) for char, target in zip(log_probs, batch_targets)])
        loss = criterion(log_probs.transpose(1, 2), batch_targets)
        loss.backward()
        # print(log_probs.size())
        # print(loss_list[0])
        # print(type(loss_list[0]))
        # loss = torch.mean(loss_list)
        # loss_list = []
        # for char, target in zip(log_probs, batch_targets):
        #     loss = criterion(char, target)
        #     loss.backward()
        #     loss_list.append(loss.item())
        # loss = np.mean(loss_list)

        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        
        optimizer.step()

        predictions = torch.argmax(log_probs, dim=-1)    # FIXME
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / (log_probs.size(1) * log_probs.size(0))

        # predictions = torch.argmax(log_probs, dim=-1)    # FIXME
        # correct = (predictions == batch_targets).sum().item()
        # accuracy = correct / log_probs.size(1)


        loss_history.append(loss.item())
        acc_history.append(accuracy)

        # if step % 2e4 == 0:
        #     print('\nLoss:', loss.item())
        #     print('Acc:', accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if (step + 1) % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    print('Final loss:', loss_history[-1])
    print('Final acc:', acc_history[-1])
    return loss_history, acc_history

###############################################################################
###############################################################################

def draw_plot(acc, loss, seq_length):
    mean = np.mean(loss, axis=0)
    std = np.std(loss, axis=0)
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    mean_acc = np.mean(acc, axis=0)
    std_acc = np.std(acc, axis=0)
    plt.plot(mean_acc)
    plt.fill_between(range(len(mean_acc)), mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)
    plt.legend(['loss', 'accuracy'])
    # plt.show()
    plt.savefig(f'/home/lgpu0376/code/output_dir/loss_acc_shakespeare_{seq_length}.png')
    # plt.savefig(f'loss_acc_{model}_{seq_length}.png')

    plt.clf()

    with open('/home/lgpu0376/code/output_dir/output_shakespeare.out', 'a+') as f:
        f.write(f'\nFinal loss for seq_length {seq_length}: {mean[-1]}')  
        f.write(f'\nFinal acc for seq_length {seq_length}: {mean_acc[-1]}')
        f.write(f'\nFinal loss std for seq_length {seq_length}: {np.mean(std)}')  
        f.write(f'\nFinal acc std for seq_length {seq_length}: {np.mean(std_acc)}')


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, #TODO: required=True
                        default="Part 2/assets/book_EN_shakespeare_complete.txt",
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,   #TODO: 30
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    loss, acc = train(config)
    draw_plot(acc, loss, config.seq_length)
