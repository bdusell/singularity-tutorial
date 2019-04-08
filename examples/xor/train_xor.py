import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy
import torch

def data_to_tensor_pair(data, device):
    x = torch.tensor([x for x, y in data], device=device)
    y = torch.tensor([y for x, y in data], device=device)
    return x, y

def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        return compute_accuracy(y_pred, y)

def compute_accuracy(predictions, expected):
    correct = 0
    total = 0
    for y_pred, y in zip(predictions, expected):
        correct += round(y_pred.item()) == round(y.item())
        total += 1
    return correct / total

def construct_model(hidden_units, num_layers):
    layers = []
    prev_layer_size = 2
    for layer_no in range(num_layers):
        layers.extend([
            torch.nn.Linear(prev_layer_size, hidden_units),
            torch.nn.Tanh()
        ])
        prev_layer_size = hidden_units
    layers.extend([
        torch.nn.Linear(prev_layer_size, 1),
        torch.nn.Sigmoid()
    ])
    return torch.nn.Sequential(*layers)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1.0)
    parser.add_argument('--output')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print('CUDA is available -- using GPU')
        device = torch.device('cuda')
    else:
        print('CUDA is NOT available -- using CPU')
        device = torch.device('cpu')

    # Define our toy training set for the XOR function.
    training_data = data_to_tensor_pair([
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ], device)

    # Define our model. Use default initialization.
    model = construct_model(hidden_units=10, num_layers=2)
    model.to(device)

    loss_values = []
    accuracy_values = []
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    for iter_no in range(args.iterations):
        print('iteration #{}'.format(iter_no + 1))
        # Perform a parameter update.
        model.train()
        optimizer.zero_grad()
        x, y = training_data
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss_value = loss.item()
        print('  loss: {}'.format(loss_value))
        loss_values.append(loss_value)
        loss.backward()
        optimizer.step()
        # Evaluate the model.
        accuracy = evaluate_model(model, x, y)
        print('  accuracy: {:.2%}'.format(accuracy))
        accuracy_values.append(accuracy)

    if args.output is not None:
        print('saving model to {}'.format(args.output))
        torch.save(model.state_dict(), args.output)

    # Plot loss and accuracy.
    fig, ax = plt.subplots()
    ax.set_title('Loss and Accuracy vs. Iterations')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.set_xlim(left=1, right=len(loss_values))
    ax.set_ylim(bottom=0.0, auto=None)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_array = numpy.arange(1, len(loss_values) + 1)
    loss_y_array = numpy.array(loss_values)
    left_plot = ax.plot(x_array, loss_y_array, '-', label='Loss')
    right_ax = ax.twinx()
    right_ax.set_ylabel('Accuracy')
    right_ax.set_ylim(bottom=0.0, top=1.0)
    accuracy_y_array = numpy.array(accuracy_values)
    right_plot = right_ax.plot(x_array, accuracy_y_array, '--', label='Accuracy')
    lines = left_plot + right_plot
    ax.legend(lines, [line.get_label() for line in lines])
    plt.show()

if __name__ == '__main__':
    main()
