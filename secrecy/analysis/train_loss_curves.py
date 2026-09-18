import numpy as np 
import matplotlib.pyplot as plt
import json
import matplotlib

def plot_curves(input_paths, labels, loss_offset=0, report_tokens=False):
    
    for i, input_path in enumerate(input_paths):
        if 'memory' in input_path:
            offset = loss_offset
        else:
            offset = 0
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        for iteration in json.load(open(input_path))['log_history']:
            step = iteration['step']
            if 'loss' in iteration:
                train_losses.append(iteration['loss'] + offset)
                train_steps.append(step/1000)
            if 'eval_loss' in iteration:
                if 'mamba' in input_path:
                    test_losses.append(train_losses[-1]+ offset)
                else:
                    test_losses.append(iteration['eval_loss'] + offset)
                if report_tokens:
                    test_steps.append(step * 128 * 512 / 1e9)
                else:
                    test_steps.append(step/1000)
    
        label = labels[i]
        print (label, test_losses[-1])
        plt.plot(test_steps, test_losses, label=f'{label} Eval')
        # plt.scatter(test_steps, test_losses, label=f'{label} Eval')
        # plt.plot(train_steps, train_losses)
        # plt.scatter(train_steps, train_losses, label=f'{label} Train')

    plt.legend(fontsize='large')
    plt.tick_params(labelsize=16)
    plt.xlabel('Steps (thousand)', fontsize='x-large')
    plt.ylabel('Cross-Entropy Loss', fontsize='x-large')
    # plt.xscale('log')
    # plt.savefig('/Users/bbadger/Desktop/figure.png', dpi=350)
    plt.show()
    plt.close()
    return


paths = [
'/home/bbadger/Desktop/c16_encoder_invertibility.json',
'/home/bbadger/Desktop/s1_noninv.json',
'/home/bbadger/Desktop/s10_noninv.json',
'/home/bbadger/Desktop/s100_noninv.json',
'/home/bbadger/Desktop/s1000_50k_noninv.json'
]

labels = [
    'No Secrecy',
    '1 Secrecy Model',
    '10 Secrecy Models',
    '100 Secrecy Models',
    '1000 Secrecy Models',
]

plot_curves(paths, labels)