import numpy as np 
import matplotlib.pyplot as plt
import json
import matplotlib
import jsonlines

def plot_curves(input_paths, labels, report_tokens=False, lines=False):
    
    for i, input_path in enumerate(input_paths):
        if 'memory' in input_path:
            offset = loss_offset
        else:
            offset = 0
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        if lines:
            data = jsonlines.open(input_path)
        else:
            data = json.load(open(input_path))['log_history']
        for iteration in data:
            print (iteration)
            step = iteration['step']
            if 'loss' in iteration:
                train_losses.append(iteration['loss'] + offset)
                train_steps.append(step/1000)
            if 'inverter_loss' in iteration:
                test_losses.append(iteration['inverter_loss'])
                if report_tokens:
                    test_steps.append(step * 128 * 512 / 1e9)
                else:
                    test_steps.append(step/1000)

            if 'eval_loss' in iteration:
                test_losses.append(iteration['eval_loss'])
                if report_tokens:
                    test_steps.append(step * 128 * 512 / 1e9)
                else:
                    test_steps.append(step/1000)
            if 'eval_inverter_loss' in iteration:
                test_losses.append(iteration['eval_inverter_loss'])
                if report_tokens:
                    test_steps.append(step * 128 * 512 / 1e9)
                else:
                    test_steps.append(step/1000)

    
        label = labels[i]
        print (label, test_losses[-1])
        plt.scatter(test_steps, test_losses, label=f'{label} Eval', alpha=0.5)
        plt.plot(test_steps, test_losses)
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
'/home/bbadger/Desktop/transformer_inversion_data.jsonl',
'/home/bbadger/Desktop/transformer_inversion_train.jsonl'
]

labels = [
'Eval', 
'Train'
]

# paths = [
# '/home/bbadger/Desktop/c16_encoder_invertibility.json',
# '/home/bbadger/Desktop/s1_noninv.json',
# '/home/bbadger/Desktop/s10_noninv.json',
# '/home/bbadger/Desktop/s100_noninv.json',
# '/home/bbadger/Desktop/s1000_50k_noninv.json'
# ]

# labels = [
#     'No Secrecy',
#     '1 Secrecy Model',
#     '10 Secrecy Models',
#     '100 Secrecy Models',
#     '1000 Secrecy Models',
# ]

plot_curves(paths, labels, lines=True, report_tokens=False)