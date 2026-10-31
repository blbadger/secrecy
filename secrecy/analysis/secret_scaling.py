import numpy as np 
import matplotlib.pyplot as plt
import json
import matplotlib
import jsonlines

def plot_curves(input_paths, labels, report_tokens=False, report_flops=True, param_scaling=False, lines=False):

    for i, input_path in enumerate(input_paths):
        train_steps, test_steps = [], []
        train_losses, test_losses = [], []
        compute_factor = int(input_path.split('_')[1])**2 * int(input_path.split('_')[2].strip('n.json')) * 1.5e7
        if lines:
            compute_factor = compute_factor*1.375
        
        parameters = [2.5, 10, 55, 252, 400]

        if lines:
            data = jsonlines.open(input_path)
        else:
            data = json.load(open(input_path))['log_history']
        
        for iteration in data:
            print ('iteration', iteration)
            step = iteration['step']
            
            if 'eval_loss' in iteration or 'eval_clm_loss' in iteration:
                loss = None
                if 'eval_clm_loss' in iteration:
                    loss = iteration['eval_clm_loss']

                elif 'eval_loss' in iteration:
                    loss = iteration['eval_loss']

                if param_scaling:
                    test_losses.append(parameters[i])
                else:
                    test_losses.append(loss)

                if report_tokens:
                    test_steps.append(step * 128 * 512 / 1e9)

                elif report_flops:
                    test_steps.append(step * compute_factor)

                else:
                    test_steps.append(step/1000)

        label = labels[i]
        print (label, test_losses[-1], test_steps)
        plt.plot(test_steps, test_losses)
        plt.scatter(test_steps, test_losses, label=f'{label}')

    plt.legend(fontsize='large', framealpha=0.95)
    plt.tick_params(labelsize=16)
    plt.xlabel('FLOPs', fontsize='x-large')
    plt.ylabel('Loss', fontsize='x-large')
    plt.xscale('log')
    return

transformer_colormap = plt.cm.Reds
toeplitz_colormap = plt.cm.Blues
mamba_colormap = plt.cm.viridis


mamba_cycler = plt.cycler('color', mamba_colormap(np.linspace(0.1, 0.95, 6)))
tr_cycler = plt.cycler('color', transformer_colormap(np.linspace(0.1, 0.95, 6)))
toeplitz_cycler = plt.cycler('color', toeplitz_colormap(np.linspace(0.1, 0.95, 6)))
# all_cycler = mamba_cycler.concat(tr_cycler).concat(toeplitz_cycler)
plt.gca().set_prop_cycle(tr_cycler)

# comparison
# paths= [
#     '/home/bbadger/Desktop/mamba_256_s128_c512.json',
#     '/home/bbadger/Desktop/scaling_1024_n16.json',
#     '/home/bbadger/Desktop/fineweb_mixer_1024_n16_h4_c512.json',
#     '/home/bbadger/Desktop/fineweb_transformer_512_n16_h8_c512.json'
# ]

# labels = [
#     'Mamba', 
#     'SRM', 
#     'Masked Mixer',
#     'Transformer'
# ]

# # dm scaling
paths = [
    '/home/bbadger/Desktop/transformer_64_n8.json',
    '/home/bbadger/Desktop/transformer_128_n8.json',
    '/home/bbadger/Desktop/transformer_256_n8.json',
    '/home/bbadger/Desktop/transformer_512_n8.json',
    '/home/bbadger/Desktop/transformer_512_n16.json',
    '/home/bbadger/Desktop/transformer_1024_n16.json',
]

labels = [
    'd=64, n=8',
    'd=128, n=8',
    'd=256, n=8',
    'd=512, n=8',
    'd=512, n=16',
    'd=1024, n=16',
]


plot_curves(paths, labels, lines=False)

plt.gca().set_prop_cycle(toeplitz_cycler)
#  dm scaling
paths = [
    '/home/bbadger/Desktop/secret_64_n8.json',
    '/home/bbadger/Desktop/secret_128_n8.json',
    '/home/bbadger/Desktop/secret_256_n8.json',
    '/home/bbadger/Desktop/secret_512_n8.json',
    '/home/bbadger/Desktop/secret_512_n16.json',    
    '/home/bbadger/Desktop/secret_1024_n16.json',
    # '/home/bbadger/Desktop/secret_1024_n24.json',
]

labels = [
    'd=64, n=8',
    'd=128, n=8',
    'd=256, n=8',
    'd=512, n=8',
    'd=512, n=16',
    'd=1024, n=16',
    # 'd=1024, n=24'
]

plot_curves(paths, labels, lines=True)

plt.show()
plt.close()