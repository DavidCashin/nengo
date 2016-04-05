import matplotlib.pyplot as plt

import nengo
from nengo import spa


# model init based off of question answering network with memory
dimensions = 32

#model = spa.SPA(label="Simple question answering")
model = spa.SPA()

with model:
    model.x_in = spa.Buffer(dimensions=dimensions)
    model.y_out = spa.Buffer(dimensions=dimensions)
    model.conv = spa.Memory(dimensions=dimensions, subdimensions=4, synapse=0.4)
    model.cue = spa.Buffer(dimensions=dimensions)
    model.out = spa.Buffer(dimensions=dimensions)
    
    # Connect the buffers
    cortical_actions = spa.Actions(
        'conv = x_in * y_out',
        'out = conv * ~cue'
    )
    model.cortical = spa.Cortical(cortical_actions)  

def x_input(t):
    if t < 0.25:
        return 'A'
    elif t < 0.5:
        return 'B'
    elif t < 0.75:
        return 'C'
    elif t < 1.0:
        return 'D'
    else:
        return '0'

#even though the values are words, they correspond to actual quantitative relationships to 
#the input. For example, using C with a value of Four will produce a closer-to-optimal value
#when the input sequence is C:3 than when it is C:1.
def y_output(t):
    if t < 0.25:
        return 'One'
    elif t < 0.5:
        return 'Two'
    elif t < 0.75:
        return 'Three'
    elif t < 1.0:
        return 'Four'
    else:
        return '0'

def cue_input(t):
    if t < 1.0:
        return '0'
    sequence = ['0', 'A', 'One', '0', 'B', 'Two', '0', 'C', 'Three', '0', 'D', 'Four']
    idx = int(((t - 1.0) // (1. / len(sequence))) % len(sequence))
    return sequence[idx]

with model:
    model.inp = spa.Input(x_in=x_input, y_out=y_output, cue=cue_input)


# ## Probe the output

# In[ ]:

with model:
    model.config[nengo.Probe].synapse = nengo.Lowpass(0.03)
    x_in = nengo.Probe(model.x_in.state.output)
    y_out = nengo.Probe(model.y_out.state.output)
    cue = nengo.Probe(model.cue.state.output)
    conv = nengo.Probe(model.conv.state.output)
    out = nengo.Probe(model.out.state.output)


sim = nengo.Simulator(model)
sim.run(1.5)

plt.figure(figsize=(10, 10))
vocab = model.get_default_vocab(dimensions)

#fig = plt.figure()
#fig.suptitle("'0', 'One', 'A', '0', 'Four', 'B', '0', 'Four', 'C', '0', 'Two', 'D'")

#original position-value mappings
plt.subplot(5, 1, 1)
for pointer in ['A * One', 'B * Two', 'C * Three', 'D * Four']:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.ylabel('Original mapping')

#new values introduced to existing categories (existing positions)
plt.subplot(5, 1, 3)
for pointer in ['A * One', 'B * Four', 'C * Four', 'D * Two']:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.ylabel("Trained sequence")


#updated category mappings superimposing new data and original mappings, representing a
#combination of similarity. That is, similar mappings will become obvious (ie, an A:One + A:One)
#while dissimilar mappings become averaged (A:One + A:Two)
plt.subplot(5, 1, 5)
for pointer in ['A * One + A * One', 'B * Two + B * Four', 'C * Three + C * Four', 'D * Four + D * Two']:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.ylabel('Final mapping')

plt.show();