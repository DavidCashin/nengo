import matplotlib.pyplot as plt

import nengo
from nengo import spa


# model init based off of question answering network with memory
dimensions = 32

#new sequence to insert into existing categories
newSequence = ['One','Four','Four','Two']

#The inputs can be visualized as a 2D graph. x_input values are locations along the x axis,
#while y_output are the values at those location. This fits nicely with the final output.

#just using 4 x values, 4 y values since I lack computational speed
#we assume a square matrix for x and y vals
xVals = ['A','B','C','D']
yVals = ['One','Two','Three','Four']

#will make a new category if new sequence val is off from original mapping by 3 or more
vigilance = 3 

#model = spa.SPA(label="Simple question answering")
model = spa.SPA()

with model:
    model.x_in = spa.Buffer(dimensions=dimensions)
    model.y_out = spa.Buffer(dimensions=dimensions)
    model.conv = spa.Memory(dimensions=dimensions, subdimensions=4, synapse=0.4)
    model.cue = spa.Buffer(dimensions=dimensions)
    model.out = spa.Buffer(dimensions=dimensions)
    
    cortical_actions = spa.Actions(
        'conv = x_in * y_out',
        'out = conv * ~cue'
    )
    model.cortical = spa.Cortical(cortical_actions)  

def x_input(t):
    if t < 0.25:
        return xVals[0]
    elif t < 0.5:
        return xVals[1]
    elif t < 0.75:
        return xVals[2]
    elif t < 1.0:
        return xVals[3]
    else:
        return '0'

#even though the values are words, they correspond to actual quantitative relationships to 
#the input. For example, using C with a value of Four will produce a closer-to-optimal value
#when the input sequence is C:3 than when it is C:1 since 3 is closer to 5 than 1 is.
def y_output(t):
    if t < 0.25:
        return yVals[0]
    elif t < 0.5:
        return yVals[1]
    elif t < 0.75:
        return yVals[2]
    elif t < 1.0:
        return yVals[3]
    else:
        return '0'

def cue_input(t):
    if t < 1.0:
        return '0'
    #initial mapping

    #assemble semantic pointer string
    spStr = ['0']
    for i in range(len(xVals)):
        spStr.insert(len(spStr),xVals[i])
        spStr.insert(len(spStr),yVals[i]) 

    #sequence=['0', 'A', 'One', '0', 'B', 'Two', '0', 'C', 'Three', '0', 'D', 'Four']
    idx = int(((t - 1.0) // (1. / len(spStr))) % len(spStr))
    return spStr[idx]

with model:
    model.inp = spa.Input(x_in=x_input, y_out=y_output, cue=cue_input)

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
originalMapping = []
for i in range(len(xVals)):
    originalMapping.insert(len(originalMapping),xVals[i]+' * '+yVals[i])
for pointer in originalMapping:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.xlabel('Original mapping')

#new values introduced to existing categories (existing positions)
plt.subplot(5, 1, 3)
for pointer in ['A * ' +newSequence[0], 'B * '+newSequence[1], 'C * '+newSequence[2], 'D * '+newSequence[3]]:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.xlabel("Trained sequence")


#updated category mappings superimposing new data and original mappings, representing a
#combination of similarity. That is, similar mappings will become obvious (ie, an A:One + A:One)
#while dissimilar mappings become averaged (A:One + A:Two)
plt.subplot(5, 1, 5)
#TODO add vigilance parameter here (only add if close enough, otherwise reinitialize the 
#final output graph with a new value?)
finalCats = []
newCats = []

for pointer in ['A * One + A * One', 'B * Two + B * Four', 'C * Three + C * Four', 'D * Four + D * Two']:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.xlabel('Final mapping')

plt.show();