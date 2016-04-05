'''
Simply run this file in a terminal using:
    python vectorClassifier.py
An output figure will be generated describing the initial categories, the categories after initial training

The model I chose to implement is a categorization algorithm inspired by Grossberg and Carpenter's ART (Adaptive Resonance Theory) model (http://cns.bu.edu/Profiles/Grossberg/CarGroRos1991NNART2A.pdf). ART has been in development for over 25 years, with both supervised and unsupervised learning implementations. It incorporates a 'vigilance' parameter to regulate the creation of new information that doesn't already fit into categories we have established in our minds. The related neurobiological areas associated with categorization that ART attempts to model are the parietal, prefrontal, hippocampal areas, amoung others. A detailed review of ART writen by Grossberg, including biological implications, can be found at http://www.scholarpedia.org/article/Adaptive_resonance_theory

My model works by initially using an array of predifined categories. In this case, I match numbers to points along a line to represent perhaps a 2D shape, object or wave. The existing categories are comprised of xVals (can literally be though of as x points along a graph) and yVals (the corresponding values to those points). On the generated graph, the first image is the initial mapping of x positions to values. All have relatively equal activation. The second generated image represents the combination of the previously learned category data with the new data added to that category. This is just to show an intermediate step before deciding whether data from this new sequence needs to have its own category. Finally, the last image represents the final category data after deciding which parts of the data were so different that they required their own categories (via the vigilance parameter). Newly generated categories are prefaced with NewCat. When a new category is generated, the category it was initially assigned to reverts back to the state it was in before new data was introduced to it.


Increasing the vigilance parameter will decrease the sensitivity existing categories have for different information. For example, using the newSequence array I have already defined, run the script. Next, run the script using a vigilance parameter of 4 instead of 3. You will notice the final output graph no longer has a new category assigned to it. This is because the vigilance parameter did not deem it different enough to warrant a new category. Using different number combinations in the newSequence variable, you will see how the actual quantitative data the word 'One', 'Two' or 'Three' represent, are applied to the quantitative bounds of the category. This is because the y_value output function orders the word-numbers, giving them increasing value.


'''

import matplotlib.pyplot as plt

import nengo
from nengo import spa


# model init based off of question answering network with memory
dimensions = 32

#new sequence to insert into existing categories
newSequence = ['One','Three','Four','One']

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

#Even though the values are words, they correspond to actual quantitative relationships to 
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

    #assemble semantic pointer string; initial mapping
    spStr = ['0']
    for i in range(len(xVals)):
        spStr.insert(len(spStr),xVals[i])
        spStr.insert(len(spStr),yVals[i]) 

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


#Updated category mappings superimposing new data and original mappings, representing a
#combination of similarity. That is, similar mappings will become obvious (ie, an A:One + A:One)
#while dissimilar mappings become averaged (A:One + A:Two)
plt.subplot(5, 1, 5)
finalCats = []
newCats = []
for i in range(len(xVals)):
    if(abs(i - yVals.index(newSequence[i])) < vigilance):
        #if the number in the new sequence is within the acceptable range to be added to 
        #that category, we go ahead and add it.
        finalCats.insert(len(finalCats),''+xVals[i]+' * '+yVals[i]+' + '+xVals[i]+' * '+newSequence[i])
    else:
        #keep the original sequence as it was
        finalCats.insert(len(finalCats),''+xVals[i]+' * '+yVals[i])
        #but add a new category representing this outlier
        finalCats.insert(len(finalCats),'NewCat'+str(i)+' * '+newSequence[i])

for pointer in finalCats:#ex:['A * One + A * One', 'B * Two + B * Four', 'C * Three + C * Four', 'D * Four + D * Two']:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.xlabel('Final mapping')

plt.show();