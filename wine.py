import numpy as np

#
# Transfer functions
#
class TransferFunctions:
    def relu(x, derivative=False):
        if (derivative == True):
            for i in range(0, len(x)):
                for k in range(len(x[i])):
                    if x[i][k] > 0:
                        x[i][k] = 1
                    else:
                        x[i][k] = 0
            return x
        for i in range(0, len(x)):
            for k in range(0, len(x[i])):
                if x[i][k] > 0:
                    pass  # do nothing since it would be effectively replacing x with x
                else:
                    x[i][k] = 0
        return x

    def sgm(x, Derivative=False):
        if not Derivative:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            #out = sgm(x)
            out = 1.0 / (1.0 + np.exp(-x))
            return out * (1.0 - out)
    
    def linear(x, Derivative=False):
        if not Derivative:
            return x
        else:
            return 1.0
    
    def gaussian(x, Derivative=False):
        if not Derivative:
            return np.exp(-x**2)
        else:
            return -2*x*np.exp(-x**2)
    
    def tanh(x, Derivative=False):
        if not Derivative:
            return np.tanh(x)
        else:
            return 1.0 - np.tanh(x)**2
    
    def truncLinear(x, Derivative=False):
        if not Derivative:
            y = x.copy()
            y[y < 0] = 0
            return y
        else:
            return 1.0
        
#
# Classes
#
class BackPropagationNetwork:
    """A back-propagation network"""
    
    #
    # Class methods
    #
    def __init__(self, layerSize, layerFunctions=None):
        """Initialize the network"""
        
        self.layerCount = 0
        self.shape = None
        self.weights = []
        self.tFuncs = []
        
        # Layer info
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize
        
        if layerFunctions is None:
            lFuncs = []
            for i in range(self.layerCount):
                if i == self.layerCount - 1:
                    lFuncs.append(TransferFunctions.linear)
                else:
                    lFuncs.append(TransferFunctions.sgm)
        else:
            if len(layerSize) != len(layerFunctions):
                raise ValueError("Incompatible list of transfer functions.")
            elif layerFunctions[0] is not None:
                raise ValueError("Input layer cannot have a transfer function.")
            else:
                lFuncs = layerFunctions[1:]
        
        self.tFuncs = lFuncs
        
        # Data from last Run
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []
        
        # Create the weight arrays
        for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.01, size = (l2, l1+1)))
            self._previousWeightDelta.append(np.zeros((l2, l1+1)))
    
    #
    # Run method
    #
    def Run(self, input):
        """Run the network based on the input data"""
        
        lnCases = input.shape[0]
        
        # Clear out the previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []
        
        # Run it!
        for index in range(self.layerCount):
            # Determine layer input
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))
            
            self._layerInput.append(layerInput)
            self._layerOutput.append(self.tFuncs[index](layerInput))
        
        return self._layerOutput[-1].T
                 
    #
    # TrainEpoch method
    #
    def TrainEpoch(self, input, target, trainingRate = 0.01, momentum = 0.7): #trainingRate = 0.02, momentum = 0.5
        """This method trains the network for one epoch"""
        
        delta = []
        lnCases = input.shape[0]
        
        # First run the network
        self.Run(input)
        
        # Calculate our deltas
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                # Compare to the target values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.tFuncs[index](self._layerInput[index], True))
            else:
                # Compare to the following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.tFuncs[index](self._layerInput[index], True))
            
        # Compute weight deltas
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index
            
            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, lnCases])])
            else:
                layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])
            
            curWeightDelta = np.sum(\
                                 layerOutput[None,:,:].transpose(2, 0 ,1) * delta[delta_index][None,:,:].transpose(2, 1, 0)\
                                 , axis = 0)
            
            weightDelta = trainingRate * curWeightDelta + momentum * self._previousWeightDelta[index]
            
            self.weights[index] -= weightDelta
            
            self._previousWeightDelta[index] = weightDelta
        
        return error
    
#normalization function
def norm(dizi):
    #global Maxs, Mins
    #Maxs = np.amax(dizi,axis=0)
    #Mins = np.amin(dizi,axis=0)
    A = []
    for d in dizi:
        a = []
        for i in range(d.shape[0]):
            new = (d[i]-Mins[i])/(Maxs[i]-Mins[i])
            a.append(new)
        A.append(a)
    New = np.array(A)
    return New
    
#sorgu yapma fonksiyonu
def Sor(inp):
    A = []
    a = []
    for i in range(inp.shape[0]):
        new = (inp[i]-Mins[i])/(Maxs[i]-Mins[i])
        a.append(new)
    A.append(a)
    New = np.array(A)
    print("normalize edildi: ",New)
    deger = bpn.Run(New)
    return deger

#
# If run as a script, create a test object
#
if __name__ == "__main__":
 
    #veri içeri alma
    from numpy import genfromtxt 
    my_data = genfromtxt('winequality-white.csv', delimiter=';',skip_header=1)
    print("dosyadan veri okundu...")
    #for normalizaiton
    global Maxs, Mins
    Maxs = np.amax(my_data,axis=0)
    Mins = np.amin(my_data,axis=0) 
    print("alt üst degerler saptandı...")
    #print(Maxs,Mins)
    
    ayrim = 200
    train = my_data[:ayrim,:] #my_data[:1200,:]
    test = my_data[ayrim:,:]
    train_input = train[:,:11]
    train_target= train[:,11:]
    test_input = test[:,:11]
    test_target = test[:, 11:]
    print("train test setleri ayrıldı. veri satır sayısı: ",train.shape )
    
    learning_rate = 0.02
    momentum = 0.5
    
    #lvTarget = norm(train_target)
    lvTarget = train_target/10
    lvInput = norm(train_input)
    #katman sayısı kadar Transfer Fonksiyonu EKLE
    lFuncs = [None, TransferFunctions.relu, TransferFunctions.sgm]
    #Katman yapısını belirle
    bpn = BackPropagationNetwork((11,15,1), lFuncs)
    print("Neural Network olusturuldu... Yapı: ",bpn.shape)
    #iterasyon sayısı ve istenilen hata oranı gir
    lnMax = 10000 #50000
    lnErr = 1e-1  #1e-6 0.1
    print("train basladı. epoch: ",lnMax," L_rate",learning_rate," momentum: ",momentum)
    for i in range(lnMax+1):
        err = bpn.TrainEpoch(lvInput, lvTarget,learning_rate, momentum)
        if i % 2000 == 0 and i > 0: #5000
            print("Iteration {0:6d}K - Error: {1:0.6f}".format(int(i/1000), err))
        if err <= lnErr:
            print("İstenilen hata oranına ulaşıldı. Iter: {0}".format(i))
            break     
    print("train tamamlandı...")   
        
    # Display output
    #lvOutput = bpn.Run(lvInput)
    #for i in range(train_input.shape[0]):
        #print("Input: {0} Output: {1} expect: {2}".format(train_input[i], lvOutput[i],lvTarget[i]))
     #   print("Output: {0} expect: {1}".format(lvOutput[i],lvTarget[i]))
#test safhası
    print("test başladı...")
    test_input_norm = norm(test_input)
    test_output = bpn.Run(test_input_norm)
    dogru = 0
    for i in range(test_output.shape[0]):
        if round(test_output[i][0],1) == test_target[i][0]/10:
            dogru = dogru + 1
    print("oran: ",100*dogru/i)