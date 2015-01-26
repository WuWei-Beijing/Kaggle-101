import theano
import theano.tensor as T
import numpy as np
from pandas import DataFrame,Series
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import pandas as pd
path='C:\\Users\\wei\\Desktop\\Kaggle\\Kaggle101\\Digit Recognizer\\'

def shared_dataset(data_xy,borrow=True):
    """
    speed up the calculation by theano,in GPU float computation.
    """
    data_x,data_y=data_xy
    shared_x=theano.shared(np.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
    shared_y=theano.shared(np.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # `shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x,T.cast(shared_y,'int32')

def load_data(path):
    print '...loading data'
    train_df=DataFrame.from_csv(path+'train.csv',index_col=False).fillna(0).astype(int)
    test_df=DataFrame.from_csv(path+'test.csv',index_col=False).fillna(0).astype(int)
    if debug_mode==False:
        train_set=[train_df.values[0:35000,1:]/255.0,train_df.values[0:35000,0]]
        valid_set=[train_df.values[35000:,1:]/255.0,train_df.values[35000:,0]]
    else:
        train_set=[train_df.values[0:3500,1:]/255.0,train_df.values[0:3500,0]]
        valid_set=[train_df.values[3500:4000,1:]/255.0,train_df.values[3500:4000,0]]
    test_set=test_df.values/255.0
    #print train_set[0][:10][:10],'\n',train_set[1][:10],'\n',valid_set[0][-10:][:10],'\n',valid_set[1][-10:],'\n',test_set[0][10:][:10]
    test_set_x=theano.shared(np.asarray(test_set,dtype=theano.config.floatX),borrow=True)
    valid_set_x,valid_set_y=shared_dataset(valid_set,borrow=True)
    train_set_x,train_set_y=shared_dataset(train_set,borrow=True)
    rval=[(train_set_x,train_set_y),(valid_set_x,valid_set_y),test_set_x]
    return rval

def dropout(X,p=0.):
    """
    Add some noise to regularize by drop out by probility p
    so to prevent overfitting
    """
    if p>0:
        retain_prob=1-p
        srng=RandomStreams()
        X*=srng.binomial(X.shape,p=retain_prob,dtype=theano.config.floatX)
        X/=retain_prob
    return X

##################################
#  The Logistic Regression Class #
##################################
class LogisticRegression(object):
    """Multi-class Logistic Regression Class"""
    def __init__(self,input,n_in,n_out,p_drop_logistic=0.2):
        """
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture(one minibatch)
        
        :type n_in: int
        :param n_in: number of input units,data dimension.
        
        :type n_out: int
        :param n_out: number of output units,label dimension.
        
        :type p_drop_logistic:float
        :param p_drop_logistic: add some noise by dropout by this probability
        """
        input=dropout(input,p_drop_logistic)
        self.W=theano.shared(value=np.zeros((n_in,n_out),dtype=theano.config.floatX),name='W',borrow=True)
        self.b=theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
        self.p_y_given_x=T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.y_pred=T.argmax(self.p_y_given_x,axis=1)
        self.params=[self.W,self.b]
        
    def negative_log_likelihood(self,y):
        """
        The cost function of multi-class logistic regression
        :type y:theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the correct label
        
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1]. T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class. LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self,y):
        """
        Return a float number of error rate: #(errors in minibatch)/#(total minibatch)
        :type y:theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the correct label
        """
        if y.ndim!=self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred')
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()
        
    def predict(self):
        return self.y_pred
        
########################################
#  The Hidden Layer (Perceptron) Class #
########################################
class HiddenLayer(object):
    def __init__(self,rng,input,n_in,n_out,W=None,b=None,activation=T.tanh,p_drop_perceptron=0.2):
        """
        Typical hidden layer of a MLP:units are fully-connected and have 
        tanh activation function.Weight matrix W is of shape (n_in,n_out),
        the bias vector b is of shape(n_out)
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        
        :type p_drop_perceptron:float
        :param p_drop_perceptron: add some noise by dropout by this probability
        """
        self.input=dropout(input,p_drop_perceptron)
        if W is None:
            W_values=np.asarray(rng.uniform(low=-np.sqrt(6./(n_in+n_out)),
                                high=np.sqrt(6./(n_in+n_out)),size=(n_in,n_out)),
                                dtype=theano.config.floatX)
            if activation==theano.tensor.nnet.sigmoid:
                W_values*=4
            W=theano.shared(value=W_values,name='W',borrow=True)
        if b is None:
            b_values=np.zeros((n_out,),dtype=theano.config.floatX)
            b=theano.shared(value=b_values,name='b',borrow=True)
        self.W=W
        self.b=b
        lin_output=T.dot(input,self.W)+self.b
        self.output=(lin_output if activation is None else activation(lin_output))    
        self.params=[self.W,self.b] 
        
    
##################################
#  The LeNet Conv&Pooling Layer  #
##################################
class LeNetConvPoolLayer(object):
    """
    A layer consists of a convolution layer and a pooling layer
    """
    def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2),p_drop_cov=0.2):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input:theano.tensor.dtensor4
        :param input: symbolic image tensor,of shape image_shape
        
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)
        
        :type image_shape:tuple or list of length 4
        :param image_shape: (batch size,num input feature maps(maps from different channels),
                             image height, image width)
        
        :type poolsize:tuple or list of length 2
        :param poolsize: the downsampling(pooling) factor(#rows,#cols)
        """
        assert image_shape[1]==filter_shape[1]
        self.input=dropout(input,p_drop_cov)
        #there are "num input feature maps(channels) * filter height * filter width"
        #input to each hidden unit
        fan_in=np.prod(filter_shape[1:])
        #each unit in the lower layer receives a gradient from:
        #"num output feature maps * filter height * filter width"
        #  / pooling size
        fan_out=(filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        #initialize weights with random weights
        W_bound=np.sqrt(6./(fan_in+fan_out))
        self.W=theano.shared(np.asarray(rng.uniform(
                                        low=-W_bound,high=W_bound,size=filter_shape),
                                        dtype=theano.config.floatX),borrow=True)
        #the bias is a 1D tensor-- one bias per output feature map
        b_values=np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b=theano.shared(value=b_values,borrow=True)
        
        #convolve input feature maps with filters
        conv_out=conv.conv2d(input=input,filters=self.W,filter_shape=filter_shape,image_shape=image_shape)
                             
        #pooling each feature map individually,using maxpooling
        pooled_out=downsample.max_pool_2d(input=conv_out,ds=poolsize,ignore_border=True)
        
        #add the bias term.Since the bias is a vector (1D array),we first 
        #reshape it to a tensor of shape(1,n_filters,1,1).Each bias will thus
        #be broadcasted across mini-batches and feature map width& height
        self.output=T.tanh(pooled_out+self.b.dimshuffle('x',0,'x','x'))
        
        #store parameters of this layer
        self.params=[self.W,self.b]
        
    def return_output():
        return self.output
        
#######################
#  Construct LetNet5  #
#######################
def train_lenet():
    learning_rate=0.001
    #if not using RMSprop learn_rate=0.1 
    #if using RMSprop learn_rate=0.001
    nkerns=[20,50]
    batch_size=500
    """
    :type nkerns:list of ints
    :param nkerns: nkerns[0] is the number of feature maps after 1 LeCovPoollayer
                   nkerns[1] is the number of feature maps after 2 LeCovPoolllayer
    """  
    rng=np.random.RandomState(1234567890)
    datasets=load_data(path)
    train_set_x,train_set_y=datasets[0]
    valid_set_x,valid_set_y=datasets[1]
    test_set_x=datasets[2]
   
    #compute number of minibatches
    n_train_batches=train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches=test_set_x.get_value(borrow=True).shape[0]
    n_train_batches/=batch_size
    n_valid_batches/=batch_size
    n_test_batches/=batch_size
    
    #allocat symbolic variables for the data
    index=T.lscalar() #index to minibatch
    x=T.matrix('x') #image
    y=T.ivector('y') #labels  
    ishape=(28,28) #MNIST image size
    p_drop_cov=0.2
    p_drop_perceptron=0.2
    p_drop_logistic=0.2 #probablities of drop-out to prevent overfitting
    
    #########################
    # Building actual model # 
    #########################
    print '...building the model'
    
    #reshape matrix of images of shape (batch_size,28,28)
    #to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input=x.reshape((batch_size,1,28,28)) #batch_size* 1 channel* (28*28)size
    
    #Construct the first convolutional pooling layer:
    #filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    #maxpooling reduces this further to (24/2,24/2)=(12,12)
    #4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0=LeNetConvPoolLayer(rng,input=layer0_input,
                               image_shape=(batch_size,1,28,28),
                               filter_shape=(nkerns[0],1,5,5),poolsize=(2,2),p_drop_cov=p_drop_cov)
    
    # Construct the second convolutional pooling layer:
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (batch_size,nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2),p_drop_cov=p_drop_cov)
    
    #the HiddenLayer being fully-connected,it operates on 2D matrices shape
    #(batch_size,num_pixels)
    #This will generate a matrix of shape (batch_size,nkerns[1]*4*4)=(500,800)
    layer2_input=layer1.output.flatten(2)
    
    #construct a fully-connected perceptron layer
    layer2=HiddenLayer(rng,input=layer2_input,n_in=nkerns[1]*4*4,n_out=500,activation=T.tanh,p_drop_perceptron=p_drop_perceptron)
    
    #classify the values of the fully connected softmax layer
    layer3=LogisticRegression(input=layer2.output,n_in=500,n_out=10,p_drop_logistic=p_drop_logistic)
    
    #the cost we minimize during training
    cost=layer3.negative_log_likelihood(y)
    
    #create a function to compute the error rate on validation set
    validate_model=theano.function([index],layer3.errors(y),
                givens={x:valid_set_x[index*batch_size:(index+1)*batch_size],
                        y:valid_set_y[index*batch_size:(index+1)*batch_size]})
    
    #create a list of gradients for all model parameters
    params=layer3.params+layer2.params+layer1.params+layer0.params
    grads=T.grad(cost,params)
    
    #using RMSprop(scaling the gradient based on running average)
    #to update the parameters of the model as a list of (variable,update expression) pairs
    def RMSprop(gparams,params,learning_rate,rho=0.9,epsilon=1e-6):
        """
        param:rho,the fraction we keep the previous gradient contribution
        """
        updates=[]
        for p,g in zip(params,gparams):
            acc=theano.shared(p.get_value()*0.)
            acc_new=rho*acc+(1-rho)*g**2
            gradient_scaling=T.sqrt(acc_new+epsilon)
            g=g/gradient_scaling
            updates.append((acc,acc_new))
            updates.append((p,p-learning_rate*g))
        return updates
                            
    #iterate to get the optimal solution using minibatch SGD
    #updates=[]
    #for param,grad in zip(params,grads):
        #updates.append((param,param-learning_rate*grad))
    
    train_model=theano.function([index],cost,updates=RMSprop(grads,params,learning_rate),
                givens={ x:train_set_x[index* batch_size:(index+1)*batch_size],
                         y:train_set_y[index* batch_size:(index+1)*batch_size]
                         }) 
    
    #########################
    # Training the  model   # 
    ######################### 
    n_epochs=100
    print '...training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                            # found
    
    #improvement_threshold = 0.995  # a relative improvement of this much is
                                    # considered significant
    # if have added drop-out noise,we can increase the value
    improvement_threshold = 1.3
    validation_frequency = min(n_train_batches, patience / 2)
                                    # go through this many
                                    # minibatche before checking the network
                                    # on the validation set; in this case we
                                    # check every epoch
    
    best_validation_loss = np.inf
    best_iter = 0
    
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
    
            iter = (epoch - 1) * n_train_batches + minibatch_index
    
            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
    
            if (iter + 1) % validation_frequency == 0:
    
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                        in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, validation error %f %%' % \
                        (epoch, this_validation_loss * 100.))
    
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
    
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
    
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
    
            if patience <= iter:
                done_looping = True
                break
    ###########################################
    # Predict with trained parameters(nonoise)#
    ###########################################
    layer0.p_drop_cov=0
    layer1.p_drop_cov=0
    layer2.p_drop_perceptron=0
    layer3.p_drop_logistic=0 #when predicting set drop-out be zeros
    model_predict=theano.function(inputs=[index],outputs=layer3.y_pred,givens={x:test_set_x[index*batch_size:(index+1)*batch_size]})
    digit_preds=Series(np.concatenate([model_predict(i) for i in xrange(n_test_batches)]))
    image_ids=Series(np.arange(1,len(digit_preds)+1))
    submission=DataFrame([image_ids,digit_preds]).T
    submission.columns=['ImageId','Label']
    submission.to_csv(path+'submission_cnn.csv',index=False)                                 
    
if __name__=='__main__':
    debug_mode=False
    train_lenet()