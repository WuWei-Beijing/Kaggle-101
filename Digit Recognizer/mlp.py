import theano
import theano.tensor as T
import numpy as np
from pandas import DataFrame,Series
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
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
    def __init__(self,input,n_in,n_out,p_drop_logistic):
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
        
########################################
#  The Multi-Layer Perceptron Class    #
########################################  
class MLP(object): 
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """  
    def __init__(self,rng,input,n_in,n_hidden,n_out,p_drop_perceptron=0.2,p_drop_logistic=0.2):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """
        #We are now dealing with "one hidden layer+ logistic regression" ---Old Network
        self.hiddenLayer=HiddenLayer(rng=rng,input=input,n_in=n_in,n_out=n_hidden,activation=T.tanh,p_drop_perceptron=p_drop_perceptron)
        self.logRegressionLayer=LogisticRegression(input=self.hiddenLayer.output,n_in=n_hidden,n_out=n_out,p_drop_logistic=p_drop_logistic)
        #L1 regularization
        self.L1=abs(self.hiddenLayer.W).sum()+abs(self.logRegressionLayer.W).sum()
        #L2 regularization
        self.L2_sqr=(self.hiddenLayer.W**2).sum()+(self.logRegressionLayer.W**2).sum()
        self.negative_log_likelihood=self.logRegressionLayer.negative_log_likelihood
        self.errors=self.logRegressionLayer.errors
        self.params=self.hiddenLayer.params+self.logRegressionLayer.params
        self.predict=self.logRegressionLayer.predict
        
#############################################
#  Train  a "old net" of MLP to solve MNIST #
#############################################
def train_old_net():
    learning_rate=0.001
    L1_reg=0.00
    L2_reg=0.0001
    n_epochs=100
    batch_size=20
    n_hidden=500
    datasets=load_data(path)
    train_set_x,train_set_y=datasets[0]
    valid_set_x,valid_set_y=datasets[1]
    test_set_x=datasets[2]
    
    #compute number of minibatches 
    n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
    n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]/batch_size
    n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size
    
    print '...building the model'
    index=T.lscalar()
    x=T.matrix('x')
    y=T.ivector('y')
    rng=np.random.RandomState(1234567890)
    #construct the MLP class
    #Attention!!!
    #this line to set p_drop_perceptron and p_drop_logistic
    #if set no dropout then decrease the early stop threshold 
    #improvement_threshold on line 292
    classifier=MLP(rng=rng,input=x,n_in=28*28,n_hidden=n_hidden,n_out=10,p_drop_perceptron=0,p_drop_logistic=0)
    
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost=classifier.negative_log_likelihood(y)+L1_reg*classifier.L1+L2_reg*classifier.L2_sqr
    
    #compiling a theano function that computes the mistake rate that 
    #made by the validate_set on minibatch
    validate_model=theano.function(inputs=[index],outputs=classifier.errors(y),
                                    givens={x:valid_set_x[index*batch_size:(index+1)*batch_size],
                                            y:valid_set_y[index*batch_size:(index+1)*batch_size]})
    
    #symbolicly compute the gradient of cost respect to params
    #the resulting gradient will be stored in list gparams
    gparams=[]
    for param in classifier.params:
        gparam=T.grad(cost,param)
        gparams.append(gparam)
    
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
        
        
    #compiling a Theano function 'train_model' that returns the cost
    #but in the same time updates the parameter of the model based on
    #the rules defined in 'updates'
    train_model=theano.function(inputs=[index],outputs=cost,updates=RMSprop(gparams,classifier.params,learning_rate=0.001),
                                givens={x:train_set_x[index*batch_size:(index+1)*batch_size],
                                        y:train_set_y[index*batch_size:(index+1)*batch_size]
                                        })
    ##############
    #Train Model##
    ##############
    print '...training'
    
    #early-stopping parameters
    patience=10000 #look as this many examples regardless
    patience_increase=2 #wait the iter number longer when a new best is found
    #improvement_threshold=0.995 # a relative improvement of this much on validation set 
                                # considered as not overfitting 
                                # if have added drop-out noise,we can increase the value
    improvement_threshold=0.995
    validation_frequency=min(n_train_batches,patience/2)
                                # every this much interval check on the validation set 
                                # to see if the net is overfitting.
                                # patience/2 because we want to at least check twice before getting the patience
                                # include n_train_batches to ensure we at least check on every epoch
    best_validation_error_rate=np.inf
    epoch=0
    done_looping=False
    
    while(epoch<n_epochs) and (not done_looping):
        epoch=epoch+1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost=train_model(minibatch_index)
            #iteration number
            iter=(epoch-1)*n_train_batches+minibatch_index
            if (iter+1)% validation_frequency==0:
                #validation
                validation_error_rate=[validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_error_rate=np.mean(validation_error_rate)
                print ('epoch %i,validation error %f %%'%(epoch,this_validation_error_rate*100.))
                
                #if we got the best validation score until now
                if this_validation_error_rate<best_validation_error_rate:
                    #improve the patience if error rate is good enough
                    if this_validation_error_rate<best_validation_error_rate*improvement_threshold:
                        patience=max(patience,iter*patience_increase)
                    best_validation_error_rate=this_validation_error_rate
            if patience<=iter:
                done_looping=True
                break

    ###########################################
    # Predict with trained parameters(nonoise)#
    ###########################################
    classifier.p_drop_perceptron=0
    classifier.p_drop_logistic=0
    y_x=classifier.predict()
    model_predict=theano.function(inputs=[index],outputs=y_x,givens={x:test_set_x[index*batch_size:(index+1)*batch_size]})
    digit_preds=Series(np.concatenate([model_predict(i) for i in xrange(n_test_batches)]))
    image_ids=Series(np.arange(1,len(digit_preds)+1))
    submission=DataFrame([image_ids,digit_preds]).T
    submission.columns=['ImageId','Label']
    submission.to_csv(path+'submission_sample.csv',index=False)
    
if __name__=='__main__':
    debug_mode=False
    train_old_net()