
# coding: utf-8

# # Helper Functions and Training Procedure:

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from skimage.feature import hog
import itertools


# In[13]:


from sklearn import svm


# In[17]:


from sklearn import neighbors


# In[3]:


def generate_hog(X, cell_size=(8,8)):
    hog_img=[]
    for img in X:
        vec=hog(img, pixels_per_cell=cell_size)
        hog_img.append(vec)
    return np.asarray(hog_img)


# In[4]:


def Sigmoid(x):
    return np.asarray(1/(1+np.exp(-x)))

def ReLU(x):
    return np.asarray(np.maximum(0,x))

def Softmax(x):
    b=np.exp(x)
    c=np.sum(b, axis=0)
    d=np.divide(b, c)
    return d

def ReLU_grad(x):
    return np.heaviside(x,0)

def Sigmoid_grad(x):
    return np.asarray(np.multiply(Sigmoid(x),1-Sigmoid(x)))

def CrossEntropy(target_, output_):
    return np.sum(-np.sum(np.multiply(target_, np.log((output_))), axis=0))

def CrossEntropy_grad(target_, output_):
    return -np.divide(target_, output_)

def Softmax_CE_grad(target_, pred_):
    grad=-pred_+target_
    return grad


# In[5]:


def add_noise(images):
    size=images.shape
    x=np.random.normal(loc=0.0, scale=10, size=size)
    noisy=np.add(images,x)
    return noisy

def labels_to_class(labels):
    return np.argmax(labels, axis=1)

def confusion_matrix(target_, pred):
    size=len(target_[0])
    target_class=labels_to_class(target_)
    pred_class=labels_to_class(pred)
    cm=np.zeros([size, size])
    for a,p in zip(target_class, pred_class):
        cm[a][p]+=1
    return cm

def cm_metrics(cm):
    diag=(np.diagonal(cm))
    psum=np.sum(cm, axis=0, dtype=np.float32)
    rsum=np.sum(cm, axis=1, dtype=np.float32)
    p=np.divide(diag, psum)
    r=np.divide(diag, rsum)
    prod=np.multiply(p,r)
    sum_=p+r
    f=2*np.divide(prod,sum_)
    a=1.*np.sum(diag)/np.sum(cm)
    return (a,p,r,f)
    
def plot_confusion_matrix(cm, target_names, title='Confusion matrix',):

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def plot_loss(train_loss, test_loss, title):
    x=200*np.arange(0, len(train_loss))
    plt.plot(x,train_loss, label='Train')
    plt.plot(x,test_loss, label='Test')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Average Loss")
    plt.legend(loc='upper left')
    plt.show()
    
def top_k_pred(pred, k):
    sort=np.argsort(pred, axis=0)
    return sort[::-1][:k]


# In[6]:


def shuffle_data(images, labels):
    index=np.random.permutation(len(images))
    shuff_images, shuff_labels=images[index], labels[index]
    return shuff_images, shuff_labels

def get_split_mask(images):
    size=len(images)/5
    a=np.ones(5*size, dtype=bool)
    b=np.arange(10)
    mask=[]
    for i in range(5):
        mask.append([False if (j<(i+1)*size)&(j>=(i)*size) else x for j,x in enumerate(a)])
    return mask

def get_split_data(mask, images, labels, index):
    mask_=mask[index]
    train_images=images[mask_]
    train_labels=labels[mask_]
    inv_mask_=np.invert(mask_)
    test_images=images[inv_mask_]
    test_labels=labels[inv_mask_]
    return train_images, train_labels, test_images, test_labels

def get_accuracy(model, test_images, test_labels):
    batch_size=len(test_images[0])
    count=np.zeros(batch_size)
    pred=model.forward(test_images)
    count=[1 if np.argmax(pred[:,i], axis=0)==np.argmax(test_labels[:,i], axis=0) else 0 for i in range(batch_size)]
    correct=np.sum(count)
    accuracy=100*correct/batch_size
    return(pred, accuracy)

def SGD_mom(model, batch_images, batch_labels, l2):
    batch_size=len(batch_images[0])
    loss=model.update(batch_images, batch_labels, l2)
    return loss
    
def one_hot(labels):
    a=np.zeros((len(labels), 10))
    a[np.arange(len(labels)),labels]=1
    return a


# In[7]:


class MLP(object):
    def __init__(self, input_size, h1_size, h2_size, output_size):
        self.W1=np.random.normal(loc=0.0, scale=0.08, size=(h1_size, input_size) )
        self.W2=np.random.normal(loc=0.0, scale=0.08, size=(h2_size, h1_size) )
        self.W3=np.random.normal(loc=0.0, scale=0.08, size=(output_size, h2_size) )
        self.B1=np.zeros(h1_size).reshape(-1,1)
        self.B2=np.zeros(h2_size).reshape(-1,1)
        self.B3=np.zeros(output_size).reshape(-1,1)
        
        self.W1_grad=np.zeros_like(self.W1)
        self.W2_grad=np.zeros_like(self.W2)
        self.W3_grad=np.zeros_like(self.W3)
        self.B1_grad=np.zeros_like(self.B1)
        self.B2_grad=np.zeros_like(self.B2)
        self.B3_grad=np.zeros_like(self.B3)
        
        self.W1_mom=np.zeros_like(self.W1)
        self.W2_mom=np.zeros_like(self.W2)
        self.W3_mom=np.zeros_like(self.W3)
        self.B1_mom=np.zeros_like(self.B1)
        self.B2_mom=np.zeros_like(self.B2)
        self.B3_mom=np.zeros_like(self.B3)
        
        
    def update(self, input_, target_, l2=0):

        batch_size=len(input_[0])
        x=input_
        h1=np.add(np.matmul(self.W1, input_), self.B1)
        a1=act[act_ind](h1)
        h2=np.add(np.matmul(self.W2, a1), self.B2)
        a2=act[act_ind](h2)
        h3=np.add(np.matmul(self.W3, a2), self.B3)
        a3=Softmax(h3)
        
        loss=CrossEntropy(target_, a3)
        _E_h3=Softmax_CE_grad(a3, target_)
        
        _a2_h2=act_grad[act_ind](h2)
        _a1_h1=act_grad[act_ind](h1)
        
        _E_W3=np.matmul(_E_h3,np.transpose(a2))
        _E_B3=np.sum(_E_h3, axis=1).reshape(-1,1)
        _E_a2=np.matmul(np.transpose(self.W3), _E_h3)
        
        _E_h2=np.multiply(_E_a2, _a2_h2)
        _E_W2=np.matmul(_E_h2, np.transpose(a1))
        _E_B2=np.sum(_E_h2, axis=1).reshape(-1,1)
        _E_a1=np.matmul(np.transpose(self.W2), _E_h2)
        
        _E_h1=np.multiply(_E_a1, _a1_h1)
        _E_W1=np.matmul(_E_h1, np.transpose(x))
        _E_B1=np.sum(_E_h1, axis=1).reshape(-1,1)
        _E_x=np.matmul(np.transpose(self.W1), _E_h1)
        
        self.W1_grad=_E_W1/batch_size+self.W1*2*l2
        self.W2_grad=_E_W2/batch_size+self.W2*2*l2
        self.W3_grad=_E_W3/batch_size+self.W3*2*l2
        self.B1_grad=_E_B1/batch_size
        self.B2_grad=_E_B2/batch_size
        self.B3_grad=_E_B3/batch_size
                
        self.W1_mom=gamma*self.W1_mom+lr*self.W1_grad
        self.W2_mom=gamma*self.W2_mom+lr*self.W2_grad
        self.W3_mom=gamma*self.W3_mom+lr*self.W3_grad
        self.B1_mom=gamma*self.B1_mom+lr*self.B1_grad
        self.B2_mom=gamma*self.B2_mom+lr*self.B2_grad
        self.B3_mom=gamma*self.B3_mom+lr*self.B3_grad
        
        self.W1-=self.W1_mom
        self.W2-=self.W2_mom
        self.W3-=self.W3_mom
        self.B1=self.B1-self.B1_mom
        self.B2=self.B2-self.B2_mom
        self.B3=self.B3-self.B3_mom

        return loss
    
    def forward(self, input_):
        x=input_
        h1=np.add(np.matmul(self.W1, input_), self.B1)
        a1=act[act_ind](h1)
        h2=np.add(np.matmul(self.W2, a1), self.B2)
        a2=act[act_ind](h2)
        h3=np.add(np.matmul(self.W3, a2), self.B3)
        a3=Softmax(h3)
        return a3


# In[28]:


input_size=81
h1_size=500
h2_size=250
output_size=10
gamma=0.99
lr=1e-3
act_ind=0
batch_size=64
epochs=10
act=[Sigmoid, ReLU]
act_grad=[Sigmoid_grad, ReLU_grad]
actstr={0:'Sigmoid',1:'ReLU'}


# In[9]:


from mnist import MNIST
data=MNIST('/home/pradeep/data/')
images, labels_ = data.load_training()

images=np.asarray(images)
labels_=np.asarray(labels_)
images=images.reshape(-1,28,28)

act=[Sigmoid, ReLU]
act_grad=[Sigmoid_grad, ReLU_grad]

images, labels_=shuffle_data(images, labels_)
labels=one_hot(labels_)
mask=get_split_mask(images)
hog_img=generate_hog(X=images)
hog_img*=255


# In[27]:


def train(model, epochs, images, labels, fold_index, l2=0):
    train_loss=[]
    test_loss=[]
    train_images, train_labels, test_images, test_labels=get_split_data(mask, images, labels, fold_index)
    num_batches=len(train_images)/batch_size
    batch_images=np.array_split(train_images, num_batches)
    batch_labels=np.array_split(train_labels, num_batches)
    for epoch in range(epochs):
        for i in range(num_batches):
            size=len(batch_images[i])
            loss=SGD_mom(model, np.transpose(batch_images[i]), np.transpose(batch_labels[i]), l2=l2)
            train_avg_loss=loss/size
            if((i)%200==0):
                print("Epoch "+str(epoch+1)+" Iteration "+str(i+1)+" : Avg Loss = "+str(train_avg_loss))
                pred, accuracy=get_accuracy(model, np.transpose(test_images), np.transpose(test_labels))
                test_avg_loss= CrossEntropy(np.transpose(test_labels), pred)/len(test_images)
                train_loss.append(train_avg_loss)
                test_loss.append(test_avg_loss)
    plot_loss(train_loss, test_loss, "Plot of loss for "+actstr[act_ind]+" for Fold "+str(fold_index+1))
    output_cm_scores(model, test_images, test_labels)
    #plot_images(model, test_images[:20], 3)


# In[10]:


def output_cm_scores_(pred, labels):
    cm=confusion_matrix_(pred=pred, target_=labels)
    a,p,r,f=cm_metrics(cm)
    print "Accuracy = "+ str(a)
    print "Precision = ", p
    print "Recall = ", r
    print "F1 Score = ", f
    target_names=['0','1','2','3','4','5','6','7','8','9']
    plot_confusion_matrix(cm,target_names)
    
def confusion_matrix_(target_, pred):
    size=10
    cm=np.zeros([size, size])
    for a,p in zip(target_, pred):
        cm[a][p]+=1
    return cm    


# In[18]:


def train_KNN(hog_img, labels, fold_index):
    KNN=neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    mask=get_split_mask(hog_img)
    train_hog, train_labels, test_hog, test_labels=get_split_data(images=hog_img, labels=labels_, mask=mask, index=fold_index)
    KNN.fit(train_hog, train_labels)
    pred=KNN.predict(test_hog)
    output_cm_scores_(pred, np.transpose(test_labels))


# In[1]:


def train_SVM(hog_img, labels, fold_index):
    SVM=svm.LinearSVC(random_state=0)
    mask=get_split_mask(hog_img)
    train_hog, train_labels, test_hog, test_labels=get_split_data(images=hog_img, labels=labels_, mask=mask, index=fold_index)
    SVM.fit(train_hog, train_labels)
    pred=SVM.predict(test_hog)
    output_cm_scores_(pred,np.transpose(test_labels))


# # Training Results:
# 

# ### Sigmoid Activation: Fold - 1 , Learning Rate=1e-3

# In[71]:


model0=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=0, model=model0)


# ### Sigmoid Activation: Fold - 2 , Learning Rate=1e-3

# In[73]:


model1=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=1, model=model1)


# ### Sigmoid Activation: Fold - 3 , Learning Rate=1e-3

# In[74]:


model2=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=2, model=model2)


# ### Sigmoid Activation: Fold - 4 , Learning Rate=1e-3

# In[75]:


model3=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=3, model=model3)


# ### ReLU Activation: Fold - 1 , Learning Rate=5e-4

# In[78]:


model0=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=0, model=model0)


# ### ReLU Activation: Fold - 2 , Learning Rate=5e-4

# In[79]:


model1=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=1, model=model1)


# ### ReLU Activation: Fold - 3 , Learning Rate=5e-4

# In[80]:


model2=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=2, model=model2)


# ### ReLU Activation: Fold - 4 , Learning Rate=5e-4

# In[81]:


model3=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=3, model=model3)


# ### ReLU Activation: Fold - 5 , Learning Rate=5e-4

# In[82]:


model4=MLP(input_size, h1_size, h2_size, output_size)
train(images=hog_img, labels=labels, epochs=10, fold_index=4, model=model4)


# ### KNN: Fold-1

# In[117]:


train_KNN(hog_img, labels, fold_index=0)


# ### KNN: Fold-2

# In[120]:


train_KNN(hog_img, labels, fold_index=1)


# ### KNN: Fold-3

# In[19]:


train_KNN(hog_img, labels, fold_index=2)


# ### KNN: Fold-4

# In[21]:


train_KNN(hog_img, labels, fold_index=3)


# ### KNN: Fold-5

# In[23]:


train_KNN(hog_img, labels, fold_index=4)


# ### SVM: Fold-1

# In[14]:


train_SVM(hog_img, labels, 0)


# ### SVM: Fold-2

# In[15]:


train_SVM(hog_img, labels, 1)


# ### SVM: Fold-3

# In[20]:


train_SVM(hog_img, labels, 2)


# ### SVM: Fold-4

# In[22]:


train_SVM(hog_img, labels, 3)


# ### SVM: Fold-5

# In[24]:


train_SVM(hog_img, labels, 4)

