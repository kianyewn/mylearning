import numpy as np

X = np.random.randn(4, 4)
np.linalg.norm(X, axis=1).shape
X / np.linalg.norm(X, axis=1).reshape(-1,1)

def cosine(x1, x2):
    # (M,N), (M, N)
    x1 = x1 / np.linalg.norm(x1, axis=1).reshape(-1,1)
    x2 = x2 / np.linalg.norm(x2, axis=1).reshape(-1,1)
    similarity = x1.dot(x2.T)
    return similarity

def index_top_similar(index, similarity_matrix):
    similarity = np.argsort(similarity_matrix[index,:])[::-1]
    # # exclude self as similar
    # similarity_matrix = similarity[1:]
    return similarity
 
def majority_vote(y):
    majority_label = None
    max_count = 0
    unique_yi = np.unique(y)
    for yi in unique_yi:
        counts = (y==yi).sum()
        if counts > max_count:
            max_count = counts
            majority_label = yi
    return majority_label

y = [1,1,0,0,0]
majority_vote(y)
x1 = np.random.randn(10, 4)
x2 = np.random.rand(10, 4)    
similarity = cosine(x1, x2)

index_top_similar(index=0, similarity_matrix=similarity)

from sklearn.metrics.pairwise import cosine_similarity
sl_similarity = cosine_similarity(x1,x2)
assert np.allclose(similarity, sl_similarity)

X = np.random.randn(100,4)
y = np.random.choice([0,1], size=(100,))

similarity_matrix = cosine(X, X)
k = 10
predictions = []
for index,features in enumerate(X):
    top_k_similar_index = index_top_similar(index=index, similarity_matrix=similarity_matrix)[:k]
    y_k = y[top_k_similar_index]
    pred = majority_vote(y_k) 
    predictions.append(pred)

from sklearn import datasets    
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = datasets.load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


similarity_matrix = cosine(X_test, X_train)
k = 1
y_pred = []
for index,features in enumerate(X_test):
    top_k_similar_index = index_top_similar(index=index, similarity_matrix=similarity_matrix)[:k]
    y_k = y_train[top_k_similar_index]
    pred = majority_vote(y_k) 
    y_pred.append(pred)
    
accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy:", accuracy)