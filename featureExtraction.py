

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets


# ⁡⁢⁣⁢Reducing Features Using Principal Components⁡

digits = datasets.load_digits()


# ⁡⁢⁣⁣standarize the feature matrix ⁡
features = StandardScaler().fit_transform(digits.data)

#or 
# feature = StandardScaler()
# features = feature.fit_transform(digits.data)

# ⁡⁢⁣⁣create a pca that will retain 99% of variance ⁡
pca = PCA(n_components=0.99,whiten=True)

# ⁡⁢⁣⁣conduct pca⁡
features_pca = pca.fit_transform(features)
# print("original :", features.shape[1])
# print("now: ",features_pca.shape[1])

#⁡⁢⁣⁣output⁡


# original : 64
# now:  54

#  ⁡⁢⁣⁢─── Reducing Features When Data Is Linearly Inseparable ──────────────────────⁡

from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

# ⁡⁢⁣⁣create linearly inseparable data⁡

features1,_ = make_circles(n_samples=1000, random_state=1,noise=0.1,factor=0.1)
kpca = KernelPCA(kernel="rbf",gamma=15,n_components=1)
features_kpca= kpca.fit_transform(features1)

# print("hello")
# print("original :", features1.shape[1])
# print("now: ",features_kpca.shape[1])


# ⁡⁢⁣⁢reducting features by maximizing class separability⁡


from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
features2= iris.data
target2 = iris.target
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features2,target2).transform(features2)

# print("original number of features: ",features2.shape[1])
# print("reduced :",features_lda.shape[1])


# ⁡⁢⁡⁢⁣⁢Reducing features using matrix factorization⁡ 

from sklearn.decomposition import NMF
digits1 = datasets.load_digits()
features3 = digits.data

nmf = NMF(n_components=10,random_state=1)
features_nmf = nmf.fit_transform(features3) #nmf -> non negative matrix factorization 
print("original: ", features3.shape[1])
print("reduced :",features_nmf.shape[1])
