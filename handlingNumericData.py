import numpy as np
from sklearn import preprocessing

#for outliers
from sklearn import covariance
from sklearn import datasets

# We use StandardScaler (or MinMaxScaler) to make sure no feature dominates just because it has bigger numbers. It makes machine learning models fairer, faster, and more accurate.


feature = np.array([
    [-500.5],
    [-100.1],
    [0],
    [100.1],
    [900.9]
])



# ─── Rescaling ────────────────────────────────────────────────────────────────


minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))   # returns an object MinMaxScale()

scaled_feature = minmax_scale.fit_transform(feature)
# Xscaled = X - Xmin / (Xmax -Xmin)

# print(minmax_scale)
# print(scaled_feature)
# ──────────────────────────────────────────────────────────────────────────────



# ─── Standardizing A Feature ──────────────────────────────────────────────────




x = np.array([
    [-1000.1],
    [-200.2],
    [500.5],
    [600.6],
    [9000.9]
])

scalar = preprocessing.StandardScaler()
standardize = scalar.fit_transform(x)
# print(standardize)


# ─── 𝘯𝘰𝘳𝘮𝘢𝘭𝘪𝘻𝘪𝘯𝘨 𝘰𝘣𝘴𝘦𝘳𝘷𝘢𝘵𝘪𝘰𝘯 ────────────────────────────

norm_features = np.array([
    [0.5, 0.5],
    [1.1, 3.4],
    [1.5, 20.2],
    [1.63, 34.4],
    [10.9, 3.3]
])

normalizer = preprocessing.Normalizer(norm="l1")
norm = normalizer.fit_transform(norm_features)
# print(norm)



# ─── Generating Polynomial And Interaction Feature ────────────────────────────

poly_features = np.array([
    [2,3],
    [2,3],
    [2,3]
])
polynomial_interaction = preprocessing.PolynomialFeatures(degree=2,include_bias=False)
pol = polynomial_interaction.fit_transform(poly_features)
#print(pol)    
#[x₁, x₂, x₁², x₁·x₂, x₂²]



# ─── Transforming Features ────────────────────────────────────────────────────


func_features = np.array([
    [2,3],
    [2,3]
])

def add_ten(x):
    return x+10

ten_transformer = preprocessing.FunctionTransformer(add_ten)
func = ten_transformer.fit_transform(func_features)

#print(func)
#[[12 13]
#[12 13]]


# ─── Detecting Outliers ───────────────────────────────────────────────────────

features, _ = datasets.make_blobs(n_samples=10, n_features=2,centers=1,random_state=1)


# replace first onservation's values with extreme values
features[0,0] = 10000
#features[0,1] =10000
#create detector 
outliers_detector = covariance.EllipticEnvelope(contamination=0.1)


print(features)


outliers_detector.fit(features)
# print(outliers_detector.predict(features))


# ─── Discretization Features ──────────────────────────────────────────────────

age = np.array([
    [6],
    [12],
    [20],
    [36],
    [65]
])

binarizer = preprocessing.Binarizer(threshold=18)

bina = binarizer.fit_transform(age)

print(bina)

# OR

print(np.digitize(age,bins=[20,30,60]))