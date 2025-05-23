#LIST OF FUNCTIONS USED


#1 ----- matrix_sparse = sparse.csr_matrix(matrix)

#2 ----- matrix3.shape



# print(np.max(matrix4))
# print(np.min(matrix4))
# print(np.max(matrix4, axis=0)) # max element in each column
# print(np.max(matrix4,axis=1)) # max element in each row , both of them return a 1d array



# print(np.mean(matrix4))
# print(np.var(matrix4))
# print(np.std(matrix4))


# print(matrix7.reshape(2,6))  # reshape into a 2x6 matrix


# print(matrix7.T)


#print(matrix9.flatten())  # flattens into a 1x1 array


#print(np.linalg.matrix_rank(matrix9))

#print(np.linalg.det(matrix9))


#print(matrix9.diagonal())
#print(matrix9.trace())   # trace is the sum of diagonals 

#eigenvalues, eigenvectors = np.linalg.eig(matrix9)
#print(np.linalg.inv(mat))



import numpy as np
from scipy import sparse
#1
matrix = np.array([[0,0],
                  [0,1],
                  [3,0]])


matrix_sparse = sparse.csr_matrix(matrix)

# print(matrix_sparse)

#2

vector = np.array([1,2,3,4,5,6])
matrix1 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

# print(vector[2])
# print(matrix1[1,1])

#3: DESCRIBE A MATRIX

matrix3 = np.array([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
])

# print(matrix3.shape) # number of rows and columns 
# print(matrix3.size) # no of elements
# print(matrix3.ndim) # number of dimensions


#4: APPLYING OPERATIONS TO ELEMENTS
matrix4 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
#print(matrix4+100) 


#5: FIND MAX, MIN VALUES

# print(np.max(matrix4))
# print(np.min(matrix4))
# print(np.max(matrix4, axis=0)) # max element in each column
# print(np.max(matrix4,axis=1)) # max element in each row , both of them return a 1d array


#6: average, variance, standard deviation

# print(np.mean(matrix4))
# print(np.var(matrix4))
#print(np.std(matrix4))


#7: reshapping array


matrix7 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12]
])

# print(matrix7.reshape(2,6))  # reshape into a 2x6 matrix


#8: TRANSPOSING A VECTOR OR MATRIX

# print(matrix7.T)  # for transpose matrix 
# print(np.array([1,2,3,4,5]).T)  # for transpose vector 

# print(np.array([[1,2,3,4,5]]).T) # transpose of a 1x1 matrix 


#9: FLATTENING A MATRIX 

matrix9 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

#print(matrix9.flatten())  # flattens into a 1x1 array


#10: FINDING RANK OF A MATRIX

#print(np.linalg.matrix_rank(matrix9))

#11: CALCULATE DETERMINANT

#print(np.linalg.det(matrix9))

#12: GETTING DIAGONAL,TRACE OF MATRIX

#print(matrix9.diagonal())
#print(matrix9.trace())   # trace is the sum of diagonals 

#13: FINDING EIGENVALUES AND EIGENVECTORS

eigenvalues, eigenvectors = np.linalg.eig(matrix9)

#print(eigenvalues,eigenvectors)

#14: DOT PRODUCT, ADD, SUBSTRACT

a = np.array([1,2,3])
b = np.array([4,5,6])

c= np.dot(a,b)
d = np.add(a,b)
e = np.subtract(a,b)
#print(e)

# works with both matrices and vectors


#15: INVERTING A MATRIX 

mat = np.array([
    [1,4],\
    [2,5]
])
print(np.linalg.inv(mat))
