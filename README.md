# Comprehensive-pure-Java-matrix-classes-for-Computer-Vision-Machine-Learning-and-AI-applications
Comprehensive pure Java matrix classes especially for Computer Vision, Machine Learning and AI applications 

These Java matrix classes supports 2D and 3D matrices (a 2D matrix is treated simply as a 3D matrix with the number of channels equal to one). The underlying data is backed by a (single dimensional) double array. Both continuous and discontinuous slicing is allowed.  Continuous slicing is very efficient and has the time complexity of O(1) (for Matk class) and just the indices of the slice are recorded in the object (making it into a view), i.e. no data is copied. All operations can be done on a view, just like a normal matrix. For, Matkc class, data is copied regardless of continuous or discontinuous slicing. 

The functionalities implemented in the classes include:

- Creating a matrix of a given number of rows, columns and channels from a given Java double, float, integer or byte array (stored either in row or column major order).
- Creating a column matrix from a given double, float, integer or byte array.
- Creating a matrix of a given number of rows, columns and channels, filled with zeros.
- Creating a matrix of a given number of rows and columns, filled with zeros.
- Creating a matrix from a 2D or 3D Java array of different types.
- Creating a matrix from a Java List of numbers of different types.
- Creating a matrix from "Apache Common Math RealMatrix".
- Creating a matrix from BufferedImage object.
- Constructing/loading a matrix from a file.
- Saving the matrix to a file.
- Converting the matrix to a BufferedImage object of different underlying types.
- Converting the matrix to 1D, 2D or 3D Java arrays of different types.
- Converting the matrix to "Apache Common Math RealMatrix".
- Reading from a file (which has been saved from a matrix) to construct a BufferedImage.
- Showing or visualizing the matrix as an image on a GUI window.
- Creating a matrix of a given number of rows, columns and channels filled with values drawn from different random distributions such as uniformly distributed pseudorandom integers between some range, and normally distributed random numbers.
- Creating a matrix of a given number rows, columns and channels filled with values starting from some number with a given increment.
- Creating a matrix similar to MATLAB's "linspace" function.
- Creating a matrix of ones, zeros, etc.
- Deep copying of this matrix to get another matrix.
- Getting the number of rows, columns and channels of this matrix.
- Getting the number of data in the matrix.
- Getting the length of the vector corresponding to this matrix (if it is a vector).
- Checking whether this matrix is a view (only in Matk class).
- Checking if this matrix is a vector, a row vector, a column vector or a channel vector.
- Continuous slicing of rows, columns, channels or any combinations of them, from this matrix.
- Discontinuous slicing of rows, columns, channels or any combinations of them, from the matrix.
- Setting the values of certain rows, columns or channels of the matrix.
- Setting a sub-matrix of this matrix to some other matrix.
- Getting or setting rows, columns and channels of this matrix.
- Vectorizing this matrix to create another matrix, or a Java array of different types.
- Transposing the matrix.
- Incrementing all the values in the matrix (either in-place or by creating a new matrix).
- Decrementing all the values in the matrix (either in-place or by creating a new matrix).
- Checking if two matrices are equal or approximately equal.
- Computing the dot product of the matrix with another matrix.
- Element-wise multiplication of the matrix with another matrix.
- Multiplying the matrix with another matrix.
- Dividing the matrix with another matrix (element-wise).
- Computing the power of a given value for all elements of the matrix.
- Adding or subtracting the matrix from another matrix and vice versa.
- Generating a matrix by replicating this matrix in a block-like fashion, similar to MATLAB's "repmat" function.
- Reshaping the matrix (either O(1) operation or if requested, by copying data).
- Rounding the values in the matrix (optionally in-place).
- Setting all values of the matrix to one, zero or some number.
- Filling the matrix with uniformly distributed pseudorandom integers between a given range.
- Filling the matrix with uniformly distributed random numbers between continuous range "rangeMin" and "rangeMax" similar to MATLAB's "rand" function.
- Filling the matrix with normally distributed random numbers similar to MATLAB's "randn" function.
- Filling this matrix with values drawn from a custom random number generator.
- Sorting the values in the matrix in many different ways such as row-wise, column wise, channel wise, etc.
- Getting maximum or minimum values in the matrix from different perspectives such as taking as rows, columns and channels.
- Computing univariate moments such as mean, variance, standard deviation, geometric mean, Kurtosis, second moment, semi-variance and skewness, from different perspectives of the matrix.
- Computing summaries of the matrix with a custom function (which could be computing a summary such as mean, median, mode and product) from looking at different perspectives of the matrix.
- Summing all the elements in the matrix in some way or perspective.
- Computing the sum of the natural logs of the entries of the matrix.
- Computing the sum of the squares of the entries of the matrix.
- Computing histograms of the matrix in different ways.
- Joining or merging the matrix with another given matrix horizontally, vertically, channel-wise, etc.
- Deleting rows, columns and channels from the matrix.
- Remove a sub-matrix (continuous) from the matrix.
- Finding the locations of the elements in this matrix that satisfy a given comparison condition.
- Clustering the data in the matrix using K-means algorithm by treating the matrix as a dataset (assuming this is a 2D matrix).
- Generate a dataset that contains bivariate Gaussian data for different number of classes. For each class, one multivariate gaussian distribution is produced.

Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

Dr. Kyaw Kyaw Htike @ Ali Abdul Ghafur

https://kyaw.xyz
