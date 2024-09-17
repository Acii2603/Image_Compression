# Image Compression Algorithms
**Author**: Alexandru-Andrei Ionita    
**Year**: 2022-2023  
**Source Files**: `main.c`, `apply.h`, `case.h`, `crop.h`, `exitfile.h`, `histogram.h`, `load.h`, `save.h`, `select.h`, `struct.h`

## 1. **Image Compression Using Singular Value Decomposition (SVD)**

### Project Overview:
In this project, I explored image compression techniques using Singular Value Decomposition (SVD). The aim was to reduce the size of an image while retaining its essential features. By applying SVD, I could decompose the image matrix into three matrices and selectively keep only the most significant singular values. This allowed me to compress the image while preserving its core details.

SVD works by factorizing the matrix \( A \) (representing the image) into \( U \Sigma V^T \), where \( \Sigma \) contains the singular values that represent the image's important features. By truncating the matrix and retaining only the top \( k \) singular values, I effectively reduced the size of the image. The result is a visually similar image stored in a more compact form.

### Key Features:
- **Matrix Factorization**: Decomposed the image matrix using SVD.
- **Compression**: Kept only the top \( k \) singular values to approximate the original image.
- **Matrix Operations**: Efficient SVD implementation to achieve substantial compression while retaining image quality.

### Outcome:
This project successfully demonstrated how to use SVD for compressing images while maintaining their visual integrity. The level of compression can be controlled by adjusting \( k \), the number of singular values retained.

---

## 2. **Image Compression Using Principal Component Analysis (PCA)**

### Project Overview:
I applied Principal Component Analysis (PCA) for image compression, focusing on reducing the dimensionality of the image data. PCA transforms the image into a lower-dimensional space, capturing the directions of maximum variance, known as principal components. By retaining only the most significant components, I was able to compress the image while preserving its key features.

The process involved projecting the original image onto the principal components, reducing the dimensionality and storage requirements. The reduced data was then used to approximate the original image, achieving effective compression without losing much detail.

### Key Features:
- **Dimensionality Reduction**: Compressed the image by projecting it onto principal components.
- **Variance Maximization**: Retained principal components that captured the highest variance in the data.
- **Image Approximation**: Reconstructed the image from the reduced components, balancing quality and size.

### Outcome:
This project highlighted the power of PCA for dimensionality reduction and image compression. By selecting the number of principal components, I could control the trade-off between image quality and compression ratio.

---

## 3. **Image Compression Using Covariance Matrix and PCA**

### Project Overview:
Building on the previous PCA project, I explored another approach to image compression using the covariance matrix. This method calculates the covariance between pixels in the image and uses eigenvalue decomposition to find the principal components. By projecting the image data onto these components, I achieved efficient compression.

The covariance matrix approach offers a different perspective on PCA, focusing on the relationships between pixel intensities. After calculating the eigenvalues and eigenvectors of the covariance matrix, I retained only the top components to compress the image while preserving its most significant features.

### Key Features:
- **Covariance Matrix Calculation**: Computed the covariance matrix to capture relationships between pixels.
- **Eigenvalue Decomposition**: Used eigenvalue decomposition to find the principal components.
- **Efficient Compression**: Reduced the dimensionality of the image data using the most significant components.

### Outcome:
This method of image compression demonstrated the effectiveness of using the covariance matrix and eigenvalue decomposition. It provided another way to balance image quality and file size through PCA.

---

## 4. **Handwritten Digit Recognition Using PCA and K-Nearest Neighbors**

### Project Overview:
This project focused on handwritten digit recognition using a combination of PCA for dimensionality reduction and K-Nearest Neighbors (KNN) for classification. I worked with the MNIST dataset, which contains thousands of handwritten digits. Using PCA, I reduced the dimensionality of each 28x28 pixel image to just 23 principal components, making the classification process more efficient.

With the reduced data, I applied KNN to classify the digits. The KNN algorithm determines the nearest neighbors to a given input and predicts the digit based on the majority class among the neighbors. By combining PCA and KNN, I achieved a highly accurate recognition system with optimized computational requirements.

### Key Features:
- **Dimensionality Reduction with PCA**: Reduced the size of each image by transforming it into a lower-dimensional space using principal components.
- **K-Nearest Neighbors Classification**: Implemented KNN to classify the digits based on their reduced representation.
- **Efficient Processing**: Significantly reduced the processing time by working with fewer dimensions while maintaining high accuracy.

### Outcome:
The system achieved excellent results, accurately recognizing handwritten digits with minimal computational overhead. By combining PCA and KNN, I was able to create a highly efficient and accurate digit recognition system.

---

## 5. **Observations and Reflections**

Throughout these projects, I observed the trade-offs between compression and image quality when varying the number of components or singular values. With SVD and PCA, it's possible to compress images to a fraction of their original size while retaining key details, but going too far in reducing the number of components can lead to noticeable quality loss. In the case of digit recognition, dimensionality reduction through PCA not only reduced computation time but also preserved the important features needed for classification.

Each method offered unique insights:
- SVD was highly effective for general image compression.
- PCA and covariance-based PCA excelled at reducing dimensionality while maintaining significant features.
- The PCA-KNN approach for digit recognition struck a great balance between accuracy and efficiency, making it an ideal choice for large datasets like MNIST.
