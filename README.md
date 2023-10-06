# Neural-Style-Transfer from scratch!
Neural Style Transfer (NST) is one of the most fun and interesting optimization techniques in deep learning. It merges two images, namely: a <strong>"content" image (C)</strong> and a <strong>"style" image (S)</strong>, to create a <strong>"generated" image (G)</strong>. The generated image G combines the "content" of the image C with the "style" of image S. 

I have combined the **Taj Mahal** in Agra, India (content image C) with an image of candy (style image S) to generate the following image:
![TAJxCANDY](https://user-images.githubusercontent.com/120567183/228225546-ad2c3553-e19a-472e-904d-7c276b813ff3.png)

**Note:** I have used Adam optimizer instead of "LBFGS" optimizer because of low compute power and also rescaled images to (300,300) which further degraded their quality. However, if these images are run on powerful GPU/TPU with more number of epochs then results can be very pleasing! At the end, I have attached some images that were run for 100,000 epochs using lbfgs optimizer on powerful GPUs.

Some more examples-
![download](https://user-images.githubusercontent.com/120567183/228230642-01007a4e-8647-4ac7-8630-ae65d52c257a.png)
![download (1)](https://user-images.githubusercontent.com/120567183/228230816-4653547a-8ba0-49bd-99f7-1978a7c7871d.png)
![download (2)](https://user-images.githubusercontent.com/120567183/228231025-a219eeb4-9a80-4567-931d-857abd0b47b6.png)


I have implemented Neural Style Transfer (NST) algorithm in three steps:

- First, I build the content cost function $J_{content}(C,G)$
- Second, I build the style cost function $J_{style}(S,G)$
- Finally, I have put it all together to get $J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$.

$J_{content}(C,G)$ has been defined as - 

$$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2\tag{1} $$

* Here, $n_H, n_W$ and $n_C$ are the height, width and number of channels of the hidden layer you have chosen, and appear in a normalization term in the cost. 
* For clarity, note that $a^{(C)}$ and $a^{(G)}$ are the 3D volumes corresponding to a hidden layer's activations. 
* In order to compute the cost $J_{content}(C,G)$, it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below.

![NST_LOSS](https://user-images.githubusercontent.com/120567183/228228449-e8ee2aef-e0c9-44f6-85e0-047b0f700341.png)
(Image source -- Deeplearning.ai)
#### Gram matrix
* The style matrix is also called a "Gram matrix." 
* In linear algebra, the Gram matrix G of a set of vectors $(v_{1},\dots ,v_{n})$ is the matrix of dot products, whose entries are ${\displaystyle G_{ij} = v_{i}^T v_{j} = np.dot(v_{i}, v_{j})  }$. 
* In other words, $G_{ij}$ compares how similar $v_i$ is to $v_j$: If they are highly similar, you would expect them to have a large dot product, and thus for $G_{ij}$ to be large. 

I have computed the Style matrix by multiplying the "unrolled" filter matrix with its transpose:

![NST_GM](https://user-images.githubusercontent.com/120567183/228229324-1559411f-3144-4fb9-9ba3-8710c9b4ab1b.png)
(Image source -- Deeplearning.ai)

$$\mathbf{G}_{gram} = \mathbf{A}_{unrolled} \mathbf{A}_{unrolled}^T$$

#### $G_{(gram)ij}$: correlation
The result is a matrix of dimension $(n_C,n_C)$ where $n_C$ is the number of filters (channels). The value $G_{(gram)i,j}$ measures how similar the activations of filter $i$ are to the activations of filter $j$. 

#### $$G_{(gram),ii}$$: prevalence of patterns or textures
* The diagonal elements $G_{(gram)ii}$ measure how "active" a filter $i$ is. 
* For example, suppose filter $i$ is detecting vertical textures in the image. Then $G_{(gram)ii}$ measures how common  vertical textures are in the image as a whole.
* If $G_{(gram)ii}$ is large, this means that the image has a lot of vertical texture. 


By capturing the prevalence of different types of features ($G_{(gram)ii}$), as well as how much different features occur together ($G_{(gram)ij}$), the Style matrix $G_{gram}$ measures the style of an image.

* The corresponding style cost for this layer is defined as: 

$$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{(gram)i,j} - G^{(G)}_{(gram)i,j})^2\tag{2} $$

### Final Cost Function

$$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$

I have optimized this cost function using Adam Optimizer with an initial learning rate of 0.02 and ran them for 2500 epochs.

## Some Images generated after running for 100,000 epochs using LBFGS optimizer ->
![taj_mahal_ben_giles_o_lbfgs_i_content_h_500_m_vgg19_cw_100000 0_sw_30000 0_tv_1 0](https://user-images.githubusercontent.com/120567183/228233209-610f31fe-cce8-45fb-9744-de30f583ba46.jpg)
![golden_gate_vg_la_cafe_o_lbfgs_i_content_h_500_m_vgg19_cw_100000 0_sw_30000 0_tv_1 0](https://user-images.githubusercontent.com/120567183/228233250-87f588d1-c5ec-4038-9633-5ee71098f292.png)
![lion_edtaonisl_o_lbfgs_i_content_h_500_m_vgg19_cw_100000 0_sw_30000 0_tv_1 0_resized](https://user-images.githubusercontent.com/120567183/228233298-bdfd466b-5752-4c22-bfa5-dce56823d367.jpg) ![lion_vg_la_cafe_o_lbfgs_i_content_h_500_m_vgg19_cw_100000 0_sw_30000 0_tv_1 0_resized](https://user-images.githubusercontent.com/120567183/228233480-4fc69a45-2f9e-440e-8986-1dcf9eb94e6d.jpg)

(Credits for these images-> https://github.com/gordicaleksa)
