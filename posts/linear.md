---
title: "Vector Spaces and Linear Algebra"
date: 2025-02-26T18:04:10-05:00
draft: false
---

# Matrices, Vector Spaces, and Dimensions

I often forget the intuition I had for properties of vector spaces and matrices like rank, dimension, the row space and column space, rank-nullity theorem, etc. This is a refresher that I can go back to which puts it in my language. 

## Vector Spaces

I will first give the formal definition, then explain it:

> A **vector space** consists of a set $V$ (elements of $V$ are called **vectors**), a field $F$ (elements of $F$ are called **scalars**), and two operations:
>
> - **Vector addition**: An operation that takes two vectors $v, w \in V$ and produces a third vector, written $v + w \in V$.
>
> - **Scalar multiplication**: An operation that takes a scalar $c \in F$ and a vector $v \in V$ and produces a new vector, written $cv \in V$.
>
>
> These operations must respect the following properties:
>
> - Associativity of Addition: $(u+v)+w=u+(v+w)$ for all $u,v,w \in V$
>
> - Zero Vector (Additive Identity): There is a vector in $V$ written as $0$ and called the *zero vector* which has the property that $u+0=u$ for all $u \in V$
>
> - Negatives (Additive Inverse): For every $u \in V$, there is a vector in $V$ written as $-u$ and called the negative of $u$ which has the property that $u+(-u)=0$.
>
> - Associativity of Scalar Multiplication: $(ab)u=a(bu)$ for any $a,b \in \mathbb{F}$ and $u \in V$
>
> - Distributivity of Scalar Multiplication: $(a+b)u=au+bu$ and $a(u+v)=au+av$ for all $a,b \in \mathbb{F}$ and $u,v \in V$
>
> - Unitarity (Multiplicative Identity): $1u=u$ for all $u \in V$


Now, there's quite a bit of jargon in the definition that needs to be deconstructed. Firstly, a field $\mathbb{F}$ is a "set of numbers with the property that if $a,b \in \mathbb{F}$, then $a+b$, $a-b$, $ab$, and $\frac{a}{b}$ are also in $\mathbb{F}$ (assuming $b \neq 0$ for division)."

The most common examples which come up in engineering are just the "numbers" we're familiar with: $\mathbb{R}, \mathbb{Q}, \mathbb{C}$. This definition will do for our purposes.

## Why Vector Spaces?

Onto the important question: why is this the definition of a vector space? This categorization proves valuable because numerous mathematical structures naturally satisfy these properties. While $\mathbb{R}^n$ is the most familiar example, vector spaces also include polynomials, continuous functions, matrices, and many other mathematical objects. This generality allows us to develop universal tools applicable across mathematical domains. We can also "translate" between vector spaces to apply techniques from one context to another, often making complex problems more intuitive. 

## The value in the generality of vector spaces

This still feels abstract. To get a more intuitive understanding for the fundamental nature of vector spaces, it's helpful to work back from intuition. We typically think about vector spaces as lines or points in a coordinate space -- and there's a very natural and fundamental reason for this. It can be encapsulated in the line: "there is only one vector space of dimension 𝑛 over a field $\mathbb{F}$ up to isomorphism". 

Coordinate spaces represent perhaps the most natural embodiment of vector spaces. For the following explanation, I'll use the term "basis" without extensive elaboration—you can think of a basis as the set of independent vectors that span the entire space. In $\mathbb{R}^2$, for example, the standard basis vectors are $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$. 

Let $V$ be an $n$-dimensional vector space over a field $F$ (like $\mathbb{R}$). Given a basis $\mathcal{B} = \{v_1, v_2, \ldots, v_n\}$ for $V$, any vector $v \in V$ can be uniquely expressed as a linear combination:

$$v = c_1v_1 + c_2v_2 + \cdots + c_nv_n$$

where $c_1, c_2, \ldots, c_n \in F$ are scalars.

The coordinate mapping with respect to basis $\mathcal{B}$ is the isomorphism $\phi_{\mathcal{B}}: V \rightarrow F^n$ defined by:

$$\phi_{\mathcal{B}}(v) = (c_1, c_2, \ldots, c_n)$$

This ordered $n$-tuple $(c_1, c_2, \ldots, c_n)$ is called the coordinate vector (or coordinate representation) of $v$ with respect to basis $\mathcal{B}$.

An example isomorphism would be from the set of polynomials up to degree 2 ($a_0+a_1x$) to $\mathbb{R}^2$. The coordinate vector representation is simply $(a_0,a_1)$.

Bringing it back, the quote above was saying that any finite-dimensional vector space over a given field can be mapped isomorphically to any other vector space with the same dimension over the same field. The coordinate space representation feels particularly intuitive because it provides a concrete, numerical way to work with abstract vector spaces. By mapping vectors to ordered tuples in F^n, the coordinate space translation offers a concise, familiar representation that other possible isomorphisms might not provide as directly. This is why coordinate spaces serve as the cornerstone of linear algebra—they provide a unified representation system for all finite-dimensional vector spaces.


Therefore, the answer to "Why vector spaces?" is that many different mathematical constructs are essentially the same as a regular coordinate space (and many other representations of vector spaces), which is something we're very capable of reasoning about. Vector spaces are equipped with a key operation that ties all of them together: linear combination.

## Basis

I mentioned the term "basis" earlier without fully explaining it. Now is the time. A basis is a set of vectors $B=\{v_1,v_2,\dots,v_n\}$ for a vector space satisfies two properties:

- Linear Indepence: No vector in the basis can be written as a linear combination of other basis vectors
- Spanning Set: Every vector in the vector space can be written as a linear combination of the basis vectors

Bases are very intuitive. They are the minimal set of vectors for which we can access every point in the vector space. Consider the "standard basis" of the Euclidean Space $\mathbb{R}^2$: $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

These two vectors are linearly independent because there's no way we can scale the other one to get the first. $0$ will stay $0$. Furthermore, we can reach any vector $\begin{bmatrix} \mathbb{R} \\ \mathbb{R} \end{bmatrix}$ in the vector space by applying a linear combination of these two basis vectors. The basis I've given is known as the "standard basis" for $\mathbb{R}^2$ because it is a basis where each vector has all zeros except a $1$ in one position. I could give another basis $\begin{bmatrix} 3 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 2 \end{bmatrix}$ and this would also respect the properties of a basis.

Also note that we can only transform a vector in another vector space to its coordinate space representation when we are given a basis. This is because the coordinates are really relative to their reference frame. We could represent any point in $\mathbb{R}^2$ using a wide variety of bases, but the coordinates we would use would vary. To reach the point $(6, 6)$ in the standard basis, our coordinates would be $(6,6)$. Using basis $\begin{bmatrix} 3 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 2 \end{bmatrix}$, our coordinates would be $(2,3)$.

The isomorphism that maps a vector space to their coordinate representation is explicitly defined in terms of a specific basis. Once we define matrices and linear transformations, the fundamental mathematical construct which allows for mapping between vector spaces.

## Dimensions

Dimensions are a fundamental property of vector spaces which tell us "how big" the space is. The dimension of a vector space is the number of basis vectors needed to span the entire space. $\mathbb{R}^2$ requires two vectors. The power of dimensions lies in its invariance to the basis representation: regardless of which basis we choose, the number of vectors needed to span a certian vector space will always be the same. 

## Vector Subspaces

Just as vector spaces let us work with diverse mathematical objects, vector subspaces let us identify subsets of a vector space which have some desirable properties. A vector subspace $W$ of a vector space $V$ is a subset which inherits the properties of the surrounding vector space, including the key properties of the zero vector and closure under addition and multiplication. A subspace is essentially a subset where linear operations remain self-contained. Notably, since the vector subspace inherits all properties of the larger vector space, it is itself a vector space.

The dimension of a subspace must be less than or equal to the dimension of its parent space. 

## Matrices and Linear Transformations

A linear transformation defines a mapping between one vector space and another. The vector space can be the same vector space, in which case we call the transformation a linear endomorphism. Linear endomorphisms are very common and intuitive when thought of as a "change in perspective". 

### Linear Endomorphisms

Take the vector $\begin{bmatrix} -1 \\ 2 \end{bmatrix}$. We know what this means graphically under the standard basis vectors in a cartesian coordinate system. You go one to the left and two up. If we use the standard basis vectors:
$$
\begin{align*}
e_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \\ 
e_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \\ 
\end{align*}
$$

We perform the following linear combination to reach that point: $-1 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 2 \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix}$. This means our coordinates are $[-1, 2]$.

We can represent this linear transformation using the identity matrix:
$$
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

Now, what if we were working with a different coordinate system? What if our basis vectors were:
$$
\begin{align*}
v_1 = \begin{bmatrix} -1 \\ 0 \end{bmatrix} \\
v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
\end{align*}
$$

To reach the same physical point in space using these basis vectors, we would need to perform the linear combination: $1 \cdot \begin{bmatrix} -1 \\ 0 \end{bmatrix} + 2 \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}$. 

This means our coordinates in this new basis are $[1, 2]$. The key insight: $[-1, 2]$ in the standard basis and $[1, 2]$ in our new basis represent the exact same point in space. The transformation is really just a "change in perspective".

We can organize our new basis vectors as columns in a matrix $P$:
$$
P = \begin{bmatrix}
-1 & 0 \\
0 & 1
\end{bmatrix}
$$

When we multiply $P$ by the coordinates in our new basis, we get the standard basis coordinates:
$$
P \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix}
-1 & 0 \\
0 & 1
\end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} -1 \\ 2 \end{bmatrix}
$$

To summarize: The linear transformation $P$ converts coordinates from our new basis to the standard basis. The columns of $P$ are precisely the new basis vectors expressed in the standard basis. When we say a point has coordinates $[1, 2]$ in the basis given by the columns of $P$, this corresponds exactly to the point with coordinates $[-1, 2]$ in the standard basis.

I want to loop back to the statement, "the dimension of a subspace must be less than or equal to the dimension of its parent space." This relates to a key fact about endomorphisms: they may map to a lower dimensional subspace.

Take the following matrix:

$$
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

Note that the columns of the matrix do not form a basis. They have dimension 1. While we can transform any 2-vector with this matrix, it will necessarily map to a lower dimensional subspace (a line) under the standard basis vectors.

Here are two example transformations:

$$
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix}
$$

$$
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix}
$$

Note that these two resultant vectors are on the same line because our matrix does not form a basis. After undergoing the linear transformation, we are "losing information" by sending all vectors onto a line. We don't know whether $(3,3)$ came from $(2,1)$ or $(1,2)$. Therefore, the transformation is not invertible. Any $n \times n$ matrix where the columns do not form a basis is not invertible and will "lose information".


### Non-endomorphic Linear Transformations

Of course, not all linear transformations are endomorphic. Oftentimes, a linear transformation will change the vector space we are working with. 

$$
\begin{bmatrix}
1 & 2 & 1 \\
3 & 1 & 4
\end{bmatrix}
$$

This matrix maps vectors in $\mathbb{R}^3$ to $\mathbb{R}^2$. It's like casting a shadow onto a 2D surface from a 3D form. No matter what, we are going to lose information. This is reflected in the matrix, as any one of the vectors can be expressed as a linear combination of one or two of the other vectors. Similar to the example where both columns were $(1,1)$, we are losing information with this transformation.

How about the other case? What if we have a tall matrix?

$$
\begin{bmatrix}
1 & 2 \\
3 & 1 \\
2 & 5
\end{bmatrix}
$$

In this case, our input is a 2-vector which we are transforming into a 3-vector. The input only has two dimensions, so there's no way we can reach every 3D point with these inputs. Alternatively, two 3-vectors cannot span all of $\mathbb{R}^3$, only a plane at best.


### Matrix Terminology

- Domain: the set of all inputs for which the function is defined. For the wide matrix example, the domain is $\mathbb{R}^3$ because we are passing in 3-vectors.
- Codomain: the set of all possible output values where the function is defined to map into. For the tall matrix example, our codomain is $\mathbb{R}^3$. You might be wondering: I thought this matrix only mapped to a plane? Well, yes, but it maps to a plane *in* $\mathbb{R}^3$. Our transformation outputs 3-vectors, so the codomain is $\mathbb{R}^3$.
- Range/Image: These refer to the same thing. The range is the set of all possible outputs of our function, determined by the subspace which the columns of the matrix span. For a matrix $A$, the image consists of all vectors $y=Ax$ for some $x$ in the domain. The image is a subspace of the codomain, although it may equal the codomain. In our tall matrix example, the range is a plane in $\mathbb{R}^3$.
- Rank - The dimension of the image/range. In the wide matrix example, we can reach any point in $\mathbb{R}^2$ with room to spare (an extra column). Therefore, the rank is $2$. In the tall matrix example, we can only reach points on a 2D plane. The rank is also $2$. A matrix is known as full rank if it spans as many possible dimensions as it can given the shape. In the case of our tall matrix, this means it must output a plane in $\mathbb{R}^3$. For the wide matrix, this means it must output all of $\mathbb{R}^2$.

### Onto and One-to-One

The terms "onto" and "one-to-one" describe functions generally, but they are very helpful for linear transformations too.

#### Onto

A function is onto if every element in the codomain is mapped to by at least one element in the domain. To put it mathematically, for a function $f: X \rightarrow Y$, it means:

> For every $y \in Y,$ there exists at least one $x \in X$ such that $f(x)=y$

Our wide matrix maps 3-vectors to 2-vectors. If our wide matrix has full rank (rank = 2 in this case), then it is onto. This means every 2-vector in the codomain $\mathbb{R}^2$ can be reached by applying our transformation to some 3-vector in the domain $\mathbb{R}^3$.

Geometrically, this means the transformation "covers" the entire 2-dimensional codomain. Since we're starting with a higher-dimensional space ($\mathbb{R}^3$) and mapping to a lower-dimensional space ($\mathbb{R}^2$), we have "enough room" in our domain to reach every point in the codomain.

However, this wide matrix cannot be one-to-one. Since we're mapping from a 3-dimensional space to a 2-dimensional space, we must "collapse" or "compress" some dimensions. This means multiple different input vectors will map to the same output vector.

#### One-to-One

A function is one-to-one if each element in the codomain is mapped to by at most one element in the domain. To put it mathematically, for a function $f: X \rightarrow Y$, it means:
> $f(x_1)=f(x_2)$ implies $x_1=x_2$.

For our tall matrix that maps 2-vectors to 3-vectors, if it has full rank (rank = 2 in this case), then it is one-to-one. This means different input vectors will always produce different output vectors when transformed.

Geometrically, this means the transformation preserves the "distinctness" of points. Since we're mapping from a lower-dimensional space ($\mathbb{R}^2$) to a higher-dimensional space ($\mathbb{R}^3$), we have "plenty of room" in our codomain for each input to have its own unique output without any overlap.

There's a symmetry here: a full-rank tall matrix will always be one-to-one, and a full-rank wide matrix will always be onto. This comes together with the following fact: A full rank square matrix is both one-to-one and onto. This property means that the transformation is invertible. We can take *any* point in the output space (onto) and uniquely (one-to-one) identify it's point in the input space.

### Nullity

Nullity lets us quantify the amount of information that is lost in a transformation. When we say that a transformation loses information, we can think about the transformation squashing the input domain down into a lower dimension. As it happens, there is some direction along which every vector is squashed directly into $0$. We call the set of vectors for which the transformation maps it to $0$ the "null space". The dimension of the null space is known as "nullity". 

In a wide matrix, we necessarily have information squashed down into a lower dimension. Therefore, the nullity must be non-zero for a wide matrix. In the case of square or tall matrices, it's not likely that we'll be squashing down given a random initialization (although it is possible if rows are linearly dependent).

If we think of the nullity as the number of dimensions which are lost in the transformation and rank as the number of dimensions which are preserved, it's relatively intuitive that the rank + nullity will give us the dimension of the input vector space. To reinforce this, I will go through examples for each type of matrix (wide, square, and tall):

#### Wide:
$$
\begin{bmatrix}
1 & 2 & 1 \\
3 & 1 & 4
\end{bmatrix}
$$

For the following matrix, we are squashing 3D space into 2D. There will necessarily be overlap (not one-to-one). This matrix is full rank because we still span all of $\mathbb{R}^2$. The nullity is one because one of the dimensions in the input space will be lost. $2+1=3$, which is the dimension of $\mathbb{R}^3$

#### Square:
$$
\begin{bmatrix}
1 & 2 \\
3 & 1
\end{bmatrix}
$$

For the following matrix, we are mapping 2D space to 2D space. This matrix is full rank because the columns span all of $\mathbb{R}^2$. The nullity is zero because no dimensions are lost in the transformation. $2+0=2$, which is the dimension of $\mathbb{R}^2$.

#### Tall:
$$
\begin{bmatrix}
1 & 2 \\
3 & 1 \\
2 & 5
\end{bmatrix}
$$

For the following matrix, we are mapping 2D space to 3D space. This matrix is full rank (with respect to the input) because it preserves both dimensions of the input space. The nullity is zero because no dimensions are lost in the transformation. $2+0=2$, which is the dimension of $\mathbb{R}^2$.

In the next post, I'll get more into the row picture side of things and break down some more advanced concepts that will be helpful for intuition of the Kalman Filter.

Source: 

- https://www.math.toronto.edu/gscott/WhatVS.pdf
- https://math.stackexchange.com/questions/459776/difference-between-coordinate-space-and-vector-space