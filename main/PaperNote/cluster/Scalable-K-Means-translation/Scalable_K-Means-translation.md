# [论文翻译] Scalable K-Means by Ranked Retrieval



## Abstract

The k-means clustering algorithm has a long history and a proven practical performance, however it does not scale to clustering millions of data points into thousands of clusters in high dimensional spaces. The main computational bottleneck is the need to recompute the nearest centroid for every data point at every iteration, a prohibitive cost when the number of clusters is large. In this paper we show how to reduce the cost of the k-means algorithm by large factors by adapting ranked retrieval techniques. Using a combination of heuristics, on two real life data sets the wall clock time per iteration is reduced from 445 minutes to less than 4, and from 705 minutes to 1.4, while the clustering quality remains within 0.5% of the k-means quality.

The key insight is to invert the process of point-to-centroid assignment by creating an inverted index over all the points and then using the current centroids as queries to this index to decide on cluster membership. In other words, rather than each iteration consisting of “points picking centroids”, each iteration now consists of “centroids picking points”. This is much more efficient, but comes at the cost of leaving some points unassigned to any centroid. We show experimentally that the number of such points is low and thus they can be separately assigned once the final centroids are decided. To speed up the computation we sparsify the centroids by pruning low weight features. Finally, to further reduce the running time and the number of unassigned points, we propose a variant of the WAND algorithm that uses the results of the intermediate results of nearest neighbor computations to improve performance.

```
K-means 聚类算法历史悠久且具有良好的实际表现，但在高维空间中将数百万数据点聚类到数千个簇时无法扩展。主要的计算瓶颈是每次迭代都需要为每个数据点重新计算最近的质心，当簇的数量很大时，这种成本是难以承受的。在本文中，我们展示了如何通过适应排序检索技术来大幅度降低 k-means 算法的成本。通过结合启发式方法，在两个真实数据集上，每次迭代的时钟时间从 445 分钟减少到不到 4 分钟，从 705 分钟减少到 1.4 分钟，同时聚类质量保持在 k-means 质量的 0.5% 以内。

关键见解是通过在所有点上创建倒排索引，然后使用当前质心作为查询来决定簇的归属，从而反转点到质心分配的过程。换句话说，每次迭代不再是“点选择质心”，而是“质心选择点”。这更为高效，但代价是某些点未分配给任何质心。我们的实验表明，这样的点数量很少，因此可以在最终确定质心后单独分配。为了加快计算速度，我们通过修剪低权重特征来稀疏化质心。最后，为了进一步减少运行时间和未分配点的数量，我们提出了一种 WAND 算法的变体，使用最近邻计算的中间结果来提高性能。
```



