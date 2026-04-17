## Graph Neural Networks for Misinformation Detection: Performance–Efficiency Trade-offs

This repository accompanies [The International Conference on Computational Science (ICCS'26)](https://www.iccs-meeting.org/iccs2026/) the paper:

- **“Graph Neural Networks for Misinformation Detection: Performance–Efficiency Trade-offs”** by *Soveatin Kuntur, Maciej Krzywda, Anna Wróblewska, Marcin Paprzycki, Maria Ganzha, Szymon Łukasik, Amir H. Gandomi* 

**Paper is available** [Arxiv](https://arxiv.org/pdf/2604.08131)

It explores the use of **classic Graph Neural Networks (GNNs)** for misinformation detection, focusing on the trade-off between **predictive performance and computational efficiency**.

The repository provides a controlled comparison between GNN models (e.g., GCN, GraphSAGE, GAT, ChebNet) and strong non-graph baselines (Logistic Regression, SVM, MLP), all operating on the same TF–IDF features and data splits.

The results show that **lightweight GNNs consistently outperform traditional models** while maintaining practical inference costs, making them a strong alternative to more complex architectures.


---

## Motivation

The rapid spread of online misinformation poses a serious challenge to public trust and information reliability. While recent advances in large language models (LLMs) have significantly improved detection performance, these approaches often come with high computational costs, limited robustness across languages, and deployment constraints in real-world settings.

As a result, many modern solutions rely on increasingly complex architectures that combine Transformers, graph models, and multiple feature sources-often at the expense of efficiency and scalability.

In contrast, **Graph Neural Networks (GNNs)** offer a lightweight and interpretable alternative by explicitly modeling relationships between data points, such as similarity between documents or shared semantic patterns. Despite the growing complexity of modern GNN variants, it remains unclear whether **classic, computationally efficient GNNs** are sufficient to achieve strong performance.


---

## Research Questions

This project investigates the following:

* **RQ1:** Are classic GNNs competitive with strong non-graph baselines?
* **RQ2:** Which GNN architectures perform best across different datasets?
* **RQ3:** Do GNNs remain effective with limited training data?
* **RQ4:** What are the performance–efficiency trade-offs compared to non-graph models?

---

## Approach

To answer these questions, we conduct a **large-scale multilingual benchmarking study** of classic GNN models (e.g., GCN, GAT, GraphSAGE, ChebNet).

* All models use identical **TF–IDF features**
* Graphs are constructed using **k-NN similarity**
* We evaluate:

  * full datasets vs low-resource settings
  * different graph sparsity levels
  * optional lightweight pretraining

Rather than proposing new architectures, the goal is to **systematically evaluate when and why classic GNNs work**.


---

## Citation

```latex
@misc{kuntur2026graphneuralnetworksmisinformation,
      title={Graph Neural Networks for Misinformation Detection: Performance-Efficiency Trade-offs}, 
      author={Soveatin Kuntur and Maciej Krzywda and Anna Wróblewska and Marcin Paprzycki and Maria Ganzha and Szymon Łukasik and Amir H. Gandomi},
      year={2026},
      eprint={2604.08131},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.08131}, 
}
```
