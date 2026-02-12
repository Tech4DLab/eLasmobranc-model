# 🧠 🦈 eLasmobranc Model – Taxonomical Identification 

First of all, an example execution hosted on Google Colab is presented, illustrating a practical reference for reproducing the results in an interactive cloud-based environment:

<p>
  <a href="https://zenodo.org/records/18549737">
    <img src="https://img.shields.io/badge/💻 Google Colab-007BFF?style=for-the-badge&logoColor=white" />
  </a>
</p>

This project develops an automatic image-based system for elasmobranch identification using deep learning, addressing the challenges of limited labeled data and high morphological similarity between related species. The approach is structured around two complementary components:

- **Elasmobranch detection model**: distinguishes sharks and rays from other objects present in the images.
- **Hierarchical taxonomic classification model**: performs progressive identification (animal type, order, family and species) by incorporating domain-specific prior knowledge to improve fine-grained recognition under data-scarce conditions.

## Elasmobranch detection model

A dedicated dataset was constructed to frame the problem as a binary classification task, where one class corresponds to elasmobranch species and the other represents all remaining objects, including random items, other marine animals and visually similar scenes without elasmobranchs:

<p align="center">
  <img src="images/no_elas1.jpg" height="150">
  <img src="images/no_elas2.jpg" height="150">
  <img src="images/no_elas3.jpg" height="150">
</p>

The detection model is based on a convolutional neural network with a pretrained EfficientNet-B0 backbone using transfer learning. Fine-tuning is applied to adapt the learned representations to the target domain by retraining the two final network blocks and replacing the output layer with one specifically designed for the problem. The proposed detection model achieves near-perfect performance on the test set, demonstrating its effectiveness for binary elasmobranch recognition. 

<div align="center">
  <img src="images/model1.png" width="800">
</div>

<table align="center">
<tr>

<td align="center">
<img src="images/mc_model1.png" width="200"><br>
<b>🧩 Confusion Matrix</b>
</td>

<td align="center">
<b>📊 Classification Performance</b><br><br>

<table>
<tr>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1-score</th>
</tr>
<tr>
<td>0.998</td>
<td>0.998</td>
<td>1.000</td>
<td>0.999</td>
</tr>
</table>

</td>

</tr>
</table>
