<div align="center">

<h1><a href="https://arxiv.org/abs/xxxx.xxxxx">PoseGAM: Robust Unseen Object Pose Estimation via Geometry-Aware Multi-View Reasoning</a></h1>

**[Jianqi Chen](https://windvchen.github.io/), [Biao Zhang](https://scholar.google.co.uk/citations?user=h5KukxEAAAAJ&hl=en&oi=ao), [Xiangjun Tang](https://scholar.google.co.uk/citations?user=9vZn91sAAAAJ&hl=en&oi=ao), and [Peter Wonka](https://scholar.google.co.uk/citations?user=0EKXSXgAAAAJ&hl=en)**

![](https://komarev.com/ghpvc/?username=windvchenPoseGAM&label=visitors)
![GitHub stars](https://badgen.net/github/stars/windvchen/PoseGAM)
[![](https://img.shields.io/badge/license-MIT-blue)](#License)
[![](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
<a href='https://windvchen.github.io/PoseGAM/'>
  <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'></a>

</div>

![PoseGAM's preface](assets/teaser.png)

### Share us a :star: if this repo does help

This is the official repository of ***PoseGAM***. We're actively organizing the code to enhance the user experience—stay tuned for updates! 🚀

If you encounter any question about the paper, please feel free to contact us. You can create an issue or just send email to me windvchen@gmail.com. Also welcome for any idea exchange and discussion.


## Updates

[**2025/12/11**] Repository init.

## TODO
- [ ] Dataset release
- [ ] Code release

## Table of Contents
- [Abstract](#abstract)
- [Results](#results)
- [Citation \& Acknowledgments](#citation--acknowledgments)
- [License](#license)

## Abstract

![PoseGAM's framework](assets/framework.jpg)

6D object pose estimation, which predicts the transformation of an object relative to the camera, remains challenging for unseen objects. Existing approaches typically rely on explicitly constructing feature correspondences between the query image and either the object model or template images. In this work, we propose PoseGAM, **a geometry-aware multi-view framework that directly predicts object pose from a query image and multiple template images, eliminating the need for explicit matching**. Built upon recent multi-view-based foundation model architectures, the method integrates object geometry information through two complementary mechanisms: explicit point-based geometry and learned features from geometry representation networks. In addition, we construct **a large-scale synthetic dataset containing more than 190k objects under diverse environmental conditions** to enhance robustness and generalization. Extensive evaluations across multiple benchmarks demonstrate our state-of-the-art performance.

## Results

<div align=center><img src="assets/Comparison.png" alt="Visual comparisons1"></div>

## Citation & Acknowledgments
If you find this paper useful in your research, please consider citing:
```
@article{chen2025PoseGAM,
  title={PoseGAM: Robust Unseen Object Pose Estimation via Geometry-Aware Multi-View Reasoning},
  author={Chen, Jianqi and Zhang, Biao and Tang, Xiangjun and Wonka, Peter},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

## License
This project is licensed under the MIT license. See [LICENSE](LICENSE) for details.