# Replication and Extension of High-Speed CIFAR-10 Training

**Yunxin Gan**  
**University of Washington, CSE 493S**  
**samganyx@uw.edu**

---

## Machine Learning Project Summary

### Project Scope

The main goal of this project was a replication and extension of the high-speed CIFAR-10 training methodology presented in the paper, "94% on CIFAR-10 in 3.29 Seconds on a Single GPU". The original paper's key contribution was achieving 94% validation accuracy in minimal Time-to-Accuracy (TTA). This project aimed to successfully replicate the speed optimizations to meet or exceed the target TTA of 3.29 seconds. The extension involved integrating and evaluating the impact of the **CutMix** data augmentation technique on both TTA and final accuracy, specifically to experiment with how it deals with overtraining as a regularization method.

### Methodology

The project began with an initial ResNet implementation achieving 94% accuracy with a TTA of ~3 minutes. The speed replication involved implementing hardware and algorithmic optimizations: 
1) **Optimized CPU-GPU interaction** using a custom GPU-Resident Loader and JIT compilation. 
2) **Pure Mixed-Precision** training, casting most weights to `FP16` while maintaining BatchNorm in `FP32` for numerical stability. 
3) The **Muon** optimizer, which uses gradient whitening and Newton-Schulz iteration for decorrelated, orthogonal gradient updates. 

The baseline training achieved a TTA of 2.84 seconds. The extension replaced standard augmentations with **CutMix** augmentation to observe its effect on overtraining. The project was executed on an NVIDIA A100 GPU.

### Results

The replication of the high-speed training was highly successful, achieving a TTA of **2.84** seconds to reach ~94.01% accuracy, which is faster than the paper's 3.29 seconds. This fully supports the original claim. However, the extension with **CutMix** augmentation led to a reduced average accuracy of **90.57%** with an increased training time of **2.98** seconds. This indicates that CutMix, when used alone as a strategy against overtraining, underperformed the standard geometric augmentations. Training accuracy was also low and unstable due to the complexity of predicting soft labels.

### What was Easy

The conceptual design of the speedup techniques was straightforward, particularly the idea of using a GPU-Resident Loader to eliminate CPU-GPU data transfer latency, which is an intuitive performance optimization. The model and dataset (`ResNet` on `CIFAR-10`) are standard, facilitating initial setup. The final performance metric, TTA, was a clear, objective number that was easily measurable.

### What was Difficult

The most time-consuming aspects were the low-level implementation details required for peak speed. The **Muon** optimizer, with its use of the Newton-Schulz iteration for gradient cleaning, required specialized implementation. Implementing the **pure Mixed-Precision** approach without automated tools meant carefully managing `FP16` and `FP32` casting for numerical stability, especially for sensitive BatchNorm layers. Finally, **CutMix** integration was challenging because generating random coordinates, slicing tensors, and calculating soft labels on the fly added overhead that was difficult to optimize within the highly accelerated pipeline.

---

## Introduction

The goal of achieving high-speed training in deep learning is vital for accelerating research and reducing experimental costs. This project is built upon the methodology introduced in the paper, "94% on CIFAR-10 in 3.29 Seconds on a Single GPU," which achieved an impressive Time-to-Accuracy (TTA) on the CIFAR-10 benchmark. The original work combined low-level system optimizations (efficient CPU-GPU interaction and mixed-precision training) with advanced algorithmic techniques (the Muon optimizer).

This report documents the successful replication of this performance claim, achieving a TTA of 2.84 seconds, faster than the original's 3.29 seconds. The report also details an extension that evaluated the **CutMix** data augmentation technique within this highly optimized pipeline, investigating its effect on both speed and final accuracy to combat overtraining, compared to the standard data augmentation used in the baseline.

## Scope of the Project

The project aimed to rigorously verify the high-speed training capability on the CIFAR-10 dataset and investigate the performance trade-offs of an advanced regularization technique aimed at mitigating overtraining within the optimized pipeline.

### Addressed Claims/Hypothesis from the Original Paper

Clearly enumerate the claims you are testing:
1. **Replication Claim:** The combination of CPU-GPU interaction optimization, JIT compilation, pure mixed-precision training, and the Muon optimizer is sufficient to achieve ~94% validation accuracy on CIFAR-10 with a Time-to-Accuracy (TTA) comparable to or faster than the original reported time of 3.29 seconds.
2. **Extension Hypothesis:** Integrating the **CutMix** data augmentation technique into the optimized pipeline will maintain a final accuracy near the ~94% baseline while retaining a fast TTA, demonstrating its effectiveness as a regularization method to address overtraining in a speed-optimized environment.

## Methodology

The project used a ResNet architecture trained on the CIFAR-10 dataset. To achieve the competitive TTA, the following methodologies were implemented:

* **CPU-GPU Interaction Optimization:** The standard `DataLoader` was replaced with a **custom GPU-Resident Loader** that pre-loads the full dataset onto the GPU, eliminating the latency of data transfer during every iteration. The model was wrapped with `torch.jit.script` for **JIT compilation**, which reduces Python overhead from many small function calls and optimizes kernel calls.
* **Pure Mixed-Precision:** The core model weights were initialized in `FP16` to leverage memory bandwidth and Tensor Cores. This manual approach avoided the overhead of `torch.cuda.amp.autocast` and `GradScaler`. Critically, **BatchNorm** layers were cast back to `FP32` for numerical stability.
* **Muon Optimizer:** The **Muon** optimizer was used instead of `SGD`. Muon uses **gradient whitening** to decorrelate gradients and treats parameters as matrices. It employs **Newton-Schulz iteration** as a fast approximation for finding an orthogonal matrix to clean up gradient updates, avoiding the slow SVD.

The CutMix extension involved replacing the baseline's geometric augmentations with the CutMix process, specifically to experiment with how it deals with overtraining. This required generating random coordinates, slicing and mixing tensors, and calculating proportional **soft labels** on the fly for every batch. The CutMix operation is defined as:

$$ \tilde{x} = \lambda x_A + (1-\lambda) x_B $$

where $x_A, x_B$ are training images and $\lambda$ is the mixing ratio.

The computational requirements were very low for single runs, with the baseline achieving TTA in 2.84 seconds. The project was implemented in `PyTorch` and executed on a NVIDIA A100 GPU.

## Results/Summary

The replication successfully validated the high-speed TTA claim, but the extension showed performance degradation with CutMix.

### Replication Result: High-Speed TTA

*(See original paper/report for figures such as Validation Accuracy over Epochs and Time)*

The implementation of the combined speed optimizations successfully achieved the replication claim, demonstrating performance superior to the original paper's reported time.

**Replication Performance Metrics**

| Metric | Original Paper Claim | Project Replication | Claim Support |
| :--- | :--- | :--- | :--- |
| Target Accuracy | ~94% | ~94.01% | Supported |
| Time-to-Accuracy (TTA) | 3.29 seconds | **2.84** seconds | Supported (Exceeded) |

The **2.84** second TTA confirms that the optimization pipeline is highly effective and reproducible.

### Extension Result: CutMix Augmentation

The CutMix extension resulted in lower final accuracy and slight time overhead, leading to the rejection of the extension hypothesis.

**Comparative Analysis: Baseline vs. CutMix Extension**

| Metric | Baseline | CutMix | Analysis |
| :--- | :--- | :--- | :--- |
| Final Accuracy | ~94.01% (Expected) | **90.57%** | CutMix underperformed standard geometric augmentations. |
| Training Time (TTA) | 2.84s | **2.98s** | CutMix added 0.14s overhead. |
| Train Accuracy | High (usually >99%) | Low / Unstable | Strong regularization to combat overtraining makes learning harder; short duration prevents convergence. |

The lower accuracy (**90.57%**) suggests that the structural benefits of standard geometric augmentations were more critical for CIFAR-10 performance than the strong anti-overtraining regularization provided by CutMix. The increased training time (**2.98s**) is due to the computational cost of on-the-fly mixing and soft label calculation.

## Discussion

The replication success, achieving a TTA of 2.84 seconds, validates that the paper's claimed speed is highly reproducible. This underscores the necessity of a holistic approach: both low-level system engineering (GPU-Resident Loader, JIT) and advanced algorithms (Muon) must be optimized for peak performance.

The CutMix extension demonstrated a clear trade-off. While it provides strong regularization to curb overtraining by ensuring all pixels contribute to learning, its performance loss in final accuracy and its time overhead (**0.14s**) suggest it's not a drop-in replacement for the highly-optimized baseline augmentations. The unstable training accuracy is a direct result of the more difficult task of predicting soft labels. Furthermore, strong regularizers like CutMix typically require longer training schedules to allow the model to learn from complex, mixed samples. In this ultra-low latency regime (<3 seconds), the model likely underfits these complex samples, leading to lower accuracy compared to the simpler baseline augmentations.

### What was Easy
Implementing the concept of eliminating the per-iteration CPU-GPU data transfer using a GPU-Resident Loader was straightforward and yielded an immediate, intuitive performance boost. The model definition (ResNet) and dataset (CIFAR-10) were standard, easing the foundational steps.

### What was Difficult
Implementing the **Muon** optimizer was difficult due to the specialized mathematical component, specifically the fast approximation of an orthogonal matrix using the Newton-Schulz iteration. The **pure mixed-precision** approach, without high-level automated tools, required laborious, manual management of `FP32` casts for components like BatchNorm to maintain numerical stability. The on-the-fly computational cost of **CutMix** in a speed-optimized environment proved challenging to manage, as it introduced unexpected overhead.

### Recommendations for Future Work

**To build upon this project, the following recommendations are suggested:**
1. **Hybrid Augmentation:** Evaluate combining CutMix with the standard geometric augmentations (flipping/random cropping) to potentially leverage the best of both worlds.
2. **CutMix Optimization:** Investigate methods to pre-calculate the CutMix parameters (random coordinates and soft labels) on the CPU to eliminate the overhead introduced to the GPU during the training loop.
3. **Higher Accuracy Target:** Attempt the replication of the ~96% accuracy milestone, which requires further investigation into hyperparameter tuning and additional state-of-the-art techniques.

## References

1. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.
2. **Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019).** CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 6023-6032.
3. **Jordan, K. (2024).** *94% on CIFAR-10 in 3.29 Seconds on a Single GPU*. arXiv:2404.00498. [https://arxiv.org/abs/2404.00498](https://arxiv.org/abs/2404.00498)
