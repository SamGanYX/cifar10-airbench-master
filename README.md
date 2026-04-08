\documentclass{article}

% Template taken from CSE517 class. Originally created by Noah Smith.

% to compile a camera-ready version, add the [final] option, e.g.:

\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc} % use 8-bit T1 fonts
\usepackage{hyperref}% hyperlinks
\usepackage{url} % simple URL typesetting
\usepackage{booktabs} % professional-quality tables
\usepackage{amsfonts} % blackboard math symbols
\usepackage{nicefrac} % compact symbols for 1/2, etc.
\usepackage{microtype} % microtypography
\usepackage{amsmath} % For math environments like align*
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{float}
\geometry{margin=1in}
\title{\textbf{Replication and Extension of High-Speed CIFAR-10 Training}}

\author{%
\textbf{Yunxin Gan} \\
\textbf{University of Washington, CSE 493S} \\
\texttt{\textbf{samganyx@uw.edu}} \\
}

\begin{document}

\maketitle

\section*{\centering Machine Learning Project Summary}

\subsection*{Project Scope}

{The main goal of this project was a replication and extension of the high-speed CIFAR-10 training methodology presented in the paper, ``94\% on CIFAR-10 in 3.29 Seconds on a Single GPU''. The original paper's key contribution was achieving $94\%$ validation accuracy in minimal Time-to-Accuracy (TTA). This project aimed to successfully replicate the speed optimizations to meet or exceed the target TTA of $3.29$ seconds. The extension involved integrating and evaluating the impact of the \textbf{CutMix} data augmentation technique on both TTA and final accuracy, specifically to experiment with how it deals with overtraining as a regularization method.}

\subsection*{Methodology}

{The project began with an initial ResNet implementation achieving $94\%$ accuracy with a TTA of $\sim3$ minutes. The speed replication involved implementing hardware and algorithmic optimizations: 1) \textbf{Optimized CPU-GPU interaction} using a custom GPU-Resident Loader and JIT compilation. 2) \textbf{Pure Mixed-Precision} training, casting most weights to $\texttt{FP16}$ while maintaining BatchNorm in $\texttt{FP32}$ for numerical stability. 3) The \textbf{Muon} optimizer, which uses gradient whitening and Newton-Schulz iteration for decorrelated, orthogonal gradient updates. The baseline training achieved a TTA of $2.84$ seconds. The extension replaced standard augmentations with \textbf{CutMix} augmentation to observe its effect on overtraining. The project was executed on a NVIDIA A100 GPU.}

\subsection*{Results}

{The replication of the high-speed training was highly successful, achieving a TTA of $\mathbf{2.84}$ seconds to reach $\sim94.01\%$ accuracy, which is faster than the paper's $3.29$ seconds. This fully supports the original claim. However, the extension with \textbf{CutMix} augmentation led to a reduced average accuracy of $\mathbf{90.57\%}$ with an increased training time of $\mathbf{2.98}$ seconds. This indicates that CutMix, when used alone as a strategy against overtraining, underperformed the standard geometric augmentations. Training accuracy was also low and unstable due to the complexity of predicting soft labels.}

\subsection*{What was Easy}

{The conceptual design of the speedup techniques was straightforward, particularly the idea of using a GPU-Resident Loader to eliminate CPU-GPU data transfer latency, which is an intuitive performance optimization. The model and dataset ($\texttt{ResNet}$ on $\texttt{CIFAR-10}$) are standard, facilitating initial setup. The final performance metric, TTA, was a clear, objective number that was easily measurable.}

\subsection*{What was Difficult}

{The most time-consuming aspects were the low-level implementation details required for peak speed. The \textbf{Muon} optimizer, with its use of the Newton-Schulz iteration for gradient cleaning, required specialized implementation. Implementing the \textbf{pure Mixed-Precision} approach without automated tools meant carefully managing $\texttt{FP16}$ and $\texttt{FP32}$ casting for numerical stability, especially for sensitive BatchNorm layers. Finally, \textbf{CutMix} integration was challenging because generating random coordinates, slicing tensors, and calculating soft labels on the fly added overhead that was difficult to optimize within the highly accelerated pipeline.}


\section{Introduction}
{The goal of achieving high-speed training in deep learning is vital for accelerating research and reducing experimental costs. This project is built upon the methodology introduced in the paper, ``94\% on CIFAR-10 in 3.29 Seconds on a Single GPU,'' which achieved an impressive Time-to-Accuracy (TTA) on the CIFAR-10 benchmark. The original work combined low-level system optimizations (efficient CPU-GPU interaction and mixed-precision training) with advanced algorithmic techniques (the Muon optimizer).}

{This report documents the successful replication of this performance claim, achieving a TTA of $2.84$ seconds, faster than the original's $3.29$ seconds. The report also details an extension that evaluated the \textbf{CutMix} data augmentation technique within this highly optimized pipeline, investigating its effect on both speed and final accuracy to combat overtraining, compared to the standard data augmentation used in the baseline.}


\section{Scope of the Project}

{The project aimed to rigorously verify the high-speed training capability on the CIFAR-10 dataset and investigate the performance trade-offs of an advanced regularization technique aimed at mitigating overtraining within the optimized pipeline.}

\subsection{Addressed Claims/Hypothesis from the Original Paper} \label{claims}

Clearly enumerate the claims you are testing:
\begin{enumerate}
\item \textbf{Replication Claim:} The combination of CPU-GPU interaction optimization, JIT compilation, pure mixed-precision training, and the Muon optimizer is sufficient to achieve $\sim94\%$ validation accuracy on CIFAR-10 with a Time-to-Accuracy (TTA) comparable to or faster than the original reported time of $3.29$ seconds.
\item \textbf{Extension Hypothesis:} Integrating the \textbf{CutMix} data augmentation technique into the optimized pipeline will maintain a final accuracy near the $\sim94\%$ baseline while retaining a fast TTA, demonstrating its effectiveness as a regularization method to address overtraining in a speed-optimized environment.
\end{enumerate}


\section{Methodology}

{The project used a ResNet \cite{He2016} architecture trained on the CIFAR-10 dataset. To achieve the competitive TTA, the following methodologies were implemented:}

\begin{itemize}
    \item \textbf{CPU-GPU Interaction Optimization:} The standard $\texttt{DataLoader}$ was replaced with a \textbf{custom GPU-Resident Loader} that pre-loads the full dataset onto the GPU, eliminating the latency of data transfer during every iteration. The model was wrapped with $\texttt{torch.jit.script}$ for \textbf{JIT compilation}, which reduces Python overhead from many small function calls and optimizes kernel calls.
    \item \textbf{Pure Mixed-Precision:} The core model weights were initialized in \texttt{FP16} to leverage memory bandwidth and Tensor Cores. This manual approach avoided the overhead of $\texttt{torch.cuda.amp.autocast}$ and $\texttt{GradScaler}$. Critically, \textbf{BatchNorm} layers were cast back to \texttt{FP32} for numerical stability.
    \item \textbf{Muon Optimizer:} The \textbf{Muon} optimizer was used instead of $\texttt{SGD}$. Muon uses \textbf{gradient whitening} to decorrelate gradients and treats parameters as matrices. It employs \textbf{Newton-Schulz iteration} as a fast approximation for finding an orthogonal matrix to clean up gradient updates, avoiding the slow SVD.
\end{itemize}

{The CutMix extension involved replacing the baseline's geometric augmentations with the CutMix process, specifically to experiment with how it deals with overtraining. This required generating random coordinates, slicing and mixing tensors, and calculating proportional \textbf{soft labels} on the fly for every batch. The CutMix operation is defined as:
\begin{equation}
    \tilde{x} = \lambda x_A + (1-\lambda) x_B
\end{equation}
where $x_A, x_B$ are training images and $\lambda$ is the mixing ratio.}

{The computational requirements were very low for single runs, with the baseline achieving TTA in $2.84$ seconds. The project was implemented in $\texttt{PyTorch}$ and executed on a NVIDIA A100 GPU.}


\section{Results/Summary}

{The replication successfully validated the high-speed TTA claim, but the extension showed performance degradation with CutMix.}

\subsection{Replication Result: High-Speed TTA}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{accuracy_over_epochs.png}
    \caption{Validation Accuracy over Epochs for different optimization versions.}
    \label{fig:acc_epochs}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{accuracy_over_time.png}
    \caption{Validation Accuracy over Time (seconds). Note the significant speedup in the final versions.}
    \label{fig:acc_time}
\end{figure}

{The implementation of the combined speed optimizations successfully achieved the replication claim, demonstrating performance superior to the original paper's reported time.}

\begin{table}[h]
\caption{\textbf{Replication Performance Metrics}}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Original Paper Claim} & \textbf{Project Replication} & \textbf{Claim Support} \\
\midrule
Target Accuracy & $\sim94\%$ & $\sim94.01\%$ & Supported \\
Time-to-Accuracy (TTA) & $3.29$ seconds & $\mathbf{2.84}$ seconds & Supported (Exceeded) \\
\bottomrule
\end{tabular}
\end{table}

{The $\mathbf{2.84}$ second TTA confirms that the optimization pipeline is highly effective and reproducible.}

\subsection{Extension Result: CutMix Augmentation}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{final_time_comparison.png}
    \caption{Average Total Training Time comparison. Lower is better.}
    \label{fig:time_comp}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{final_accuracy_comparison.png}
    \caption{Average Final Validation Accuracy comparison. Higher is better.}
    \label{fig:acc_comp}
\end{figure}

{The CutMix extension resulted in lower final accuracy and slight time overhead, leading to the rejection of the extension hypothesis.}

\begin{table}[h]
\caption{\textbf{Comparative Analysis: Baseline vs. CutMix Extension}}
\centering
\begin{tabular}{lccp{5cm}}
\toprule
\textbf{Metric} & \textbf{Baseline} & \textbf{CutMix} & \textbf{Analysis} \\
\midrule
Final Accuracy & $\sim94.01\%$ (Expected) & $\mathbf{90.57\%}$ & CutMix underperformed standard geometric augmentations. \\
Training Time (TTA) & $2.84$s & $\mathbf{2.98}$s & CutMix added $0.14$s overhead. \\
Train Accuracy & High (usually $>99\%$) & Low / Unstable & Strong regularization to combat overtraining makes learning harder; short duration prevents convergence. \\
\bottomrule
\end{tabular}
\end{table}

{The lower accuracy ($\mathbf{90.57\%}$) suggests that the structural benefits of standard geometric augmentations were more critical for CIFAR-10 performance than the strong anti-overtraining regularization provided by CutMix. The increased training time ($\mathbf{2.98}$s) is due to the computational cost of on-the-fly mixing and soft label calculation.}


\section{Discussion}

{The replication success, achieving a TTA of $2.84$ seconds, validates that the paper's claimed speed is highly reproducible. This underscores the necessity of a holistic approach: both low-level system engineering (GPU-Resident Loader, JIT) and advanced algorithms (Muon) must be optimized for peak performance.}

{The CutMix extension demonstrated a clear trade-off. While it provides strong regularization to curb overtraining by ensuring all pixels contribute to learning, its performance loss in final accuracy and its time overhead ($\mathbf{0.14}$s) suggest it's not a drop-in replacement for the highly-optimized baseline augmentations. The unstable training accuracy is a direct result of the more difficult task of predicting soft labels. Furthermore, strong regularizers like CutMix typically require longer training schedules to allow the model to learn from complex, mixed samples. In this ultra-low latency regime ($<3$ seconds), the model likely underfits these complex samples, leading to lower accuracy compared to the simpler baseline augmentations.}

\subsection{What was Easy}
{Implementing the concept of eliminating the per-iteration CPU-GPU data transfer using a GPU-Resident Loader was straightforward and yielded an immediate, intuitive performance boost. The model definition (ResNet) and dataset (CIFAR-10) were standard, easing the foundational steps.}

\subsection{What was Difficult}
{Implementing the \textbf{Muon} optimizer was difficult due to the specialized mathematical component, specifically the fast approximation of an orthogonal matrix using the Newton-Schulz iteration. The \textbf{pure mixed-precision} approach, without high-level automated tools, required laborious, manual management of $\texttt{FP32}$ casts for components like BatchNorm to maintain numerical stability. The on-the-fly computational cost of \textbf{CutMix} in a speed-optimized environment proved challenging to manage, as it introduced unexpected overhead.}

\subsection{Recommendations for Future Work}

\textbf{To build upon this project, the following recommendations are suggested:}
\begin{enumerate}
    \item \textbf{Hybrid Augmentation:} Evaluate combining CutMix with the standard geometric augmentations (flipping/random cropping) to potentially leverage the best of both worlds.
    \item \textbf{CutMix Optimization:} Investigate methods to pre-calculate the CutMix parameters (random coordinates and soft labels) on the CPU to eliminate the overhead introduced to the GPU during the training loop.
    \item \textbf{Higher Accuracy Target:} Attempt the replication of the $\sim96\%$ accuracy milestone, which requires further investigation into hyperparameter tuning and additional state-of-the-art techniques.
\end{enumerate}


\begin{thebibliography}{1}

\bibitem{He2016}
\textbf{He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep Residual Learning for Image Recognition. \textit{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 770-778.}

\bibitem{Yun2019}
\textbf{Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., \& Yoo, Y. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. \textit{Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}, 6023-6032.}

\bibitem{Jordan2024}
\textbf{Jordan, K. (2024). \textit{94\% on CIFAR-10 in 3.29 Seconds on a Single GPU}. arXiv:2404.00498.} \url{https://arxiv.org/abs/2404.00498}

\end{thebibliography}


\end{document}
