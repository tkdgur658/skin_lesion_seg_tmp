\noindent\textbf{Datasets}\enspace
Table~\ref{tab:datasets} summarizes the datasets and splits used in the experiments. Three dermoscopic benchmarks were evaluated, namely ISIC2017~\cite{codella2017isic} with 2,150 images, ISIC2018~\cite{codella2018isic} with 2,594 images, and PH2~\cite{mendonca2013dermoscopic} with 200 images. Each dataset provided pixel-wise lesion annotations that were used as the ground truth. For all benchmarks, the data were randomly partitioned into training, validation, and test sets in a 6:2:2 ratio. Consistent with prior research~\cite{ruan2023efficient}, all images were normalized and resized to 256 $\times$ 256 pixels while preserving aspect ratio with zero-padding. For training, the data was augmented with horizontal and vertical flips as well as random rotations to increase sample diversity.

\noindent\textbf{Implementation Details}\enspace
The proposed method was implemented in PyTorch 2.1.0 and all experiments were run on a single NVIDIA RTX 3090 GPU. The model was trained for 300 epochs with a mini-batch size of 8 using the AdamW optimizer with an initial learning rate of 1e-3, beta values of 0.9 and 0.999, epsilon of 1e-8, and weight decay of 1e-2. The learning rate followed cosine annealing with $T_{\text{max}}=50$ and a minimum learning rate of 1e-6. A compound objective that adds binary cross entropy and Dice loss with equal weights was optimized. The random seed ranged from 1 to 10 to ensure reproducibility across ten runs, and distributed training and mixed precision were disabled.

\noindent\textbf{Evaluation Measures}\enspace
This paper adopts a storage-first perspective. Primary indicators are on-disk model size (KB) and parameter count (K) as proxies for cache residency and working-set compactness, followed by runtime latency and FPS on the target CPU. Segmentation quality is reported with IoU and Dice (mean $\pm$ std over ten runs) to verify that cache-aligned compactness does not degrade accuracy.
\begin{itemize}
\item Intersection over union (IoU) represents the intersection to the union ratio of the predicted and ground truth regions, indicating the degree of overlap between the two. The mathematical definition is as follows
\begin{equation}
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}. 
\end{equation}
\item The Dice coefficient (Dice) is similar to IoU but emphasizes more on the overlap by weighting the intersection area more heavily. The mathematical definition is as
\begin{equation} 
\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}. 
\end{equation}
For IoU and Dice, \( A \) corresponds to the predicted region, whereas \( B \) represents the ground truth region. 
\end{itemize}
In addition to the segmentation metrics, computational efficiency was evaluated with four indicators: the number of parameters, inference time, frames per second (FPS), and the storage footprint.
FPS is computed as $1000/$\,(inference time in ms) under the same CPU setup and measurement protocol.
Unless otherwise specified, the storage footprint denotes the on-disk model size. This paper also reports an FP32 estimate as $\mathrm{Params}\times 4$ bytes for hardware-agnostic comparison.
Inference time was measured on an Intel Core i5-12400F CPU with PyTorch~2.1.0 using a batch size of 1 and inputs of $256\times256$.
The measurement involved 100 warm-up passes followed by 100 timed forward passes, and this paper reports the mean and standard deviation in milliseconds, excluding data loading.
