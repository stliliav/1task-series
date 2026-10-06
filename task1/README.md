## 1. Advantages of selected libraries and solutions

The project is built on top of two core deep learning frameworks—**Hugging Face (`transformers`)** and **PyTorch (`torch`, `torchvision`)**—complemented by specialized media processing libraries.

* **Hugging Face (`transformers`)**: Provides a unified, high-level API (`pipeline`, `AutoModel`, `AutoTokenizer`) to interface with state-of-the-art pretrained models. It simplifies model loading, tokenization, and inference, while offering built-in memory management and precision handling.
* **PyTorch & Torchvision**: Offers full low-level control over tensor operations, computational graphs, and hardware acceleration devices (CPU/GPU). Torchvision supplies production-ready, optimized vision architectures such as Faster R-CNN.
* **`librosa` & `soundfile**`: Enables direct audio decoding, array manipulation, and resampling inside Python without requiring a system-level `ffmpeg` installation. This ensures seamless cross-platform execution.
* **Pillow (PIL)**: Provides lightweight raster image loading and color space transformations (converting images to standard RGB format) required by Vision Transformer models.

---

## 2. Principle of operation of models and reasons for selection

### Text (Spam detection) — `prajjwal1/bert-tiny`

* **Operating principle**: A bidirectional Transformer encoder that analyzes contextual relationships of tokens from both directions simultaneously to form sequence embeddings for classification.
* **Reason for selection**: Highly lightweight footprint (~4.4 million parameters) enabling instantaneous inference on CPU while maintaining strong accuracy for short text classification tasks.

### Audio (emotion recognition) — `superb/wav2vec2-base-superb-er`

* **Operating principle**: Operates directly on raw audio waveforms using 1D temporal convolutions to extract feature representations, which are then processed by Transformer layers to capture temporal acoustic patterns.
* **Reason for selection**: Eliminates the overhead of manual spectrogram/MFCC extraction, is robust to background noise, and is natively calibrated for 16 kHz audio signals.

### Image (OCR) — `microsoft/trocr-small-printed`

* **Operating principle**: An Encoder-Decoder architecture combining a Vision Transformer (ViT) encoder that converts image patches into visual embeddings, and a language model decoder (RoBERTa) that generates the corresponding text token sequence.
* **Reason for selection**: Achieves high optical character recognition accuracy on printed text without relying on traditional, complex character segmentation pipelines.

### Video (object detection) — `Faster R-CNN (ResNet-50-FPN)`

* **Operating principle**: A two-stage object detector. A Region Proposal Network (RPN) suggests candidate object regions, followed by RoI Pooling and classification/regression heads that refine bounding box coordinates and assign class probabilities.
* **Reason for selection**: Delivers superior detection accuracy and precise bounding box localization across varied object scales in dynamic video streams.

### Local LLM — `Qwen/Qwen2.5-1.5B-Instruct`

* **Operating principle**: An auto-regressive, decoder-only causal Transformer model utilizing Grouped Query Attention (GQA) to generate responses token-by-token based on context.
* **Reason for selection**: Strikes an optimal balance between instruction-following quality and a compact parameter size (1.5 billion parameters), making full local deployment feasible on standard hardware without external API calls.

---

## 3. Dataset structure (training vs inference)

### Project directory layout for inference

```
project_root/
├── samples/                  # Input test data directory
│   ├── test_audio.wav        # Raw audio waveform (16 kHz)
│   ├── printed_text.png      # Single-line printed text image
│   └── test_video.mp4        # Input video stream (MP4 format)
├── 01_text/                  # Spam detection module
├── 02_audio/                 # Speech emotion recognition module
├── 03_image/                 # TrOCR image recognition module
├── 04_video/                 # Object detection video pipeline
└── 05_llm/                   # Local instruction LLM pipeline

```

### Dataset comparison across tasks

* **Text processing**: Trained on labeled text corpora (SMS/Email paired with binary labels `0` or `1`). Inference receives raw Python `str` inputs.
* **Audio processing**: Trained on speech emotion datasets (such as IEMOCAP) consisting of audio clips and target emotion labels. Inference consumes `.wav` or `.m4a` files converted into 1D `numpy.ndarray` floating-point tensors sampled at 16 kHz.
* **Image processing (OCR)**: Trained on synthetic and real word/line image-text pairs. Inference accepts RGB `PIL.Image` instances transformed into numerical `pixel_values` tensors.
* **Video object detection**: Trained on the MS COCO dataset containing images annotated with bounding box coordinates $[x_{\min}, y_{\min}, w, h]$ and category IDs. Inference processes video frames converted into 4D tensor batches `[Batch, Channels, Height, Width]`.
* **LLM text generation**: Trained on instruction-tuning corpora structured as conversational pairs. Inference receives ChatML-formatted prompt arrays containing role and content definitions.

---

## 4. Metrics for model quality assessment

Model performance and system robustness are monitored through three distinct evaluation tiers:

### Machine Learning metrics

* **Text and audio classification**: Accuracy (overall correctness), Precision (minimizing false positives), Recall (minimizing false negatives), and F1-Score (harmonic mean of Precision and Recall).
* **Optical character recognition (OCR)**: Character Error Rate (CER) and Word Error Rate (WER) to measure string edit distance against ground truth text.
* **Video object detection**: Mean Average Precision (mAP @ IoU=0.5:0.95) to quantify bounding box overlap accuracy across detected object classes.
* **Language modeling (LLM)**: Perplexity (PPL) to assess the model's confidence in predicting subsequent token distributions.

### Product metrics

* **User acceptance rate**: Percentage of generated LLM responses and OCR extractions that meet quality criteria without requiring manual correction.
* **Task completion rate**: Success rate of end-to-end execution across input files without runtime errors.

### Infrastructure metrics

* **Latency**: Time required to process a single text string, audio file, or frame (measured in milliseconds).
* **Throughput**: Frames Per Second (FPS) for video processing, and Tokens Per Second (tokens/sec) for LLM text generation.
* **Resource utilization**: Peak RAM consumption, VRAM usage, and CPU/GPU compute load during active inference.

---

## 5. Implementation features and algorithm efficiency

* **Decoupling from system dependencies**: By loading audio files directly into memory arrays via `librosa`, the audio execution pipeline bypasses reliance on external binary utilities such as `ffmpeg`.
* **Dynamic hardware routing and precision**: The code dynamically detects available hardware compute targets. When CUDA is available, models load in FP16 (`torch.float16`) to minimize VRAM usage. On CPU environments, models use `low_cpu_mem_usage=True` to prevent weight offloading to meta-devices (`meta device`), ensuring stable CPU execution.
* **Sequence control in OCR generation**: The OCR module explicitly passes `max_new_tokens` parameters to the model generation method to avoid early truncation or infinite generation loops on visual feature representations.
* **Modular autonomy**: Each task is implemented as an independent execution script. Model weights are cached locally upon initial download, enabling fully offline operation for subsequent executions.