# From Posts to Patterns: Detecting Anorexia on Reddit

Anorexia is a serious eating disorder characterized by an intense fear of gaining weight and a distorted body image. Identifying individuals suffering from anorexia early on can lead to timely intervention and support. In this repository, we focus on automatically identifying anorexia from social media posts using machine learning techniques.

### Example Post
*''I wasn't even craving foods but still felt like, compelled to binge I suppose? How are you doing today? I was battling the urge this morning but now I feel like I could go a week without eating anything...So weird. Not a recipe, actually! I found it in the freezer section at my grocery store - the brand is 'Golden'. They're ok, not great. Alexia's sweet potato products are SO much better but way higher calorie/fat content :( Another day of high protein for strength training. I've been trying to stick to 800 but I went over my calorie goal yesterday by 200 and need to make up for it today :( it's hard working in an office where all of the snacks I like are so readily accessible. I'm going to distract myself with gum and work, and hope that my willpower will win out.''*

This repository compares different NLP approaches to predict eating disorders through Reddit posts and comments. All models are trained on the **eRisk 2018** and **eRisk 2019** datasets and tested on the **eRisk 2018** testing dataset.

## Table of Contents
- [Approaches](#approaches)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Insights](#key-insights)
- [Performance Comparison](#performance-comparison)
- [Conclusion](#conclusion)
- [Model Overview](#model-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running Inference](#running-inference)
- [Model Weights](#model-weights)
- [Arguments](#arguments)
- [References](#references)

## Approaches

We compare 4 different approaches, with variations, against both naive and advanced models. Two base models — **BERT** and **Longformer** — were evaluated with modifications like the addition of **Multi-headed Attention (MHA)** and **Time Series Transformer (TST)**. We also evaluated models using large language models (LLMs) such as **Llama3 8B**, **Mistral 7B v1**, **Gemma 7B**, **GPT-3.5**, and **MentalLLaMa**.

## Evaluation Metrics
All models are evaluated based on **Precision**, **Recall**, **F1-score**, and **ERDE (Early Risk Detection Error)**. The lower the ERDE, the better. You can find the definition of ERDE in this [paper](https://citius.usc.es/sites/default/files/publicacions_postprints/clef.pdf).

## Key Insights
The repository explores how adding different techniques improves model performance. It was found that incorporating cross-attention between text features enhances model performance. By breaking text into sub-texts, extracting features, and applying cross-attention, it's possible to aggregate features of the entire text into a single matrix.

Posts are also associated with date and time, and time plays a significant role in determining eating disorders. Combining temporal features with text features helps in the early detection of anorexia. Among all evaluated approaches, **Longformer-TST**, i.e., Longformer with Patch-TST and MHA, performed the best.

## Performance Comparison

| **Category**       | **Feature**        | **Threshold** | **Precision** | **Recall**   | **F1-score** | **ERDE_5**    | **ERDE_50**   |
|--------------------|--------------------|---------------|---------------|--------------|--------------|---------------|---------------|
| Tf-IDF             | -                  | 0.4           | 0.828         | 0.707        | 0.763        | 8.72%         | 5.82%         |
| BERT               | -                  | 0.3           | 0.711         | **0.853**    | 0.771        | 6.27%         | 5.31%         |
| BERT               | SA                 | 0.5           | 0.671         | 0.568        | 0.615        | 8.10%         | 5.95%         |
| BERT               | MHA                | 0.2           | 0.778         | **0.853**    | 0.814        | 6.39%         | 5.27%         |
| BERT               | MHA+TST            | 0.6           | 0.846         | 0.804        | 0.825        | 6.34%         | 5.53%         |
| Longformer         | -                  | 0.1           | 0.697         | 0.590        | 0.639        | 7.37%         | 5.48%         |
| Longformer         | SA                 | 0.4           | 0.623         | 0.601        | 0.609        | 7.31%         | 5.19%         |
| Longformer         | MHA                | 0.5           | 0.731         | 0.656        | 0.691        | 6.36%         | 4.53%         |
| Longformer         | MHA+TST            | 0.8           | **0.871**     | 0.831        | **0.852**    | **5.66%**     | **4.05%**     |

### Performance Comparison of Different LLMs in Early Anorexia Detection (Zero Shot Setup)

| **Model**           | **Precision** | **Recall** | **F1-score** | **ERDE_5** |
|---------------------|---------------|------------|--------------|------------|
| LLAMA3 8b           | 0.122         | **0.877**  | 0.214        | 2.378      |
| Mistral 7b v1       | **0.159**     | 0.840      | **0.267**    | 2.040      |
| Gemma 7b Instruct   | 0.105         | 0.894      | 0.187        | 1.722      |
| GPT 3.5             | 0.090         | 0.900      | 0.163        | 2.009      |
| MentaLLaMa          | 0.125         | 0.875      | 0.218        | 2.2516     |
| **Proposed Approach**| **0.871**     | 0.831      | **0.852**    | **5.661%** |


## Conclusion
The Longformer model combined with Patch-TST and Multi-headed Attention (MHA) achieved the best performance, indicating the effectiveness of combining text and temporal features with attention mechanisms for early risk detection of anorexia.

## Model Overview
The model combines:
- **Longformer** for text embeddings, handling long sequences of text.
- **PatchTST** for processing time series data.
- A **Binary Classifier** that takes the outputs from both models and produces the final classification.

The final output is a binary prediction indicating whether a patient has anorexia or not.

## Requirements
To run this project, you'll need the following installed:

- Python 3.8+
- PyTorch 1.10+
- Hugging Face Transformers
- numpy
- argparse
- tqdm
- torch
- torchvision

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/anorexia-prediction.git
    cd anorexia-prediction
    ```

2. Download the pre-trained model weights for PatchTST and the Binary Classifier as explained below.

3. Make sure to update the paths to these models in the arguments when running the script.

## Running Inference
The `inference.py` script takes text input and time series data (entered interactively) and predicts whether the patient has anorexia.

To run the script, use:
```bash
python inference.py --patchtst_path "/path/to/patchtst/model" \
                    --classifier_path "/path/to/classifier" \
                    --input_text "This is a sample input text" \
                    --num_time_stamps 5 \
```
The script will interactively ask for the time series data (hr, min) and return the prediction of whether the patient has anorexia. The input time should be in 24 Hr format.

### Model Weights
You will need to download the following pre-trained model weights:
- **Longformer**: [Download Here](https://huggingface.co/allenai/longformer-base-4096), One does not need to download the checkpoints of longformer explicitly, the code itself downloads it from hugging face.
- **PatchTST**: [Link to model weights] (Provide the URL for downloading PatchTST weights)
- **Binary Classifier**: [Link to model weights] (Provide the URL for downloading Binary Classifier weights)

Save these models locally and provide their paths as arguments when running the script.

## Arguments
Below are the arguments that can be passed to `inference.py`:

| Argument             | Type   | Description                                           | Default Value |
|----------------------|--------|-------------------------------------------------------|---------------|
| `--patchtst_path`     | str    | Path to the PatchTST model.                           | **Required**  |
| `--classifier_path`   | str    | Path to the Binary Classifier model.                  | **Required**  |
| `--input_text`        | str    | Input text for prediction.                            | **Required**  |
| `--num_time_stamps`   | int    | Number of time series stamps to input.                | **Required**  |
| `--threshold`         | float  | Decision threshold for anorexia prediction.           | 0.566         |

## References
- **Longformer**: [AllenAI Longformer Model](https://huggingface.co/allenai/longformer-base-4096)
- **PatchTST**: [Model Link] (Provide URL to PatchTST model)
- **Binary Classifier**: [Model Link] (Provide URL to Binary Classifier model)

