# From Posts to Patterns: Detecting Anorexia on Reddit

Anorexia is a serious eating disorder characterized by an intense fear of gaining weight and a distorted body image. Identifying individuals suffering from anorexia early on can lead to timely intervention and support. In this repository, we focus on automatically identifying anorexia from social media posts using machine learning techniques.

### Example Post
*''I wasn't even craving foods but still felt like, compelled to binge I suppose? How are you doing today? I was battling the urge this morning but now I feel like I could go a week without eating anything...So weird. Not a recipe, actually! I found it in the freezer section at my grocery store - the brand is 'Golden'. They're ok, not great. Alexia's sweet potato products are SO much better but way higher calorie/fat content :( Another day of high protein for strength training. I've been trying to stick to 800 but I went over my calorie goal yesterday by 200 and need to make up for it today :( it's hard working in an office where all of the snacks I like are so readily accessible. I'm going to distract myself with gum and work, and hope that my willpower will win out.''*

This repository compares different NLP approaches to predict eating disorders through Reddit posts and comments. All models are trained on the **eRisk 2018** and **eRisk 2019** datasets and tested on the **eRisk 2018** testing dataset.

### Approaches

We compare 4 different approaches, with variations, against both naive and advanced models. Two base models — **BERT** and **Longformer** — were evaluated with modifications like the addition of **Multi-headed Attention (MHA)** and **Time Series Transformer (TST)**. We also evaluated models using large language models (LLMs) such as **Llama3 8B**, **Mistral 7B v1**, **Gemma 7B**, **GPT-3.5**, and **MentalLLaMa**.

### Evaluation Metrics
All models are evaluated based on **Precision**, **Recall**, **F1-score**, and **ERDE (Early Risk Detection Error)**. The lower the ERDE, the better. You can find the definition of ERDE in this [paper](https://citius.usc.es/sites/default/files/publicacions_postprints/clef.pdf).

### Key Insights
The repository explores how adding different techniques improves model performance. It was found that incorporating cross-attention between text features enhances model performance. By breaking text into sub-texts, extracting features, and applying cross-attention, it's possible to aggregate features of the entire text into a single matrix.

Posts are also associated with date and time, and time plays a significant role in determining eating disorders. Combining temporal features with text features helps in the early detection of anorexia. Among all evaluated approaches, **Longformer-TST**, i.e., Longformer with Patch-TST and MHA, performed the best.

### Performance Comparison

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

### Conclusion
The Longformer model combined with Patch-TST and Multi-headed Attention (MHA) achieved the best performance, indicating the effectiveness of combining text and temporal features with attention mechanisms for early risk detection of anorexia.

