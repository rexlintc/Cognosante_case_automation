## Background Information
Case Types and Corresponding Labels
|*Case Labels*|*Case Types*|
|    :---:    |   :---:    |
|      1      |    APTC    |
|      2      |   Premium  |
|      3      |   Fraud    |
|      4      |   Appeal   |
|      5      |  No 1095   |
|      6      |  Coverage  |
|      7      |  Period    |

### Parameters to Tune

#### Hidden Layers
- dropout
- neurons/nodes

#### Training Parameters
- num epochs
- batch size

What has been done and tested.
- synthetically create data to balance all 7 classes but the results were not good

What else could be done
- Word2Vec or try different vectorization methods to try to capture as much information as possible from text data
- Generate different models for specific cases to reduce the severity of class imbalance
- Use [SMOTE](http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html) to balance dataset
