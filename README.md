This serves as a showcase for one of the projects I completed while I was a Data Analyst Intern at Cognosante (June 2018 - August 2018)

## Background Information
Cognosante employs hundreds of seasonal call center representatives to take calls from customers who have issues regarding their 1095-A forms. They would transcribe/summarize their conversations with the customer and the issue woulde be resolved by other analysts. The goal of this project is to reduce the overhead cost of hiring by automating the process of classifying customer cases so fewer case resolution analysts are needed.

### Methodology
Various classification models were attempted and the final model was a 2 layer LSTM model.

### Challenges
One of the main challenges I ran into was significant class imabalance. Initially, I had to classify 18 cases and among them, several cases had only 1 sample data. I tried to balance the classes using various methods such as over- and under-sampling, gathering more data, and I also tried to use a character level RNN to create more data. In the end, the best solution given the time I had was to regroup and merge some of the classes. As a result, the model changed from classifying 18 classes to 7 classes (shown below).

Case Types and Corresponding Labels 

| Case Label    | Case Type     |
| ------------- |:-------------:|
| 1             | APTC          |
| 2             | Premium       |
| 3             | Fraud         |
| 4             | Appeal        |
| 5             | No 1095       |
| 6             | Coverage      |
| 7             | Period        |

### Final model accuracy
The final accuracy achieved by the model is 97.7%.

### Future improvements
- Better and newer word vectorization/embedding methods such as [GLOVE](https://nlp.stanford.edu/projects/glove/) or [Fasttext](https://fasttext.cc/) can be tested
- a better loss function, [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf), that addresses class imbalance can be tested
- a different model architecture can be tested, in particular [Transformers](https://arxiv.org/abs/1706.03762v5)
- Because of the regrouping of the classes, smaller models can be built to further classify grouped classes to identify subclasses

### Note
All sensitive information and data have been removed.
