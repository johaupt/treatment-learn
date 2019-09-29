# Causal Machine Learning | Uplift Modeling | Heterogeneous Treatment Effects

A comprehensive collection of approaches to estimate individualized treatment effects with a focus on machine learning, known as causal machine learning or uplift modeling.


## Direct Treatment Effect Models

### Linear Additive Treatment Variable (S-Learner)
Include treatment indicator into the model
Advantages:
- Single model
- Interpretable

Disadvantages: 
- Treatment effect typically small relative to other effects
- Model might ignore treatment variable

#### Treatment Interaction Effects (Covariate Transformation)
Include interaction effects between treatment indicator and each covariate


### Outcome Transformation (Modified Outcome Method | Class Variable Transformation | Generalized Weighted Uplift Method)
The transformed outcome (including propensity weights $e(X)$) is:

\\[
Y^*_i = W_i \cdot \frac{Y_i(1)}{e(X_i)} - (1-W_i) \cdot \frac{Y_i(0)}{1-e(X_i)}
\\]
#### Double Robust Estimation 

#### Pollienated transformed-outcome Tree/Forest
Build a forest on the transfored outcome, but replace the leaf estimates with $\bar{Y}(1) - \bar{Y}(0)$.


### Causal Tree
(Rzepakowsk, P., & Jaroszewics, S. (2010, December 14). Decision Trees for Uplift Modeling. https://doi.org/10.1109/ICDM.2010.62)
Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. Proceedings of the National Academy of Sciences, 113(27), 7353–7360. https://doi.org/10.1073/pnas.1510489113

#### Boosted Causal Trees
Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.

#### Generalized Random Forest 
Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. The Annals of Statistics, 47(2), 1148–1178.

#### Bagged Causal MARS
Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.

## Indirect Models | Multi-model Approaches | Metalearners 

### Difference in Conditional Means (K-Model approach | T-Learner | Conditional Mean Regressions)


#### Bayesian Additive Regression Trees
Hill, J. L. (2011). Bayesian Nonparametric Modeling for Causal Inference. Journal of Computational and Graphical Statistics, 20(1), 217–240. https://doi.org/10.1198/jcgs.2010.08162

#### Treatment Residual Neural Network
Farrell, M. H., Liang, T., & Misra, S. (2018). Deep Neural Networks for Estimation and Inference: Application to Causal Effects and Other Semiparametric Estimands. ArXiv E-Prints, arXiv:1809.09953.

#### DragonNet
Correct for violation of the overlap assumption through joint prediction of conditional means and treatment propensity in a multi-output neural network

### Treatment Effect Projection (X-Learner)
Use a single model to estimate the ITE as estimated by any method above. The second-stage model can be a linear regression for interpretability or any single model to replace several models in the first stage. 
Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165.


## Benchmark studies
- Devriendt, F., Moldovan, D., & Verbeke, W. (2018). A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: a stepping stone toward the development of prescriptive analytics. Big Data, 6(1), 13–41. https://doi.org/10.1089/big.2017.0104
- Gubela, R. M., Bequé, A., Gebert, F., & Lessmann, S. (2019). Conversion uplift in e-commerce: A systematic benchmark of modeling strategies. International Journal of Information Technology & Decision Making, 18(3), 747–791.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165.
- Powers, S., Qian, J., Jung, K., Schuler, A., Shah, N. H., Hastie, T., & Tibshirani, R. (2017). Some methods for heterogeneous treatment effect estimation in high-dimensions. CoRR, arXiv:1707.00102v1.
- Wendling, T., Jung, K., Callahan, A., Schuler, A., Shah, N. H., & Gallego, B. (2018). Comparing methods for estimation of heterogeneous treatment effects using observational data from health care databases. Statistics in Medicine, 37, 3309–3324. https://doi.org/10.1002/sim.7820
