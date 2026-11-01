# Literature review and claim register

Plant Disease Detection using Transfer-Learning CNN Optimized with Differential Evolution

Team 10 · Bio-Inspired Algorithms · 23DS7BSBIO · Academic Year 2026–2027

Research checked: 1 November 2026

This review separates published evidence, the project's experimental protocol, and hypotheses that require our own results. Published accuracies below are **not this project's accuracies**. Bibliographic metadata comes from Crossref or DataCite and is independently compared with the BibTeX title, first author, year, and DOI. The DOI evidence cache and `results/citation_check.json` record each verification. A DOI redirect proves the registered identifier resolves; it does not imply the publisher's full text is freely accessible.

## 1. Crop impact and the Indian motivation

Crop disease is a material food-security problem, but numerical losses depend on crop, location, pest pressure, and estimation method. Savary and colleagues estimated global losses associated with pathogens and pests for five major food crops; their estimates range from 17.2% for potato to 30.0% for rice. These are **not tomato-specific losses, not disease-only losses, and not an India-wide loss estimate**. They support the importance of crop health, not a claim that our classifier will recover a specified fraction of farm production. [Savary et al.](https://doi.org/10.1038/s41559-018-0793-y), key `savary2019losses`.

For tomato's Indian production context, the Ministry of Agriculture and Farmers Welfare's final estimate released on 17 March 2026 reports 205.99 lakh tonnes for the 2024–2025 crop year, equivalent to 20.599 million tonnes. This dated official figure motivates studying an economically substantial crop. We deliberately omit a national percentage of tomato yield lost to disease: a suitable primary estimate was not verified. The government's crop year must not be confused with the project's Academic Year 2026–2027. [Official PIB release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2241379&lang=1&reg=3), key `pib2026tomato`; permitted no-DOI government-source exception.

## 2. PlantVillage and what its performance means

Hughes and Salathé introduced an openly accessible collection of curated healthy and diseased leaf images to enable computer-vision research. Mohanty, Hughes, and Salathé then demonstrated deep learning on PlantVillage and reported 99.35% accuracy across 38 crop-condition categories under their experimental setup. This established feasibility on controlled imagery. It does not establish the accuracy of a different model on a different tomato-only split. [Dataset paper](https://doi.org/10.48550/arxiv.1511.08060), `hughes2015dataset`; [classification study](https://doi.org/10.3389/fpls.2016.01419), `mohanty2016plant`.

Our protocol therefore fixes the tomato class mapping and splits before optimization, removes exact duplicates, and keeps perceptually similar groups within one split. These are project controls, rather than claims that the earlier authors used our procedure. PlantVillage provides useful internal evaluation; it should not be described as a representative sample of all field photographs. Images collected with different lighting, backgrounds, cameras, or co-occurring stresses require separate validation. The important scientific outcome is a reproducible comparison on a documented split, including neutral or negative findings.

## 3. CNN disease recognition and domain shift

Ferentinos compared CNN models on a larger plant-disease collection and reported 99.53% accuracy on 17,548 previously unseen images. Too and colleagues compared transfer-learning architectures on the 38-class PlantVillage task and reported 99.75% for DenseNet. Their datasets, label spaces, and evaluation procedures differ from this project's ten-class tomato experiment. These results show that high controlled-dataset performance is possible, not that a proposed optimizer is automatically better. [Ferentinos](https://doi.org/10.1016/j.compag.2018.01.009), `ferentinos2018plant`; [Too et al.](https://doi.org/10.1016/j.compag.2018.03.032), `too2019plant`.

Kamilaris and Prenafeta-Boldú review deep-learning applications across agriculture. PlantDoc supplies non-laboratory images and exposes a different image distribution. Gui and colleagues separately report 99.84% on PlantVillage and 72.03% on their Field-PV evaluation after their proposed adaptations. This is strong evidence for treating field deployment as a distinct evaluation problem. Synthetic blur, noise, lighting, and compression stress tests are useful controlled probes, but cannot substitute for an independently collected field dataset. [Agriculture survey](https://doi.org/10.1016/j.compag.2018.02.016), `kamilaris2018survey`; [PlantDoc](https://doi.org/10.1145/3371158.3371196), `singh2020plantdoc`; [field-recognition study](https://doi.org/10.1016/j.compag.2021.106523), `gui2021field`.

### Related-work comparison

| Work | Model or method | Dataset and reported evaluation scope | Published accuracy | Comparison caveat |
|---|---|---|---:|---|
| Mohanty et al., 2016 | CNN transfer-learning study | PlantVillage, 38 crop-condition classes; author's held-out evaluation | 99.35% | Multi-crop task and original split, not our tomato split |
| Ferentinos, 2018 | CNN architecture comparison | 87,848 images; 58 classes; 17,548 unseen evaluation images | 99.53% | Expanded image collection and label space |
| Too et al., 2019 | DenseNet, compared with other fine-tuned CNNs | PlantVillage, 38 classes; paper-specific test split | 99.75% | Exact split ratio was not established from the accessible abstract; no ratio is invented |
| Gui et al., 2021 | CNN with background/leaf augmentation and additional feature objectives | PlantVillage training; independent Field-PV evaluation | 99.84% on PlantVillage; 72.03% on Field-PV | Field result is the relevant generalization comparison; it is not a tomato-only score |
| This project | EfficientNet-B0 with DE-selected training recipe | Fixed tomato-only split, grouped near-duplicates, held-out final evaluation | Read from `results/final_metrics.json` only | No result is presumed from the literature |

The table is descriptive, not a leaderboard. A fair claim that DE improves our model must come from our fixed baseline, matched training budget, repeated seeds, and paired test predictions.

## 4. EfficientNet, mobile architectures, and transfer depth

EfficientNet introduces compound scaling of network depth, width, and input resolution. EfficientNet-B0 is the fixed backbone required by this project; the study does not claim to invent EfficientNet or repeat its architecture search. MobileNetV2's inverted residuals and linear bottlenecks and MobileNetV3's hardware-aware design are relevant deployment alternatives, but changing to either would change the submitted project scope. [EfficientNet](https://doi.org/10.48550/arxiv.1905.11946), `tan2019efficientnet`; [MobileNetV2](https://doi.org/10.48550/arxiv.1801.04381), `sandler2018mobilenet`; [MobileNetV3](https://doi.org/10.48550/arxiv.1905.02244), `howard2019mobilenet`.

Yosinski and colleagues show that feature transferability depends on layer specificity, co-adaptation, and distance between source and target tasks. This supports evaluating fine-tuning depth rather than assuming that a frozen backbone or a fully trainable backbone is always optimal. In this project, the number of unfrozen feature blocks is an architecture/training decision encoded in the search vector. Its actual benefit remains an empirical question. [Transferability study](https://doi.org/10.48550/arxiv.1411.1792), `yosinski2014transfer`.

## 5. Differential Evolution and adaptation

Storn and Price introduced Differential Evolution as a derivative-free method for real-valued optimization. A scaled population difference supplies a search direction; mutation, crossover, and greedy replacement define the update. The surveys by Das and colleagues explain the broad family of strategies and parameter-control methods. Our implementation uses the explicit DE/rand/1/bin operators, not a library optimizer, so the assessment can inspect each operation. [Original DE](https://doi.org/10.1023/a:1008202821328), `storn1997de`; [2011 survey](https://doi.org/10.1109/tevc.2010.2059031), `das2011survey`; [2016 update](https://doi.org/10.1016/j.swevo.2016.01.004), `das2016recent`.

Brest et al.'s jDE adapts control parameters at the individual level. SHADE uses successful parameter histories, while L-SHADE adds a decreasing population schedule. They motivate benchmark ablations and future work, not an assertion that our fixed-parameter DE is universally best. Our parent-to-bound random repair is an explicitly documented implementation choice. Binomial crossover acts on coordinates: **do not describe the complete DE/rand/1/bin algorithm as rotation-invariant**. Similarly, “few parameters” does not mean parameter-free or immune to noisy fitness. [jDE](https://doi.org/10.1109/tevc.2006.872133), `brest2006jde`; [SHADE](https://doi.org/10.1109/cec.2013.6557555), `tanabe2013shade`; [L-SHADE](https://doi.org/10.1109/cec.2014.6900380), `tanabe2014lshade`.

## 6. Nature-inspired methods: explain the mechanism

Genetic algorithms model selection and variation; PSO maintains particles with position and velocity updates; ant systems construct solutions using learned pheromone information. Simulated annealing uses a temperature-dependent acceptance mechanism inspired by statistical mechanics. ABC, Firefly, GWO, and BBO provide other examples of population search based on foraging, attraction, social organization, or migration. BBO is biogeography-inspired; classifying it as human-inspired would be misleading. The taxonomy is a teaching aid and should acknowledge that categories overlap. [Holland's DOI-bearing 1992 edition](https://doi.org/10.7551/mitpress/1090.001.0001), `holland1992adaptation`; [PSO](https://doi.org/10.1109/icnn.1995.488968), `kennedy1995pso`; [Ant System](https://doi.org/10.1109/3477.484436), `dorigo1996aco`; [simulated annealing](https://doi.org/10.1126/science.220.4598.671), `kirkpatrick1983annealing`.

A metaphor is not evidence of optimization quality. Sörensen's critique argues for scientifically meaningful algorithmic contributions and evaluation. Consequently, the project compares explicit update rules under equal objective-evaluation budgets rather than calling a method superior because its biological story is appealing. [ABC](https://doi.org/10.1007/s10898-007-9149-x), `karaboga2007abc`; [Firefly](https://doi.org/10.1007/978-3-642-04944-6_14), `yang2009firefly`; [GWO](https://doi.org/10.1016/j.advengsoft.2013.12.007), `mirjalili2014gwo`; [BBO](https://doi.org/10.1109/tevc.2008.919004), `simon2008bbo`; [metaphor critique](https://doi.org/10.1111/itor.12001), `sorensen2015metaphors`.

## 7. Hyperparameter optimization and evolutionary CNN design

Random search is the essential control because it measures the value of adaptive search beyond simply trying more configurations. Bayesian optimization builds a surrogate over expensive evaluations; Hyperband allocates resources adaptively among sampled configurations. The original JMLR random-search article is a useful historical pointer, but it is not included as a DOI-bearing citation because no authoritative DOI was established. The report uses DOI-bearing related HPO literature rather than inventing one. [Bayesian HPO](https://doi.org/10.48550/arxiv.1206.2944), `snoek2012bayesian`; [Hyperband preprint](https://doi.org/10.48550/arxiv.1603.06560), `li2016hyperband`.

Four credible examples establish that evolutionary methods can tune learning systems: Real et al. use regularized evolution for architecture search; Wang et al. evolve variable-length CNNs using PSO; a separate Wang et al. paper uses hybrid DE to evolve CNN architecture; Awad et al.'s DEHB combines DE with Hyperband. These methods operate over different spaces and budgets, so their results cannot be attributed to our seven-gene tomato setup. We verified these general CNN/HPO studies rather than padding the review with unverified plant-specific DE claims. [Regularized evolution](https://doi.org/10.1609/aaai.v33i01.33014780), `real2019evolution`; [PSO CNN design](https://doi.org/10.1109/cec.2018.8477735), `wang2018psoarchitecture`; [hybrid DE CNN design](https://doi.org/10.1007/978-3-030-03991-2_24), `wang2018dearchitecture`; [DEHB](https://doi.org/10.24963/ijcai.2021/296), `awad2021dehb`.

## 8. Benchmark functions and the limits of benchmark evidence

The benchmark-function survey by Jamil and Yang provides a source for standard continuous test landscapes. Sphere checks smooth separable convergence; Rosenbrock has a narrow curved valley; Rastrigin and Ackley introduce multimodal structure; Griewank combines a quadratic sum with oscillatory interactions. Formula, domain, dimension, known optimum, random seed, and evaluation budget must all be stated because a function's name alone does not define an experiment. [Benchmark survey](https://doi.org/10.1504/ijmmno.2013.055204), `jamil2013functions`.

For our experiment, the expensive resource is a fitness call. Initial-population evaluations count toward the same budget as later trials. All runs, including failures, contribute to the reported distribution. Median/IQR convergence curves and final-error distributions are more informative than a selected best run. Successful analytic-function optimization validates implementation behavior; it does not prove that DE will improve a noisy neural-network validation score. That second claim requires the actual model-search comparison.

## 9. Statistical evaluation, confidence, and explanations

Dietterich discusses statistical comparison of trained classifiers and the paired structure captured by McNemar's test. Our paired test uses the same held-out images for both classifiers. Bootstrap confidence intervals quantify uncertainty under resampling assumptions; they should not be presented as uncertainty over all farms, cameras, or future distribution shifts. Repeated training seeds address a different source of variability and must be reported separately. [Classifier-comparison study](https://doi.org/10.1162/089976698300017197), `dietterich1998tests`; [bootstrap](https://doi.org/10.1214/aos/1176344552), `efron1979bootstrap`.

A confidence score is not automatically a reliable probability. Guo et al. motivate calibration assessment; Grad-CAM provides class-conditioned visual evidence from gradients and feature maps, not a verified lesion segmentation or causal proof. Hendrycks and Dietterich distinguish common corruptions from adversarial perturbations. These sources support complementary calibration, explanation, and corruption evaluations, each with a clearly limited interpretation. [Calibration](https://doi.org/10.48550/arxiv.1706.04599), `guo2017calibration`; [Grad-CAM](https://doi.org/10.1109/iccv.2017.74), `selvaraju2017gradcam`; [corruption robustness](https://doi.org/10.48550/arxiv.1903.12261), `hendrycks2019corruptions`.

## 10. Applications and deployment case studies

Mrisho and colleagues evaluated PlantVillage Nuru on cassava disease/pest symptoms. Their work describes offline smartphone diagnosis and shows that acquisition protocol, including the number of leaves examined, affects performance. This is a credible deployment case study. It is a cassava study, not a benchmark of this project's tomato classifier, and the app's results should never be shown as our results. [Nuru evaluation](https://doi.org/10.3389/fpls.2020.590889), `mrisho2020nuru`.

For this project, offline educational diagnosis, comparative visualization, and agricultural-extension support are plausible uses; claims of proven farm-level benefit remain future work. DE's broader engineering applications are reviewed by Das et al., while the CNN architecture papers provide directly verified machine-learning examples. We do not claim commercial Plantix performance, user counts, or diagnostic coverage because a suitable DOI-bearing primary evaluation was not verified. A closed ten-class classifier also lacks a reliable “unknown plant/disease” class unless an explicit rejection method has been developed and validated.

## Research gap and defensible contribution

The literature already establishes high accuracy on curated leaf datasets, transfer learning, and evolutionary model design. Our contribution is a transparent, reproducible **evaluation of a fixed EfficientNet-B0 backbone and a mixed seven-gene training/head space on a documented tomato split**, with a pre-registered frozen-backbone baseline and an equal-budget random-search control. Group-aware split integrity, logged evaluations, final repeated-seed evaluation, uncertainty, calibration, and corrupted-image stress tests make the comparison auditable. The contribution can remain valid if DE matches or loses to random search; the conclusion must follow observed evidence. We make no unverified first-ever or state-of-the-art claim.

## Claims we can safely make

- PlantVillage enabled curated image-based plant-disease research: `hughes2015dataset`, `mohanty2016plant`.
- Published controlled-image accuracy does not establish field-photo reliability: `mohanty2016plant`, `singh2020plantdoc`, `gui2021field`.
- EfficientNet studies joint depth/width/resolution scaling: `tan2019efficientnet`.
- Feature transferability varies with layer and task: `yosinski2014transfer`.
- DE uses scaled population differences and selection without objective gradients: `storn1997de`, `das2011survey`.
- jDE, SHADE, and L-SHADE use different parameter/population adaptation mechanisms: `brest2006jde`, `tanabe2013shade`, `tanabe2014lshade`.
- Evolutionary and swarm methods have been applied to CNN architecture/HPO: `real2019evolution`, `wang2018psoarchitecture`, `wang2018dearchitecture`, `awad2021dehb`.
- Calibration, uncertainty, and explanation answer different questions: `guo2017calibration`, `efron1979bootstrap`, `selvaraju2017gradcam`.
- An offline agricultural diagnosis app has been evaluated on cassava: `mrisho2020nuru`.
- **Only our generated result files can establish our model's accuracy, speed, significance, or improvement.**

## Verification and known bibliographic nuances

- Run `python scripts/14_check_citations.py` for the final gate. It checks fresh DOI resolution, registry metadata, and all report citation keys.
- `--metadata-only` is an explicitly limited research-stage mode; it cannot mark the final report gate passed.
- `--offline` rechecks current entries against saved retrieval evidence. It is identified as cached verification in the output.
- The PSO paper's Crossref record omits its publication year. The checker explicitly records IEEE's conference date as supplemental evidence instead of confusing a later DOI deposit date with publication.
- Sörensen's paper appeared online in 2013 and in the 2015 print volume; the bibliography uses the verified print year.
- Hyperband is cited here through its registered 2016 arXiv preprint DOI; it must not be relabeled as the 2018 JMLR article.
- The DOI-bearing Holland reference is the 1992 MIT Press edition, not an invented DOI for the original edition.
- `CITATION_CONTACT_EMAIL` can supply a real contact address for registry requests. Until configured, requests identify the project repository; no personal email is fabricated.
- Every retained reference must be cited in the final report. Unused references should be removed from the final bibliography rather than hidden through a blanket bibliography inclusion.

## Verified bibliography (IEEE-style reference register)

1. **`storn1997de`** — R. Storn, K. Price, “Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces,” *Journal of Global Optimization*, 1997. [DOI: 10.1023/a:1008202821328](https://doi.org/10.1023/a:1008202821328). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

2. **`ferentinos2018plant`** — K. P. Ferentinos, “Deep learning models for plant disease detection and diagnosis,” *Computers and Electronics in Agriculture*, 2018. [DOI: 10.1016/j.compag.2018.01.009](https://doi.org/10.1016/j.compag.2018.01.009). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

3. **`too2019plant`** — E. C. Too, L. Yujian, S. Njuki, L. Yingchun, “A comparative study of fine-tuning deep learning models for plant disease identification,” *Computers and Electronics in Agriculture*, 2019. [DOI: 10.1016/j.compag.2018.03.032](https://doi.org/10.1016/j.compag.2018.03.032). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

4. **`singh2020plantdoc`** — D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, N. Batra, “PlantDoc: A Dataset for Visual Plant Disease Detection,” *Proceedings of the 7th ACM IKDD CoDS and 25th COMAD*, 2020. [DOI: 10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

5. **`savary2019losses`** — S. Savary, L. Willocquet, S. J. Pethybridge, P. Esker, N. McRoberts, A. Nelson, “The global burden of pathogens and pests on major food crops,” *Nature Ecology &amp; Evolution*, 2019. [DOI: 10.1038/s41559-018-0793-y](https://doi.org/10.1038/s41559-018-0793-y). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

6. **`das2016recent`** — S. Das, S. S. Mullick, P. Suganthan, “Recent advances in differential evolution – An updated survey,” *Swarm and Evolutionary Computation*, 2016. [DOI: 10.1016/j.swevo.2016.01.004](https://doi.org/10.1016/j.swevo.2016.01.004). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

7. **`brest2006jde`** — J. Brest, S. Greiner, B. Boskovic, M. Mernik, V. Zumer, “Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical Benchmark Problems,” *IEEE Transactions on Evolutionary Computation*, 2006. [DOI: 10.1109/tevc.2006.872133](https://doi.org/10.1109/tevc.2006.872133). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

8. **`tanabe2013shade`** — R. Tanabe, A. Fukunaga, “Success-history based parameter adaptation for Differential Evolution,” *2013 IEEE Congress on Evolutionary Computation*, 2013. [DOI: 10.1109/cec.2013.6557555](https://doi.org/10.1109/cec.2013.6557555). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

9. **`tanabe2014lshade`** — R. Tanabe, A. S. Fukunaga, “Improving the search performance of SHADE using linear population size reduction,” *2014 IEEE Congress on Evolutionary Computation (CEC)*, 2014. [DOI: 10.1109/cec.2014.6900380](https://doi.org/10.1109/cec.2014.6900380). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

10. **`dorigo1996aco`** — M. Dorigo, V. Maniezzo, A. Colorni, “Ant system: optimization by a colony of cooperating agents,” *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)*, 1996. [DOI: 10.1109/3477.484436](https://doi.org/10.1109/3477.484436). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

11. **`sorensen2015metaphors`** — K. Sörensen, “Metaheuristics—the metaphor exposed,” *International Transactions in Operational Research*, 2015. [DOI: 10.1111/itor.12001](https://doi.org/10.1111/itor.12001). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

12. **`jamil2013functions`** — M. Jamil, X. S. Yang, “A literature survey of benchmark functions for global optimisation problems,” *International Journal of Mathematical Modelling and Numerical Optimisation*, 2013. [DOI: 10.1504/ijmmno.2013.055204](https://doi.org/10.1504/ijmmno.2013.055204). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

13. **`dietterich1998tests`** — T. G. Dietterich, “Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms,” *Neural Computation*, 1998. [DOI: 10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

14. **`selvaraju2017gradcam`** — R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” *2017 IEEE International Conference on Computer Vision (ICCV)*, 2017. [DOI: 10.1109/iccv.2017.74](https://doi.org/10.1109/iccv.2017.74). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

15. **`real2019evolution`** — E. Real, A. Aggarwal, Y. Huang, Q. V. Le, “Regularized Evolution for Image Classifier Architecture Search,” *Proceedings of the AAAI Conference on Artificial Intelligence*, 2019. [DOI: 10.1609/aaai.v33i01.33014780](https://doi.org/10.1609/aaai.v33i01.33014780). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

16. **`yang2009firefly`** — X. S. Yang, “Firefly Algorithms for Multimodal Optimization,” *Stochastic Algorithms: Foundations and Applications*, 2009. [DOI: 10.1007/978-3-642-04944-6_14](https://doi.org/10.1007/978-3-642-04944-6_14). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

17. **`awad2021dehb`** — N. Awad, N. Mallik, F. Hutter, “DEHB: Evolutionary Hyberband for Scalable, Robust and Efficient Hyperparameter Optimization,” *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence*, 2021. [DOI: 10.24963/ijcai.2021/296](https://doi.org/10.24963/ijcai.2021/296). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

18. **`hughes2015dataset`** — D. P. Hughes, M. Salathe, “An open access repository of images on plant health to enable the development of mobile disease diagnostics,” *arXiv*, 2015. [DOI: 10.48550/arxiv.1511.08060](https://doi.org/10.48550/arxiv.1511.08060). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

19. **`tan2019efficientnet`** — M. Tan, Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” *arXiv*, 2019. [DOI: 10.48550/arxiv.1905.11946](https://doi.org/10.48550/arxiv.1905.11946). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

20. **`yosinski2014transfer`** — J. Yosinski, J. Clune, Y. Bengio, H. Lipson, “How transferable are features in deep neural networks?,” *arXiv*, 2014. [DOI: 10.48550/arxiv.1411.1792](https://doi.org/10.48550/arxiv.1411.1792). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

21. **`guo2017calibration`** — C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, “On Calibration of Modern Neural Networks,” *arXiv*, 2017. [DOI: 10.48550/arxiv.1706.04599](https://doi.org/10.48550/arxiv.1706.04599). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

22. **`hendrycks2019corruptions`** — D. Hendrycks, T. Dietterich, “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,” *arXiv*, 2019. [DOI: 10.48550/arxiv.1903.12261](https://doi.org/10.48550/arxiv.1903.12261). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

23. **`snoek2012bayesian`** — J. Snoek, H. Larochelle, R. P. Adams, “Practical Bayesian Optimization of Machine Learning Algorithms,” *arXiv*, 2012. [DOI: 10.48550/arxiv.1206.2944](https://doi.org/10.48550/arxiv.1206.2944). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

24. **`li2016hyperband`** — L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, A. Talwalkar, “Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization,” *arXiv*, 2016. [DOI: 10.48550/arxiv.1603.06560](https://doi.org/10.48550/arxiv.1603.06560). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

25. **`sandler2018mobilenet`** — M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L. C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” *arXiv*, 2018. [DOI: 10.48550/arxiv.1801.04381](https://doi.org/10.48550/arxiv.1801.04381). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

26. **`howard2019mobilenet`** — A. Howard, M. Sandler, G. Chu, L. C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, H. Adam, “Searching for MobileNetV3,” *arXiv*, 2019. [DOI: 10.48550/arxiv.1905.02244](https://doi.org/10.48550/arxiv.1905.02244). Metadata: datacite; DOI resolution and field comparison: `results/citation_check.json`.

27. **`mrisho2020nuru`** — L. M. Mrisho, N. A. Mbilinyi, M. Ndalahwa, A. M. Ramcharan, A. K. Kehs, P. C. McCloskey, H. Murithi, D. P. Hughes, J. P. Legg, “Accuracy of a Smartphone-Based Object Detection Model, PlantVillage Nuru, in Identifying the Foliar Symptoms of the Viral Diseases of Cassava–CMD and CBSD,” *Frontiers in Plant Science*, 2020. [DOI: 10.3389/fpls.2020.590889](https://doi.org/10.3389/fpls.2020.590889). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

28. **`mohanty2016plant`** — S. P. Mohanty, D. P. Hughes, M. Salathé, “Using Deep Learning for Image-Based Plant Disease Detection,” *Frontiers in Plant Science*, 2016. [DOI: 10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

29. **`das2011survey`** — S. Das, P. N. Suganthan, “Differential Evolution: A Survey of the State-of-the-Art,” *IEEE Transactions on Evolutionary Computation*, 2011. [DOI: 10.1109/tevc.2010.2059031](https://doi.org/10.1109/tevc.2010.2059031). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

30. **`kennedy1995pso`** — J. Kennedy, R. Eberhart, “Particle swarm optimization,” *Proceedings of ICNN'95 - International Conference on Neural Networks*, 1995. [DOI: 10.1109/icnn.1995.488968](https://doi.org/10.1109/icnn.1995.488968). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

31. **`simon2008bbo`** — D. Simon, “Biogeography-Based Optimization,” *IEEE Transactions on Evolutionary Computation*, 2008. [DOI: 10.1109/tevc.2008.919004](https://doi.org/10.1109/tevc.2008.919004). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

32. **`mirjalili2014gwo`** — S. Mirjalili, S. M. Mirjalili, A. Lewis, “Grey Wolf Optimizer,” *Advances in Engineering Software*, 2014. [DOI: 10.1016/j.advengsoft.2013.12.007](https://doi.org/10.1016/j.advengsoft.2013.12.007). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

33. **`gui2021field`** — P. Gui, W. Dang, F. Zhu, Q. Zhao, “Towards automatic field plant disease recognition,” *Computers and Electronics in Agriculture*, 2021. [DOI: 10.1016/j.compag.2021.106523](https://doi.org/10.1016/j.compag.2021.106523). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

34. **`kamilaris2018survey`** — A. Kamilaris, F. X. Prenafeta-Boldú, “Deep learning in agriculture: A survey,” *Computers and Electronics in Agriculture*, 2018. [DOI: 10.1016/j.compag.2018.02.016](https://doi.org/10.1016/j.compag.2018.02.016). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

35. **`kirkpatrick1983annealing`** — S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, “Optimization by Simulated Annealing,” *Science*, 1983. [DOI: 10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

36. **`karaboga2007abc`** — D. Karaboga, B. Basturk, “A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm,” *Journal of Global Optimization*, 2007. [DOI: 10.1007/s10898-007-9149-x](https://doi.org/10.1007/s10898-007-9149-x). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

37. **`efron1979bootstrap`** — B. Efron, “Bootstrap Methods: Another Look at the Jackknife,” *The Annals of Statistics*, 1979. [DOI: 10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

38. **`holland1992adaptation`** — J. H. Holland, “Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence,” *MIT Press*, 1992. [DOI: 10.7551/mitpress/1090.001.0001](https://doi.org/10.7551/mitpress/1090.001.0001). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

39. **`wang2018psoarchitecture`** — B. Wang, Y. Sun, B. Xue, M. Zhang, “Evolving Deep Convolutional Neural Networks by Variable-Length Particle Swarm Optimization for Image Classification,” *2018 IEEE Congress on Evolutionary Computation (CEC)*, 2018. [DOI: 10.1109/cec.2018.8477735](https://doi.org/10.1109/cec.2018.8477735). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

40. **`wang2018dearchitecture`** — B. Wang, Y. Sun, B. Xue, M. Zhang, “A Hybrid Differential Evolution Approach to Designing Deep Convolutional Neural Networks for Image Classification,” *AI 2018: Advances in Artificial Intelligence*, 2018. [DOI: 10.1007/978-3-030-03991-2_24](https://doi.org/10.1007/978-3-030-03991-2_24). Metadata: crossref; DOI resolution and field comparison: `results/citation_check.json`.

41. **`pib2026tomato`** — Ministry of Agriculture and Farmers Welfare, “Horticulture area and production estimates,” Press Information Bureau, 17 March 2026. [Official release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2241379&lang=1&reg=3). Accessed 1 November 2026. No-DOI government-source exception.
