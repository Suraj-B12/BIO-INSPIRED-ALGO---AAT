# Viva preparation

Team 10 · Bio-Inspired Algorithms · 23DS7BSBIO · Academic Year 2026–2027

Use `results/` for measured evidence. Final results are complete. The measured answers below are generated from `results/final_metrics.json`; its source hash is recorded in `docs/speaker_notes.json`.

1. **What is the optimization problem?** Maximize validation macro-F1 after training EfficientNet-B0 with a decoded configuration. The decision vector has learning rate, weight decay, dropout, head width, unfreezing depth, label smoothing, and augmentation strength.

2. **Why Differential Evolution?** It is a derivative-free population method for continuous spaces, has explicit operators that can be audited, and accommodates our normalized mixed-variable encoding. Suitability is a rationale, not proof that it will outperform every alternative.

3. **What makes the implementation bio-inspired?** It maintains a population and uses mutation, recombination, and survival by fitness. The mathematical difference-vector operation is the mechanism that actually produces candidate changes.

4. **Explain DE/rand/1/bin.** “rand” selects the mutation base randomly, “1” uses one scaled difference vector, and “bin” denotes coordinate-wise binomial crossover.

5. **Write the mutation equation.** `v = x[r1] + F * (x[r2] - x[r3])`, where all three indices differ from each other and from the current target index.

6. **Why must donor indices be distinct?** Reusing indices can collapse the difference vector or create a different algorithm. We test this invariant across population sizes and seeds.

7. **What do F and CR control?** F scales the difference vector; CR controls the probability of taking each mutant coordinate during crossover. Their effects interact with landscape, dimension, and population diversity.

8. **Why use j_rand?** It forces at least one coordinate to come from the mutant even when random crossover masks reject every coordinate. It does not guarantee the numerical trial differs if mutant and parent values coincide.

9. **How are out-of-range coordinates repaired?** For each violated boundary, sample a value between the target coordinate and that boundary. This keeps the result feasible without piling every violation at the boundary through clipping.

10. **Does greedy selection always improve the score?** With a deterministic stored fitness and replacement only when the trial is at least as good, the population's best recorded fitness cannot decrease. With noisy training, that property concerns recorded scores, not an unknowable true generalization score.

11. **Is DE rotation-invariant?** Do not claim that for DE/rand/1/bin. Coordinate-wise binomial crossover depends on the coordinate system, even though difference-vector mutation has useful geometric properties.

12. **How many fitness calls does a run require?** With population size NP and G complete generations, the standard total is `NP * (G + 1)`: initialization also costs NP evaluations. Cache hits may reduce training jobs but remain explicitly logged.

13. **What is the cost per generation?** The optimizer arithmetic is approximately proportional to NP times dimension, while NP training evaluations dominate this application. Fitness evaluation cost must be included when discussing efficiency.

14. **Why use macro-F1?** Each class contributes equally to the mean, preventing the most frequent classes from dominating model selection as they can with raw accuracy. We also report per-class metrics and accuracy rather than relying on a single score.

15. **How are integer and categorical genes handled?** Search occurs in the unit hypercube; decoding maps selected coordinates to the allowed discrete values. Tests check endpoints, range, and that all intended categories are reachable.

16. **Does the optimizer train the neural weights directly?** No. Gradient-based training fits network weights for each candidate recipe. DE chooses hyperparameters and head/fine-tuning architecture choices by evaluating the resulting validation score.

17. **What is fixed in the baseline?** ImageNet-pretrained EfficientNet-B0, frozen backbone, 256-unit head, dropout 0.2, Adam learning rate 0.001, no weight decay or label smoothing, and the common crop-and-flip pipeline without extra RandAugment. The exact registered values live in `config.yaml`.

18. **Why compare random search?** It distinguishes a useful adaptive search strategy from improvement caused merely by trying many configurations. Search space, fidelity, subset, seed, and evaluation budget must match.

19. **What if random search wins?** Report that result, including all planned seeds and uncertainty. Do not weaken the baseline, omit runs, or change the test split to preserve a preferred narrative.

20. **Why use pretrained EfficientNet-B0?** It provides learned visual features and a compact established backbone. The project evaluates how much of that representation to fine-tune; it does not claim that EfficientNet is universally the best plant model.

21. **Why keep frozen BatchNorm layers in evaluation mode?** Otherwise their running statistics can change during training even when their weights do not require gradients, undermining the stated frozen-backbone baseline.

22. **What is proxy fidelity?** A lower-cost training setting using a fixed smaller training subset, resolution, and epoch budget. A proxy score is a search signal, not a final test metric.

23. **How do you check the proxy is useful?** Compare configuration rankings at proxy and higher fidelity, report rank correlation, and rerank distinct top candidates using validation data. A low correlation is evidence to revise fidelity, not hide the check.

24. **How do you avoid test leakage?** Split and group images before model selection; use training/validation only during search; freeze choices before the final evaluation. Repeated decisions based on held-out scores would turn that set into another validation set.

25. **Why group near-duplicates?** Closely related views in training and testing can make evaluation overly optimistic. Group-aware splitting keeps such images together; exact duplicates and corrupted files are logged separately.

26. **What does seed averaging measure?** Variation across repeat training runs with the same protocol and split. It does not measure performance across arbitrary new farms or geographic regions.

27. **What does McNemar's test compare?** The discordant paired correctness outcomes on the same test images: baseline right/optimized wrong versus baseline wrong/optimized right. A small p-value alone says neither how large the gain is nor whether it matters operationally.

28. **What does a bootstrap interval mean here?** It describes metric uncertainty under resampling the evaluated observations. Its interpretation depends on independence; correlated shots may require group-level resampling for an honest uncertainty estimate.

29. **Why report calibration?** A classifier can be accurate but either overconfident or underconfident. ECE and a reliability diagram compare predicted confidence with observed correctness within bins; ECE also depends on binning choices.

30. **What does Grad-CAM prove?** It shows class-conditioned regions influencing a chosen network layer's score. It is an explanatory diagnostic, not a ground-truth lesion mask, causal proof, or guarantee the model ignores the background.

31. **Is a corruption stress test equivalent to field validation?** No. Deterministic image corruptions isolate selected disturbances. Field imagery also changes backgrounds, cameras, symptom stages, multiple stresses, and class distribution.

32. **Can the demo diagnose any plant photo?** The trained label space is tomato-only and closed-set. Unknown species, diseases, and non-leaf images may still receive a confident in-scope label; confidence alone is not a validated rejection mechanism.

33. **Why include analytic benchmark functions?** They test optimizer behavior independently of expensive, noisy CNN training. A known optimum and equal evaluation budget make implementation comparisons interpretable, but do not prove superiority on our application.

34. **How is jDE different from classic DE?** It evolves individual control parameters alongside candidate vectors. SHADE instead uses a history of successful settings, and L-SHADE additionally reduces population size. Only variants actually run can be called experimental results.

35. **How can a stopped search resume reproducibly?** Persist population, evaluated fitnesses, generation/individual position, random-generator state, and cache atomically after each evaluation. Reconstructing only the best candidate loses the search trajectory.

36. **Why run offline?** It makes classroom demonstrations reproducible without relying on network access and demonstrates local inference. Downloading dependencies and weights during setup is distinct from the installed demo's offline operation.

37. **What is the main limitation?** Controlled PlantVillage imagery and a single crop constrain external validity. Good internal metrics do not establish safe autonomous agronomic decisions; independent field evaluation is necessary.

38. **What would you improve next?** Evaluate independently collected field images, validate an unknown-input rejection method, test stronger multi-fidelity search, and measure a multi-objective accuracy/latency trade-off. These are future experiments, not completed claims.

39. **Where can the examiner audit a headline number?** In the corresponding committed result JSON/CSV, its generating script/configuration, and its seed/checkpoint provenance. Report tables and the demo should read those artifacts rather than type results manually.

40. **What is the honest contribution if the final gain is small?** A tested optimizer, a reproducible search/evaluation pipeline, explicit controls, and an evidence-backed account of accuracy, uncertainty, robustness, and computation. A measured neutral result is more useful than an invented improvement.

41. **Can raw loss curves rank baseline and searched recipes directly?** Their cross-entropy objectives use recipe-specific label smoothing, so absolute losses are not the same objective. Compare macro-F1 or accuracy across recipes. Within each method, the plotted mean and standard deviation use only epochs shared by all seed runs; stopped runs are not extrapolated.

42. **Are the system-check runtime estimates guaranteed upper bounds?** No. Synthetic-tensor training throughput excludes data loading, model construction, and other pipeline overhead. It supports resource planning; actual search and training costs must come from the measured experiment logs.

43. **Did DE beat random search in the final experiment?** No superiority was established. Both mean accuracies round to 99.85% and are equal apart from floating-point representation; macro-F1 rounds to 99.80% for DE and 99.80% for random search. Only one search per method was run, followed by three training seeds per chosen recipe.

44. **What are the most important negative results?** DE corruption accuracy is 69.19%, below baseline 85.77%. Its seed-zero clean ECE is 9.23%, above baseline 0.81%. DE is more sensitive to noise and stronger lighting changes despite its better clean classification.

45. **Is DE’s larger calibration error caused by overconfidence?** The dominant clean-data pattern is underconfidence: most DE predictions fall in a confidence bin whose observed accuracy is higher than its confidence. Label smoothing is consistent with less extreme probabilities, but this comparison does not isolate a causal effect.

46. **Did the augmentation gene explain the robustness regression?** We cannot make that causal claim. The selected DE recipe and baseline both use no extra RandAugment, while also differing in fine-tuning, optimizer and other regularization. A controlled ablation on fresh validation data is needed. Random search was not included in the corruption evaluation.

47. **Were the Grad-CAM cases selected because DE looked better?** The grid uses the first held-out example of each of its first six classes in manifest order. Both models classify those examples correctly. Some DE maps are broader or include background; the Late blight case is not a clear localization improvement. The separate prediction gallery includes real DE errors.

48. **What does the actual McNemar result show?** Seed zero has 60 DE-only correct images and 3 baseline-only correct images, with exact two-sided p approximately 9.05e-15. It supports a paired baseline-versus-DE clean-test difference; it is not a DE-versus-random-search test or proof of field validity.

49. **Does a displayed ROC-AUC of 100.00 mean perfect classification?** No. The table rounds percent-scale AUC to two decimals. Ranking and argmax accuracy are different, and saved confusion matrices still contain errors. Consult the unrounded JSON before claiming exact perfection.

50. **Why does the report’s training time differ from the slide’s?** The report cost table shows the mean and sample standard deviation per training seed; the slide sums the three final runs. Search time is separate and excludes reranking unless explicitly labeled otherwise. Forward latency excludes decoding, Grad-CAM, and rendering.

51. **Do pHash groups identify individual plants or capture sessions?** No. They group visually similar content, including transitive near-neighbor relationships. They reduce one leakage mechanism but do not prove plant-identity separation or independence across farms.

52. **Which classes improved and which remain difficult?** Every class has a higher mean F1 under DE than the baseline. Early blight and Target Spot improve most. Early blight remains the weakest DE class; seed-zero errors include Early/Late blight confusion and Early blight predicted as Target Spot.
