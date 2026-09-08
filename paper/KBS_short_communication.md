Knowledge-Guided Decision Intelligence for Strategic-Value-Aware Customer Churn Intervention Under Budget Constraints

Short Communication

Dr. Sai Krishna Thota, PhD

Independent Researcher, USA


## Abstract

Customer churn scores are often treated as if they were already intervention policies. Existing prescriptive churn methods primarily optimise economic targeting or incentives, while strategic customer-value coverage and scarce human review are less often integrated under a shared budget. This short communication develops a leakage-safe calibrated ensemble with a budget-constrained decision layer. Knowledge guidance is operationalised mainly through customer lifetime value (CLTV) and explicit review constraints; a service-contract graph is an auxiliary structural diagnostic. On IBM Telco data (N = 7043), ensemble AUROC is 0.860 ± 0.001 and expected calibration error is 0.012 ± 0.003. Under a common expected-cost budget, the nominal 85/15 strategic-review policy reaches 56.4% of high-value churners versus 43.8% for cost-aware targeting (+12.6 percentage points; 1000/1000 paired bootstrap resamples positive; 95% interval [9.3, 16.5] pp). A matched CLTV-reserved offer-only policy reaches only 44.1%. A post hoc mechanism bootstrap attributes a 12.3 percentage-point gain to review routing (95% interval [8.8, 15.6] pp). The corresponding net-benefit difference is +$2372, but its 95% interval [-$149, $4875] includes zero. Ablation attributes the coverage gain to the high-CLTV review reserve; entropy and graph-residual gates are non-binding at the nominal point. An exact epsilon-constraint benchmark achieves the same strategic reach while retaining 99.68% of the utility optimum versus 93.02% for the fixed-share heuristic. The contribution is a decision-support architecture that separates strategic coverage from economic utility and quantifies the efficiency cost of a simple operational heuristic.

Keywords: churn prediction; decision intelligence; strategic customer value; budget-constrained allocation; human-in-the-loop; knowledge-guided decision support


## 1. Introduction

Recent surveys of machine-learning churn research show that predictive pipelines are mature, yet profit-oriented evaluation, operational allocation, and explainable decision support remain comparatively thin [1,2]. Ensemble and boosting classifiers dominate published benchmarks [2,9,10], while profit-driven and prescriptive approaches improve monetary criteria and can account for heterogeneous lifetime value or incentive cost [4,5]. Prediction-to-retention-policy integration itself is not new: Chu et al. [18] proposed a hybrid churn-and-policy architecture, and recent Knowledge-Based Systems work has developed explainable decision support for strategic customer development [19]. Most directly, Latorre et al. [20] jointly optimise retention targeting and incentive level for profit in a predictor-agnostic prescriptive layer. Probability calibration is required when scores are used as decision inputs [11], and human inspection belongs inside a decision-support system rather than as an afterthought [12]. Knowledge bases and graphs have also been used to enrich churn features [3] and to support structured decisions in other domains [13,14].

The scientific gap is therefore not another increment in churn AUROC, nor simply the move from prediction to policy. Existing prescriptive churn work primarily optimises economic targeting and incentives [4,5,18,20]. Comparatively little work formulates strategic customer-value coverage and economic utility as non-identical objectives under a shared budget with scarce human review, while retaining calibrated risk and auxiliary structured knowledge. Two operational facts motivate that formulation. First, economic targeting based on monthly charges is not equivalent to strategic targeting based on lifetime-value knowledge: in the IBM Telco workbook, CLTV and the simulated annual-margin proxy have Pearson correlation 0.10. Second, routing every uncertain case to review can consume budget without improving strategic allocation. The central question is therefore how to allocate limited intervention and review capacity when economic utility and strategic customer coverage are related but non-identical objectives.

This short communication separates prediction from allocation. A calibrated ensemble estimates churn probability. CLTV supplies strategic-value knowledge, while a compact service-contract graph provides an auxiliary fold-local structural risk diagnostic. Here, "knowledge-guided" refers primarily to the use of CLTV-derived strategic-value knowledge and explicit operational decision constraints; the graph supplies auxiliary structural context rather than being the source of the primary allocation gain. The proposed knowledge-guided decision intelligence (KGDI) policy combines majority economic allocation with a minority review reserve for strategically high-value customers. High-value churn reach under a shared expected-cost budget is the main strategic evaluation endpoint in this analysis; overall churn reach and an outcome-anchored net-benefit proxy are secondary. Because the policy explicitly uses the same CLTV quantile that defines high-value evaluation, the coverage result is interpreted as achievement of an intended strategic objective rather than discovery of a hidden segment. A matched CLTV-reserved offer-only comparator and an exact multi-objective efficiency benchmark are used to test mechanism and efficiency. Only two primary tables and two figures are reported.


## 2. Proposed Framework


### 2.1 Predictive model

Leakage-safe repeated stratified out-of-fold prediction is used (5 folds × 5 repeats). Predictors are restricted to an explicit allowlist of service, contract, billing, and tenure fields. Direct leakage fields (churn label, churn score, churn reason) are excluded; gender and senior-citizen status are excluded from the model and policy and retained only for audit. Logistic regression, random forest, XGBoost, and LightGBM are calibrated with a sigmoid map fitted on a dedicated calibration split inside each training fold [11]. The ensemble probability pᵢ is the equal-weight mean of the four calibrated probabilities. Normalised binary entropy of pᵢ is the primary uncertainty signal; between-model disagreement is retained only as a diagnostic. No synthetic oversampling is applied; imbalance is handled through cost-aware utilities [6-8]. Ranking and calibration, not F1, feed the decision equations.


### 2.2 Knowledge representation

A heterogeneous service-contract graph is built from observed IBM fields. Customer nodes link to contract type, internet service, payment method, and add-on or protection services. Motifs were specified from service-contract semantics-month-to-month fibre, electronic-check payment, unprotected internet, and short tenure-before fold-local rate estimation; they were not selected by scanning full-dataset outcomes. One-hop knowledge risk kᵢ for a held-out customer is the training-fold empirical churn rate of incident motifs, averaged out-of-fold. Test-fold labels never enter motif rates. Knowledge risk is a structural prior (AUROC 0.81) rather than a replacement for the ensemble (AUROC 0.86; correlation of kᵢ with pᵢ = 0.76). The residual kᵢ − pᵢ is evaluated as an optional routing and diagnostic signal; no incremental effect is assumed a priori. This construction is a compact domain graph for structural context, not an industrial ontology, large language-model knowledge base, or graph-reasoning engine [3,13,14]. Related work on assembling and reusing structured relations is cited only for that limited precedent [15-17].

IBM CLTV is a customer-node attribute converted to a percentile rank and is never treated as currency. Economic value in the utility is the simulated margin proxy Vᵢ = monthly charges × 12 × 0.60.


### 2.3 Decision model

Let cₒ be offer cost, s the assumed retention-success probability, cₕ the review cost, and (Se, Sp) reviewer sensitivity and specificity. Expected offer and review utilities are

Uᵢ^offer = pᵢ s Vᵢ − cₒ                                              (1)

Uᵢ^review = pᵢ Se s Vᵢ − (cₕ + cₒ [pᵢ Se + (1 − pᵢ)(1 − Sp)])     (2)

Operationally, a reviewed true churner proceeds to a retention offer with probability Se, whereas a reviewed non-churner receives an unnecessary follow-up offer with probability 1 − Sp; expected review cost therefore includes both the review cost and the probability-weighted downstream offer cost.

The planning budget is B = N cₒ × 0.20 and is enforced on ex-ante expected selected cost. Three primary baselines receive the same B: (i) probability-threshold ranking among pᵢ ≥ 0.5; (ii) cost-aware greedy ranking by expected offer utility among pᵢ ≥ 0.20 and positive expected offer utility; and (iii) uncertainty review, which routes eligible customers with normalised entropy ≥ 0.65 to human review and otherwise offers [4,5,12]. The cost-aware baseline is the primary economic comparator because it directly ranks positive expected offer utility under the same budget.

The proposed KGDI policy uses a nominal operating point of 85% of B on cost-aware offers and 15% reserved for human review. The minority review reserve is an interpretable capacity assumption, not a uniquely optimal split. Review eligibility requires pᵢ ≥ 0.20, CLTV at or above the 75th percentile, positive review utility, and either normalised entropy ≥ 0.45 or kᵢ − pᵢ > 0. Eligible review candidates are ranked by expected review utility divided by expected review cost. Any leftover budget returns to cost-aware offers. A robustness path over economic shares {0.70, 0.75, 0.80, 0.85, 0.90, 1.00} is evaluated. Observed churn is never used to choose actions.

Default values are cₒ = $100, s = 0.35, cₕ = $25, Se = Sp = 0.85. Inference versus cost-aware targeting uses 1000 paired customer bootstrap resamples with policy rerouting and budget reallocation inside every resample. The bootstrap conditions on previously generated out-of-fold risk estimates and therefore quantifies decision-allocation uncertainty rather than full model-training uncertainty. Reported quantities are observed deltas, percentile 95% intervals, and the fraction of resamples with a positive delta; those fractions are not classical p-values.


## 3. Experimental Design

The IBM Telco Customer Churn workbook contains 7043 customers and a 26.5% churn rate. High-value churners, used only for evaluation, are observed churners with CLTV percentile at least 0.75 (n = 365). Because the proposed policy uses the same 0.75 CLTV quantile as a review gate, high-value reach is an intended strategic coverage objective, not an independent surprise metric. All primary policies share planning budget $140,860. The main strategic evaluation endpoint is high-value churn reach under the common expected-cost budget; overall churn reach and the outcome-anchored net-benefit proxy are secondary. Action rate may exceed 20% when cheaper reviews replace offers while expected selected cost remains at B. Public telecommunications tables of this type remain a standard test-bed [1,2,7].

Two post hoc secondary analyses probe mechanism and efficiency without refitting the predictive models. First, a matched CLTV-reserved offer-only comparator uses the same 85/15 budget structure: 85% is allocated by the cost-aware rule, while the reserve is spent on eligible high-CLTV direct offers rather than human review; leftover budget again returns to cost-aware offers. This isolates whether review routing adds value beyond simply reserving budget for CLTV. Second, an exact binary epsilon-constraint programme is applied to the frozen out-of-fold scores to trace the efficient frontier between expected economic utility and expected high-value churn-risk mass reached, subject to the same planning budget and one action per customer. Observed churn labels are excluded from that optimisation and used only for ex-post evaluation. These analyses are mechanism and efficiency checks rather than confirmatory redefinitions of the primary analysis.


## 4. Results and Discussion

Table 1 shows that the ensemble is well calibrated (Brier 0.130 ± 0.001; expected calibration error 0.012 ± 0.003) and therefore supports using these probabilities in Eqs. (1)-(2). No predictive state of the art is claimed [1,2,7,9]. Logistic regression has a higher F1 than the ensemble; the policy uses ranking and calibration rather than F1.

Table 1. Repeated out-of-fold predictive performance (mean ± sd over five repeats).

Notes: Brier and ECE are lower-is-better. ECE: expected calibration error.

Table 2 summarises the decision results. Cost-aware targeting is the strongest primary economic baseline among offer-only policies ($65,209). Uncertainty review raises high-value reach to 54.2% with 1650 reviews, but the proxy falls to $59,253, showing the cost of indiscriminate review capacity. Proposed KGDI issues 1297 offers and 166 reviews (20.8% action rate because reviews are cheaper than offers; expected selected cost remains within B). High-value churn reach is 56.4% versus 43.8% for cost-aware targeting (+12.6 percentage points; 1000/1000 primary bootstrap resamples positive; 95% interval [9.3, 16.5] pp). Overall churn reach is essentially unchanged (50.6% versus 49.9%). The net-benefit proxy is $67,529 (+$2320), but its 95% interval [-$92, $5397] includes a small negative value, so economic dominance is not confirmed. The matched CLTV-reserved offer-only comparator reaches only 44.1% high-value churn at $65,157 net benefit. Thus strategic reservation by itself changes little relative to cost-aware targeting. In a paired policy-rerun bootstrap for the mechanism comparison between the CLTV-only review-reserve policy-which is action-identical to full KGDI at the nominal point-and the matched CLTV-reserved offer-only policy, high-value reach improves by 12.3 percentage points (1000/1000 resamples positive; 95% interval [8.8, 15.6] pp), while the net-benefit difference is +$2372 (967/1000 positive; 95% interval [-$149, $4875]). Under the assumed reviewer sensitivity, specificity, and review cost, review-based triage therefore converts the strategic reserve into substantially greater high-value coverage than direct offers; the effect should not be interpreted as reviewer intelligence independent of those modelling assumptions.

Table 2. Budget-matched decision performance. Primary deltas for KGDI are versus cost-aware targeting; the CLTV-reserved offer-only row is a post hoc matched mechanism comparator.

Notes: Action %: proportion assigned either a retention offer or human review. HV: high-value churn reach. CI: percentile interval. Resample fractions are not classical p-values. Primary KGDI versus cost-aware HV-delta 95% interval: [9.3, 16.5] pp. Matched CLTV-review versus CLTV-offer-only HV-delta interval: [8.8, 15.6] pp.

Component ablation under the same 85/15 split identifies the operative mechanism. A CLTV-only review reserve (no entropy filter and no graph residual) produces exactly the same 1297 offers, 166 reviews, 50.6% overall churn reach, 56.4% high-value reach, and $67,529 proxy as full KGDI (0 of 7043 actions differ). Adding entropy does not change that set. CLTV plus graph residual without entropy yields fewer reviews (101) and lower high-value reach (51.8%). Removing the CLTV gate increases reviews to 305 and drops high-value reach to 47.1%. Setting the economic share to 1.00 returns high-value reach to 44.1%, near cost-aware targeting. The empirical coverage gain is therefore attributable to the reserved high-CLTV review slice, not to incremental graph-residual routing at the nominal point. The graph residual also fails the model-underestimation hypothesis on this table: positive-residual terciles have low mean predicted probability (0.11-0.17) and observed churn slightly below pᵢ. Residual and model disagreement are distinct (correlation -0.29; top-decile Jaccard overlap 0.012), but disagreement-not the residual-marks the higher-risk ambiguous cases. The graph is therefore retained as a leak-safe structural diagnostic rather than claimed as the source of the decision gain.

Expected utility is the ex-ante optimisation objective constructed from calibrated risk and simulated action parameters; the outcome-anchored net-benefit proxy is a separate ex-post descriptive quantity computed after actions are fixed using observed churn labels. The exact epsilon-constraint benchmark provides an efficiency envelope for the heuristic. The utility-optimal endpoint has expected utility 64,016 and reaches 47.58% of expected high-value churn-risk mass. The maximum-strategic endpoint reaches 53.48% of that mass while retaining 99.68% of the utility optimum; its ex-post high-value churn reach is 56.4% and its outcome-anchored net-benefit proxy is $69,280. The 85/15 heuristic reaches the same 53.48% expected high-value risk mass and the same 56.4% observed high-value churn reach, but retains only 93.02% of the utility optimum and yields $67,529 ex-post net benefit. At the same strategic target, the exact endpoint therefore retains 6.66 percentage points more of the utility optimum (expected utility 63,809 versus 59,549), an approximately $4260 expected-utility gap under the study's monetary utility formulation. Thus the simple reserve rule is operationally interpretable but not Pareto-efficient: the strategic target is achievable with materially better expected-utility allocation when exact optimisation is available. Across 11 scenario perturbations of offer cost, retention success, review cost, reviewer accuracy, and budget fraction, the high-value coverage advantage remains positive, although it narrows when review is expensive or reviewer accuracy is low; point-estimate economic superiority changes sign in several scenarios. As a reviewer-skill sanity check, setting Se = Sp = 0.50 eliminates all review selections: the policy reverts to 1408 direct offers, 49.9% overall churn reach, 43.8% high-value churn reach, and a $65,209 net-benefit proxy. This chance-level check shows, within the assumed reviewer model, that the strategic-review gain depends on informative reviewer performance rather than on budget reservation alone.

Fig. 1 shows offers concentrated at higher calibrated risk, with human review almost entirely above the 75th CLTV percentile. Marker size encodes graph motif risk as an auxiliary structural diagnostic, not as the allocation engine. Fig. 2 reports the post hoc efficiency benchmark on frozen out-of-fold scores. The 85/15 strategic-review heuristic reaches the same maximum strategic-risk coverage shown by the exact endpoint, but with lower expected-utility retention.

Fig. 1. Strategic-value-guided decision map. Horizontal axis: calibrated ensemble churn probability. Vertical axis: CLTV percentile. Marker size encodes graph motif risk as an auxiliary structural diagnostic. The dotted line is the high-value CLTV quantile (0.75).

Fig. 2. Post hoc efficiency benchmark on frozen out-of-fold scores. The curve is the exact epsilon-constraint frontier between expected high-value churn-risk mass reached and expected utility retained relative to the utility optimum. The 85/15 strategic-review heuristic reaches the same maximum strategic-risk coverage shown by the exact endpoint but with lower expected-utility retention. Observed churn labels are not used in the optimisation.


## 5. Limitations

Business parameters are simulated, reviewers are represented by fixed sensitivity and specificity, and only one public cohort is analysed. High-value reach uses the same CLTV threshold that the policy gates on; the result therefore demonstrates achievement of a declared strategic coverage objective rather than discovery of an unknown value segment. The matched CLTV-offer analysis and the epsilon-constraint benchmark are post hoc secondary analyses and should be interpreted as mechanism and efficiency checks. The graph residual is non-binding at the nominal operating point and does not detect model underestimation on this dataset. The policy-rerun bootstrap conditions on frozen out-of-fold scores and therefore does not include full model-training uncertainty. The 85/15 heuristic is not Pareto-efficient relative to the exact optimisation benchmark. The chance-level reviewer sanity check further shows that the strategic-review advantage is conditional on reviewer skill as represented by the assumed sensitivity, specificity, and review cost. Reviewer performance, intervention success, customer value, and economic costs would require empirical estimation in a deployment setting. External validity requires another cohort or a temporal validation split [1,2].


## 6. Conclusion

This short communication formulates customer-churn intervention as a constrained decision problem rather than a classification endpoint. A calibrated ensemble supplies risk; CLTV supplies strategic-value knowledge; scarce human review and a shared expected-cost budget define the operational decision space; and a compact service-contract graph supplies auxiliary structural context [3-5,11,12,14]. On IBM Telco data, the 85/15 strategic-review policy raises high-value churn reach from 43.8% under cost-aware targeting to 56.4% while leaving overall churn reach essentially unchanged. A matched CLTV-reserved offer-only policy reaches only 44.1%, showing that the strategic gain is not explained by CLTV reservation alone; under the stated reviewer model, review routing is the operative mechanism. The economic point estimate favours the strategic-review policy but remains interval-qualified. Ablation further shows that entropy and graph-residual gates do not add incremental routing at the nominal point. Finally, the exact epsilon-constraint benchmark reaches the same strategic coverage with substantially better expected-utility retention, establishing both the value of strategic review and the efficiency limits of the simple fixed-share heuristic. The contribution is therefore a transparent budget-constrained decision-support architecture and mechanism analysis, not a new classifier, a claim of graph reasoning, or a claim of economic optimality.


## Data Availability

The IBM Telco Customer Churn sample is a public IBM Cognos Analytics benchmark describing a fictional telecommunications company and 7043 customers; IBM documents the sample, including churn status and customer lifetime value, in its official sample materials [21]. The raw workbook is not redistributed in the companion code repository. Analysis code is available at https://github.com/drsaikrishnathota1/kbs-churn-decision-intelligence. Reproduction requires obtaining the IBM sample data and running main.py to generate the leakage-safe out-of-fold scores. experiment_c.py, letter_outputs.py, review_analyses.py, and experiment_b.py regenerate the strategic-review, manuscript, mechanism, and epsilon-constraint efficiency analyses from those frozen scores. Generated result directories are intentionally excluded from version control. The supplementary reproducibility package provided with this manuscript contains the frozen run configuration and result manifest, primary table CSVs, policy/ablation/sensitivity outputs, the matched-mechanism summary, the reviewer-skill sanity check, and Experiment B frontier and utility-retention CSVs.


## Declaration of Competing Interest

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this manuscript.


## Declaration of Generative AI and AI-Assisted Technologies

During preparation of this manuscript, AI-assisted tools were used for language drafting, editing, organisation, and manuscript formatting support. The author reviewed and edited the content and takes full responsibility for the final manuscript. The experimental data, tables, and figures were generated from executable simulation code and reproducible CSV outputs.


## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.


## CRediT Authorship Contribution Statement

Sai Krishna Thota: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review and editing, Visualization.


## References

[1] Manzoor, A., Qureshi, M. A., Kidney, E., & Longo, L. (2024). A review on machine learning methods for customer churn prediction and recommendations for business practitioners. IEEE Access, 12, 70434-70463. https://doi.org/10.1109/ACCESS.2024.3402092

[2] Imani, M., Joudaki, M., Beikmohammadi, A., & Arabnia, H. R. (2025). Customer churn prediction: A systematic review of recent advances, trends, and challenges in machine learning and deep learning. Machine Learning and Knowledge Extraction, 7(3), 105. https://doi.org/10.3390/make7030105

[3] Shahabikargar, M., Beheshti, A., Mansoor, W., Zhang, X., Foo, E. J., Jolfaei, A., Hanif, A., & Shabani, N. (2025). ChurnKB: A generative AI-enriched knowledge base for customer churn feature engineering. Algorithms, 18(4), 238. https://doi.org/10.3390/a18040238

[4] Jiang, P., Liu, Z., Abedin, M. Z., Wang, J., Yang, W., & Dong, Q. (2024). Profit-driven weighted classifier with interpretable ability for customer churn prediction. Omega, 125, 103034. https://doi.org/10.1016/j.omega.2024.103034

[5] Feng, Y., Yin, Y., Wang, D., Ignatius, J., Cheng, T. C. E., Marra, M., & Guo, Y. (2024). Enhancing e-commerce customer churn management with a profit- and AUC-focused prescriptive analytics approach. Journal of Business Research, 184, 114872. https://doi.org/10.1016/j.jbusres.2024.114872

[6] Rao, C., Xu, Y., Xiao, X., Hu, F., & Goh, M. (2024). Imbalanced customer churn classification using a new multi-strategy collaborative processing method. Expert Systems with Applications, 247, 123251. https://doi.org/10.1016/j.eswa.2024.123251

[7] Wang, C., Rao, C., Hu, F., Xiao, X., & Goh, M. (2024). Risk assessment of customer churn in telco using FCLCNN-LSTM model. Expert Systems with Applications, 248, 123352. https://doi.org/10.1016/j.eswa.2024.123352

[8] Haddadi, S. J., Farshidvard, A., dos Santos Silva, F., dos Reis, J. C., & Reis, M. S. (2024). Customer churn prediction in imbalanced datasets with resampling methods: A comparative study. Expert Systems with Applications, 246, 123086. https://doi.org/10.1016/j.eswa.2023.123086

[9] Joy, U. G., Hoque, K. E., Uddin, M. N., Chowdhury, L., & Park, S.-B. (2024). A big data-driven hybrid model for enhancing streaming service customer retention through churn prediction integrated with explainable AI. IEEE Access, 12, 69130-69150. https://doi.org/10.1109/ACCESS.2024.3401247

[10] Poudel, S. S., Pokharel, S., & Timilsina, M. (2024). Explaining customer churn prediction in telecom industry using tabular machine learning models. Machine Learning with Applications, 17, 100567. https://doi.org/10.1016/j.mlwa.2024.100567

[11] Văduva, A.-G., Oprea, S. V., Niculae, A.-M., Bâra, A., & Andreescu, A.-I. (2024). Improving churn detection in the banking sector: A machine learning approach with probability calibration techniques. Electronics, 13(22), 4527. https://doi.org/10.3390/electronics13224527

[12] Kostopoulos, G., Davrazos, G., & Kotsiantis, S. (2024). Explainable artificial intelligence-based decision support systems: A recent review. Electronics, 13(14), 2842. https://doi.org/10.3390/electronics13142842

[13] Guo, Z., Zhou, D., Yu, D., Zhou, Q., Wu, H., & Hao, A. (2024). An ontology-based method for knowledge reuse in the design for maintenance of complex products. Computers in Industry, 161, 104124. https://doi.org/10.1016/j.compind.2024.104124

[14] Su, C., Jiang, Q., Han, Y., Wang, T., & He, Q. (2025). Knowledge graph-driven decision support for manufacturing process: A graph neural network-based knowledge reasoning approach. Advanced Engineering Informatics, 64, 103098. https://doi.org/10.1016/j.aei.2024.103098

[15] Greif, L., Hauck, S., Kimmig, A., & Ovtcharova, J. (2025). A knowledge graph framework to support life cycle assessment for sustainable decision-making. Applied Sciences, 15(1), 175. https://doi.org/10.3390/app15010175

[16] Park, C., Lee, H., Lee, S., & Jeong, O. (2025). Synergistic joint model of knowledge graph and LLM for enhancing XAI-based clinical decision support systems. Mathematics, 13(6), 949. https://doi.org/10.3390/math13060949

[17] Marandi, S., Hu, Y.-S., & Modarres, M. (2025). Complex system diagnostics using a knowledge graph-informed and large language model-enhanced framework. Applied Sciences, 15(17), 9428. https://doi.org/10.3390/app15179428

[18] Chu, B.-H., Tsai, M.-S., & Ho, C.-S. (2007). Toward a hybrid data mining model for customer retention. Knowledge-Based Systems, 20(8), 703-718. https://doi.org/10.1016/j.knosys.2006.10.003

[19] Abbaspour Onari, M., Jahangoshai Rezaee, M., Saberi, M., & Nobile, M. S. (2024). An explainable data-driven decision support framework for strategic customer development. Knowledge-Based Systems, 295, 111761. https://doi.org/10.1016/j.knosys.2024.111761

[20] Latorre, P., Meza, A., López-Ospina, H., Verbeke, W., & Pérez, J. (2025). A prescriptive analytics framework for jointly optimizing retention incentives and targeting. Knowledge-Based Systems, 330, 114649. https://doi.org/10.1016/j.knosys.2025.114649

[21] IBM Cognos Analytics Samples Team. (2019). Telco customer churn (11.1.3+). IBM Community. https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
