Knowledge-Guided Decision Intelligence for Cost-Aware Customer Churn Intervention Under Predictive Uncertainty

Short communication

Dr. Sai Krishna Thota, PhD

Independent Researcher, USA

---

Abstract

Customer churn scores are often treated as if they were already an intervention policy. Existing studies largely optimise prediction or expected profit, while calibrated risk, structured domain knowledge, strategic customer value, predictive uncertainty, human review, and a shared budget are rarely treated as one constrained decision problem. This short communication places a leakage-safe calibrated ensemble under a knowledge-guided allocation layer. A service–contract graph supplies out-of-fold motif risk. IBM customer lifetime value (CLTV) is a dimensionless strategic-priority index; simulated annual margin from monthly charges is the economic value in the utility. Under a common expected-cost budget, most spend goes to positive expected-utility offers and a minority is reserved for human review of high-CLTV cases. On IBM Telco data (N = 7043), ensemble AUROC is 0.860 ± 0.001 and expected calibration error is 0.012 ± 0.003. The prespecified primary decision endpoint is high-value churn reach. Versus cost-aware targeting it rises by 12.6 percentage points (1000/1000 paired bootstrap resamples positive; 95% interval [9.3, 16.5] pp). Overall churn reach stays comparable. The net-benefit proxy rises by $2320 (968/1000 resamples positive; 95% interval [−$92, $5397]) and is not claimed as confirmed economic superiority. Component ablation shows that this coverage gain is produced by the reserved high-CLTV review slice, not by incremental graph-residual routing at the nominal 85/15 operating point. The contribution is a constrained decision-support procedure, not a new classifier.

Keywords: churn prediction; decision support; knowledge graph; predictive uncertainty; cost-sensitive learning; human-in-the-loop

---

1. Introduction

Recent surveys of machine-learning churn research show that predictive pipelines are mature, yet profit-oriented evaluation, operational allocation, and explainable decision support remain comparatively thin [1,2]. Ensemble and boosting classifiers dominate published benchmarks [2,9,10]. Profit-driven ensembles and prescriptive methods that trade expected profit against discrimination, including heterogeneous lifetime value and incentive cost, improve monetary scores relative to accuracy-only training [4,5]. Probability calibration is required if those scores are to be used as decision inputs [11]. Human inspection belongs inside a decision-support system rather than as an afterthought [12]. Knowledge bases and graphs have been used to enrich churn features [3] and, in other domains, to support structured decision-making [13,14].

The scientific gap is more specific than “better churn AUROC.” Existing work largely optimises prediction or expected profit in isolation. Comparatively little work treats calibrated risk, structured domain knowledge, strategic customer value, predictive uncertainty, scarce human review, and a shared intervention budget as one constrained decision problem. Two operational facts make that joint problem necessary. First, economic targeting based on monthly charges is not equivalent to strategic targeting based on lifetime-value knowledge: in the IBM Telco workbook, CLTV and the simulated annual-margin proxy have Pearson correlation 0.10. Second, converting predictive entropy into a blanket review rule can consume the budget with cheap reviews and leave too little capacity for high-utility offers.

This short communication therefore separates prediction from allocation. A calibrated ensemble estimates churn probability. A compact service–contract graph and CLTV encode domain structure. The proposed knowledge-guided decision intelligence (KGDI) policy is budget-matched: majority economic allocation plus a reserved review slice for high-CLTV customers. High-value churn reach under that shared budget is the primary decision endpoint. Overall churn reach and an outcome-anchored net-benefit proxy are secondary. The 12.6 percentage-point coverage gain is interpreted as achievement of that intended strategic objective, not as unexpected discovery of a superior classifier. Only two tables and two figures are reported.

2. Proposed Framework

2.1 Predictive model

Leakage-safe repeated stratified out-of-fold prediction is used (5 folds × 5 repeats). Predictors are an allowlist of service, contract, billing, and tenure fields. Direct leakage fields (churn label, churn score, churn reason) are excluded. Gender and senior-citizen status are excluded from the model and from the policy. Logistic regression, random forest, XGBoost, and LightGBM are calibrated with a sigmoid map fitted on a dedicated calibration split inside each training fold [11]. The ensemble probability p_i is the equal-weight mean of the four calibrated probabilities. Primary uncertainty is the normalised binary entropy of p_i. Between-model disagreement is retained only as a diagnostic. No synthetic oversampling is applied; imbalance is handled through cost-aware utilities [6–8]. Ranking and calibration, not F1, feed Eqs. (1)–(2).

2.2 Knowledge representation

A heterogeneous service–contract graph is built from observed IBM fields. Customer nodes link to contract type, internet service, payment method, and add-on or protection services. Motifs were specified from service–contract semantics that are standard in telecommunications operations—month-to-month fibre, electronic-check payment, unprotected internet, and short tenure—before fold-local rate estimation. They were not selected by scanning outcome labels on the full file. One-hop knowledge risk k_i for a held-out customer is the training-fold empirical churn rate of incident motifs, averaged out-of-fold. Test-fold labels never enter motif rates. Knowledge risk is a structural prior (AUROC 0.81) rather than a replacement for the ensemble (AUROC 0.86; correlation of k_i with p_i = 0.76). The residual k_i − p_i is available as an optional routing flag. This construction is a domain graph for decision gating, not an industrial ontology, a large language-model knowledge base, or a graph-reasoning engine [3,13,14]. Related work on assembling and reusing structured relations in other domains is cited only for that limited precedent [15–20].

IBM CLTV is a customer-node attribute converted to a percentile rank and is never treated as currency. Economic value in the utility is V_i = monthly charges × 12 × 0.60.

2.3 Decision model

Let c_o be offer cost, s the assumed retention-success probability, c_h the review cost, and (Se, Sp) reviewer sensitivity and specificity. Expected offer and review utilities are

U_i^{offer} = p_i s V_i − c_o, (1)

U_i^{review} = p_i Se s V_i − (c_h + c_o [p_i Se + (1 − p_i)(1 − Sp)]). (2)

The planning budget is B = N c_o × 0.20 and is enforced on ex-ante expected selected cost. Three baselines receive the same B: (i) probability-threshold ranking among p_i ≥ 0.5; (ii) cost-aware greedy ranking by U_i^{offer} among p_i ≥ 0.20 and U_i^{offer} > 0; (iii) uncertainty review, which routes high-entropy eligible customers to human review and otherwise offers [4,5,12].

The proposed KGDI policy uses a nominal operating point of 85% of B on cost-aware offers and 15% reserved for human review. That 15% represents a deliberately limited minority review capacity, not a unique optimum. A robustness path over economic shares {0.70, 0.75, 0.80, 0.85, 0.90, 1.00} is shown in Fig. 2. Review eligibility requires p_i ≥ 0.20, CLTV at or above the 75th percentile, positive review utility, and either moderate-or-higher entropy or k_i − p_i > 0. Leftover budget returns to cost-aware offers. Observed churn is not used to choose actions.

Default values are c_o = $100, s = 0.35, c_h = $25, Se = Sp = 0.85. Inference versus cost-aware targeting uses 1000 paired customer bootstrap resamples with policy rerouting and budget reallocation inside every resample. The bootstrap conditions on the previously generated out-of-fold risk estimates and therefore quantifies decision-allocation uncertainty rather than full model-training uncertainty. Reported quantities are the observed delta, a percentile 95% interval, and the fraction of resamples with a positive delta. That fraction is not a classical p-value.

3. Experimental Design

The IBM Telco Customer Churn workbook contains 7043 customers and a 26.5% churn rate. High-value churners, used only for evaluation, are observed churners with CLTV percentile at least 0.75 (n = 365). Because the proposed policy also uses the 0.75 CLTV quantile as a review gate, high-value reach is an intended strategic coverage objective, not an independent surprise metric. All policies share planning budget $140,860. Public telecommunications tables of this type remain a standard test-bed [1,2,7].

Primary decision endpoint: high-value churn reach under the common expected-cost budget. Secondary decision endpoints: overall churn reach and the outcome-anchored net-benefit proxy. Component ablations (CLTV-only review reserve; CLTV plus entropy; CLTV plus graph residual; full KGDI) and scenario perturbations of offer cost, retention success, review cost, reviewer accuracy, and budget fraction are reported in the Results as supplementary analyses, not as additional primary tables. Intervention rate may exceed 20% when cheaper reviews replace offers while expected selected cost remains at B.

4. Results and Discussion

Table 1 shows that the ensemble is well calibrated (Brier 0.130 ± 0.001; expected calibration error 0.012 ± 0.003). Table 1 licences the probabilities in Eqs. (1)–(2). No predictive state of the art is claimed [1,2,7,9]. Logistic regression has a higher F1 than the ensemble; the policy uses ranking and calibration rather than F1.

Table 1
Repeated out-of-fold predictive performance (mean ± sd over five repeats).

Model | AUROC | PR-AUC | F1 | Brier | ECE
Logistic regression | 0.856 ± 0.000 | 0.669 ± 0.002 | 0.616 ± 0.003 | 0.132 ± 0.000 | 0.013 ± 0.003
Random forest | 0.849 ± 0.002 | 0.656 ± 0.008 | 0.563 ± 0.011 | 0.135 ± 0.001 | 0.015 ± 0.002
XGBoost | 0.859 ± 0.001 | 0.679 ± 0.005 | 0.600 ± 0.006 | 0.131 ± 0.000 | 0.011 ± 0.003
LightGBM | 0.851 ± 0.001 | 0.665 ± 0.005 | 0.586 ± 0.006 | 0.134 ± 0.000 | 0.014 ± 0.003
Ensemble | 0.860 ± 0.001 | 0.680 ± 0.005 | 0.601 ± 0.005 | 0.130 ± 0.001 | 0.012 ± 0.003

Notes: Brier and ECE are lower-is-better. ECE: expected calibration error.

Table 2 is the decision result. Cost-aware targeting is the strongest economic baseline among the offer-only policies ($65,209). Uncertainty review raises high-value reach to 54.2% with 1650 reviews, but the proxy falls to $59,253: coverage bought by flooding the budget. Proposed KGDI issues 1297 offers and 166 reviews (intervention rate 20.8% because reviews are cheaper than offers; expected cost remains at B). High-value churn reach is 56.4% versus 43.8% for cost-aware targeting (+12.6 percentage points; 1000/1000 resamples positive; 95% interval [9.3, 16.5] pp). Overall churn reach is essentially unchanged (50.6% versus 49.9%): the policy reallocates, it does not catch more churners in total. The net-benefit proxy is $67,529 (+$2320). In 968/1000 resamples the economic delta was positive, but the 95% interval [−$92, $5397] includes a small negative value, so economic dominance is not confirmed. Relative to uncertainty review, KGDI matches the coverage goal without the review flood. The point-estimate economic increment is about $184 per additional percentage point of high-value coverage versus cost-aware targeting; that ratio is descriptive, not a causal return.

Table 2
Budget-matched decision performance. Deltas are versus cost-aware targeting (1000 paired policy-rerun bootstrap resamples, conditional on frozen OOF scores).

Strategy | Interv. % | Churn reach % | High-value reach % | Offers / reviews | Net-benefit proxy ($) | Δ $ | 95% CI ($) | Resamples Δ$>0 | Δ HV (pp) | Resamples ΔHV>0
Probability threshold | 20.0 | 51.7 | 43.0 | 1408 / 0 | 53,786 | — | — | — | — | —
Cost-aware | 20.0 | 49.9 | 43.8 | 1408 / 0 | 65,209 | — | — | — | — | —
Uncertainty review | 24.5 | 56.4 | 54.2 | 79 / 1650 | 59,253 | — | — | — | — | —
Proposed KGDI | 20.8 | 50.6 | 56.4 | 1297 / 166 | 67,529 | 2320 | [−92, 5397] | 968/1000 | 12.6 | 1000/1000

Notes: Interv.: intervention rate. HV: high-value churn reach. CI: percentile interval. Resample fractions are not classical p-values. High-value Δ 95% interval [9.3, 16.5] pp.

Component ablation under the same 85/15 split shows why the coverage gain occurs. A CLTV-only review reserve (no entropy filter, no graph residual) produces the same 1297 offers, 166 reviews, 50.6% overall reach, 56.4% high-value reach, and $67,529 proxy as full KGDI (0 of 7043 actions differ). Adding entropy does not change that set. CLTV plus graph residual without entropy yields fewer reviews (101) and lower high-value reach (51.8%). Removing the CLTV gate (entropy or residual still allowed) increases reviews to 305 and drops high-value reach to 47.1%. Setting the economic share to 1.00 (no review reserve) returns high-value reach to 44.1%, near cost-aware targeting. The operative mechanism is therefore the reserved high-CLTV review slice. At this nominal point the graph residual is not an incremental routing signal.

The residual was nonetheless tested as a knowledge-conflict hypothesis. Customers with k_i − p_i ≤ 0 have high mean probability (0.59) and observed churn 0.9 percentage points above p_i. Positive-residual terciles have low mean probability (0.11–0.17) and observed churn slightly below p_i. Thus k_i − p_i > 0 does not identify model underestimation on this table. Graph residual and model disagreement are distinct (correlation −0.29; Jaccard overlap of the top 10% sets 0.012). The graph-conflict-only top decile has churn 8.6%, versus 41.6% for disagreement-only, so disagreement—not the residual—marks ambiguous high-risk cases. The graph remains a leak-safe structural prior for visualisation (Fig. 1 marker size) and for stating domain motifs; it is not claimed as an inference engine.

Fig. 1 shows offers at high calibrated risk and human review almost entirely above the 75th CLTV percentile in the moderate-probability band. Fig. 2 is the two-objective plane. High-value reach is 56.4% for economic shares 0.70–0.90 and falls to 44.1% at share 1.00. The 0.85 point is a convenient minority-review operating point on that path, not a uniquely optimal configuration.

Scenario perturbations of offer cost, retention success, review cost, reviewer accuracy, and budget fraction leave the qualitative story intact only in part. The high-value coverage advantage versus cost-aware targeting remained positive in all 11 scenarios examined, but it was small when review cost was $40 (+0.5 pp) or reviewer accuracy was 0.75 (+2.5 pp). Point-estimate economic superiority changed sign in several scenarios. The letter therefore treats strategic coverage under the base scenario as the confirmed finding and treats dollars as interval-qualified and scenario-dependent.

Fig. 1. Knowledge-guided decision map. Horizontal axis: calibrated ensemble churn probability. Vertical axis: CLTV percentile. Marker size encodes graph motif risk. The dotted line is the high-value CLTV quantile (0.75).

Fig. 2. Budget-matched tradeoff between the outcome-anchored net-benefit proxy and high-value churn reach. The path traces alternative economic-allocation fractions.

5. Limitations

Business parameters are simulated. Reviewers are modelled by fixed sensitivity and specificity. One public dataset is used. High-value reach uses the same CLTV threshold that the policy gates on; the result shows that the intended coverage objective is achieved, not that a hidden value segment was discovered. The graph residual does not detect underestimation here and does not change the 0.85 allocation. The bootstrap conditions on frozen OOF scores. External validity requires another cohort or a later time split [1,2].

6. Conclusion

This short communication formulates churn intervention as a constrained decision problem: calibrated risk, a compact service–contract graph, CLTV as strategic priority, scarce review, and a shared budget [3,4,5,11,12,14]. On IBM Telco data, reserving a minority of that budget for high-CLTV review raises high-value churn reach by 12.6 percentage points versus cost-aware targeting (1000/1000 resamples). Overall churn reach stays comparable. The economic proxy is favourable in point estimate (968/1000 resamples) and remains interval-qualified. Ablation attributes the coverage gain to the CLTV review reserve, not to incremental graph-residual routing. That is a decision-support result, not a new classifier.


---

CRediT authorship contribution statement

Sai Krishna Thota: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing – original draft, Writing – review and editing.

Declaration of competing interest

The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

Data availability

The IBM Telco Customer Churn workbook is a public telecommunications benchmark of the same class used in recent peer-reviewed churn studies [1,2,7]. Retention costs, success probabilities, reviewer parameters, margin, horizon, and budget are simulated scenario assumptions. Tables 1–2, Figs. 1–2, and the supplementary ablation, residual, disagreement, and sensitivity files are reproduced from frozen out-of-fold scores by letter_outputs.py and review_analyses.py in https://github.com/drsaikrishnathota1/kbs-churn-decision-intelligence, using results/experiment_a_frozen.

Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Cursor (Grok-assisted drafting) in order to organise the manuscript to the journal Guide for Authors and to check consistency with locked numerical results. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

---

References

[1] Manzoor, A., Qureshi, M. A., Kidney, E., & Longo, L. (2024). A review on machine learning methods for customer churn prediction and recommendations for business practitioners. IEEE Access, 12, 70434–70463. https://doi.org/10.1109/ACCESS.2024.3402092

[2] Imani, M., Joudaki, M., Beikmohammadi, A., & Arabnia, H. R. (2025). Customer churn prediction: A systematic review of recent advances, trends, and challenges in machine learning and deep learning. Machine Learning and Knowledge Extraction, 7(3), 105. https://doi.org/10.3390/make7030105

[3] Shahabikargar, M., Beheshti, A., Mansoor, W., Zhang, X., Foo, E. J., Jolfaei, A., Hanif, A., & Shabani, N. (2025). ChurnKB: A generative AI-enriched knowledge base for customer churn feature engineering. Algorithms, 18(4), 238. https://doi.org/10.3390/a18040238

[4] Jiang, P., Liu, Z., Abedin, M. Z., Wang, J., Yang, W., & Dong, Q. (2024). Profit-driven weighted classifier with interpretable ability for customer churn prediction. Omega, 125, 103034. https://doi.org/10.1016/j.omega.2024.103034

[5] Feng, Y., Yin, Y., Wang, D., Ignatius, J., Cheng, T. C. E., Marra, M., & Guo, Y. (2024). Enhancing e-commerce customer churn management with a profit- and AUC-focused prescriptive analytics approach. Journal of Business Research, 184, 114872. https://doi.org/10.1016/j.jbusres.2024.114872

[6] Rao, C., Xu, Y., Xiao, X., Hu, F., & Goh, M. (2024). Imbalanced customer churn classification using a new multi-strategy collaborative processing method. Expert Systems with Applications, 247, 123251. https://doi.org/10.1016/j.eswa.2024.123251

[7] Wang, C., Rao, C., Hu, F., Xiao, X., & Goh, M. (2024). Risk assessment of customer churn in telco using FCLCNN-LSTM model. Expert Systems with Applications, 248, 123352. https://doi.org/10.1016/j.eswa.2024.123352

[8] Haddadi, S. J., Farshidvard, A., dos Santos Silva, F., dos Reis, J. C., & Reis, M. S. (2024). Customer churn prediction in imbalanced datasets with resampling methods: A comparative study. Expert Systems with Applications, 246, 123086. https://doi.org/10.1016/j.eswa.2023.123086

[9] Joy, U. G., Hoque, K. E., Uddin, M. N., Chowdhury, L., & Park, S.-B. (2024). A big data-driven hybrid model for enhancing streaming service customer retention through churn prediction integrated with explainable AI. IEEE Access, 12, 69130–69150. https://doi.org/10.1109/ACCESS.2024.3401247

[10] Poudel, S. S., Pokharel, S., & Timilsina, M. (2024). Explaining customer churn prediction in telecom industry using tabular machine learning models. Machine Learning with Applications, 17, 100567. https://doi.org/10.1016/j.mlwa.2024.100567

[11] Văduva, A.-G., Oprea, S. V., Niculae, A.-M., Bâra, A., & Andreescu, A.-I. (2024). Improving churn detection in the banking sector: A machine learning approach with probability calibration techniques. Electronics, 13(22), 4527. https://doi.org/10.3390/electronics13224527

[12] Kostopoulos, G., Davrazos, G., & Kotsiantis, S. (2024). Explainable artificial intelligence-based decision support systems: A recent review. Electronics, 13(14), 2842. https://doi.org/10.3390/electronics13142842

[13] Guo, Z., Zhou, D., Yu, D., Zhou, Q., Wu, H., & Hao, A. (2024). An ontology-based method for knowledge reuse in the design for maintenance of complex products. Computers in Industry, 161, 104124. https://doi.org/10.1016/j.compind.2024.104124

[14] Su, C., Jiang, Q., Han, Y., Wang, T., & He, Q. (2025). Knowledge graph-driven decision support for manufacturing process: A graph neural network-based knowledge reasoning approach. Advanced Engineering Informatics, 64, 103098. https://doi.org/10.1016/j.aei.2024.103098

[15] Greif, L., Hauck, S., Kimmig, A., & Ovtcharova, J. (2025). A knowledge graph framework to support life cycle assessment for sustainable decision-making. Applied Sciences, 15(1), 175. https://doi.org/10.3390/app15010175

[16] Park, C., Lee, H., Lee, S., & Jeong, O. (2025). Synergistic joint model of knowledge graph and LLM for enhancing XAI-based clinical decision support systems. Mathematics, 13(6), 949. https://doi.org/10.3390/math13060949

[17] Marandi, S., Hu, Y.-S., & Modarres, M. (2025). Complex system diagnostics using a knowledge graph-informed and large language model-enhanced framework. Applied Sciences, 15(17), 9428. https://doi.org/10.3390/app15179428

[18] Li, J., Qian, L., Liu, P., & Liu, T. (2024). Construction of legal knowledge graph based on knowledge-enhanced large language models. Information, 15(11), 666. https://doi.org/10.3390/info15110666

[19] Yang, Y., Liu, X., Tu, X., Lu, Y., & Wang, Y. (2025). Automating the construction of environmental policy knowledge graph with large language models. Sustainability, 17(22), 10282. https://doi.org/10.3390/su172210282

[20] Zhou, Q., Zhou, D., Wang, Y., Guo, Z., & Dai, C. (2024). Knowledge reuse for ontology modelling and application of maintenance motion state sequence. Journal of Industrial Information Integration, 41, 100659. https://doi.org/10.1016/j.jii.2024.100659
