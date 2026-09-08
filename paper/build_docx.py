#!/usr/bin/env python3
"""Build an Elsevier-style single-column, double-spaced Word manuscript."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "KBS_short_communication.docx"
FIG1 = ROOT / "Figure_1.png"
FIG2 = ROOT / "Figure_2.png"


def set_run_font(run, size=12, bold=False, italic=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def set_paragraph_format(p, *, after=0, before=0, first_line=True, align="left", space=2.0):
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.space_before = Pt(before)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = space
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "justify":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = Pt(18) if first_line else Pt(0)


def add_text(doc, text, *, size=12, bold=False, italic=False, after=0, before=0,
             first_line=True, align="justify", space=2.0):
    p = doc.add_paragraph()
    set_paragraph_format(p, after=after, before=before, first_line=first_line, align=align, space=space)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_h1(doc, text):
    p = doc.add_heading(text, level=1)
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)
    pf.space_before = Pt(6)
    for run in p.runs:
        set_run_font(run, size=12, bold=True)
    return p


def add_h2(doc, text):
    p = doc.add_heading(text, level=2)
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 2.0
    pf.space_after = Pt(0)
    pf.space_before = Pt(4)
    for run in p.runs:
        set_run_font(run, size=12, bold=True)
    return p


def shade_header(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "D9E2F3")
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_cell_border(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "666666")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def fill_table(doc, headers, rows, col_widths):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_run_font(run, size=8, bold=True)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        shade_header(cell)
        set_cell_border(cell)
        cell.width = Inches(col_widths[i])
    for r_i, row in enumerate(rows, start=1):
        for c_i, val in enumerate(row):
            cell = table.rows[r_i].cells[c_i]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            set_run_font(run, size=8)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            set_cell_border(cell)
            cell.width = Inches(col_widths[c_i])
    return table


def add_caption(doc, text):
    p = doc.add_paragraph()
    set_paragraph_format(p, after=4, before=2, first_line=False, align="left", space=1.0)
    run = p.add_run(text)
    set_run_font(run, size=10, italic=True)


def add_picture(doc, path, width=6.3):
    p = doc.add_paragraph()
    set_paragraph_format(p, after=0, before=2, first_line=False, align="center", space=1.0)
    p.add_run().add_picture(str(path), width=Inches(width))


def word_count(doc):
    return sum(len(p.text.split()) for p in doc.paragraphs)


def build():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style.paragraph_format.line_spacing = 2.0

    add_text(
        doc,
        "Knowledge-Guided Decision Intelligence for Strategic-Value-Aware Customer Churn Intervention Under Budget Constraints",
        size=14, bold=True, first_line=False, align="center", after=6,
    )
    add_text(doc, "Short Communication", size=12, first_line=False, align="center", after=12)
    add_text(doc, "Dr. Sai Krishna Thota, PhD*", size=12, first_line=False, align="center", after=0)
    add_text(
        doc,
        "Independent Researcher, United States",
        size=12, first_line=False, align="center", after=0,
    )
    add_text(
        doc,
        "* Corresponding author. E-mail: drsaikrishnathota@ieee.org",
        size=11, italic=True, first_line=False, align="center", after=12, space=1.15,
    )

    add_h1(doc, "Abstract")
    add_text(
        doc,
        "Customer churn scores are often treated as if they were already intervention policies. Existing prescriptive churn methods primarily optimise economic targeting or incentives, while strategic customer-value coverage and scarce human review are less often integrated under a shared budget. This short communication develops a leakage-safe calibrated ensemble with a budget-constrained decision layer. Knowledge guidance is operationalised mainly through customer lifetime value (CLTV) and explicit review constraints; a service-contract graph is an auxiliary structural diagnostic. On IBM Telco data (N = 7043), ensemble AUROC is 0.860 ± 0.001 and expected calibration error is 0.012 ± 0.003. Under a common expected-cost budget, the nominal 85/15 strategic-review policy reaches 56.4% of high-value churners versus 43.8% for cost-aware targeting (+12.6 percentage points; 1000/1000 paired bootstrap resamples positive; 95% interval [9.3, 16.5] pp). A matched CLTV-reserved offer-only policy reaches only 44.1%. A post hoc mechanism bootstrap attributes a 12.3 percentage-point gain to review routing (95% interval [8.8, 15.6] pp). The corresponding net-benefit difference is +$2372, but its 95% interval [-$149, $4875] includes zero. Ablation attributes the coverage gain to the high-CLTV review reserve; entropy and graph-residual gates are non-binding at the nominal point. An exact epsilon-constraint benchmark achieves the same strategic reach while retaining 99.68% of the utility optimum versus 93.02% for the fixed-share heuristic. The contribution is a decision-support architecture that separates strategic coverage from economic utility and quantifies the efficiency cost of a simple operational heuristic.",
        first_line=False,
    )
    p = doc.add_paragraph()
    set_paragraph_format(p, after=0, before=6, first_line=False, align="left")
    r0 = p.add_run("Keywords: ")
    set_run_font(r0, size=12, bold=True)
    r1 = p.add_run(
        "churn prediction; decision intelligence; strategic customer value; budget-constrained allocation; human-in-the-loop; knowledge-guided decision support"
    )
    set_run_font(r1, size=12, bold=False)

    add_h1(doc, "1. Introduction")
    add_text(
        doc,
        "Recent surveys of machine-learning churn research show that predictive pipelines are mature, yet profit-oriented evaluation, operational allocation, and explainable decision support remain comparatively thin [1,2]. Ensemble and boosting classifiers dominate published benchmarks [2,9,10], while profit-driven and prescriptive approaches improve monetary criteria and can account for heterogeneous lifetime value or incentive cost [4,5]. Prediction-to-retention-policy integration itself is not new: Chu et al. [15] proposed a hybrid churn-and-policy architecture, and recent Knowledge-Based Systems work has developed explainable decision support for strategic customer development [16]. Most directly, Latorre et al. [17] jointly optimise retention targeting and incentive level for profit in a predictor-agnostic prescriptive layer. Probability calibration is required when scores are used as decision inputs [11], and human inspection belongs inside a decision-support system rather than as an afterthought [12]. Knowledge bases and graphs have also been used to enrich churn features [3] and to support structured decisions in other domains [13,14].",
    )
    add_text(
        doc,
        "The scientific gap is therefore not another increment in churn AUROC, nor simply the move from prediction to policy. Existing prescriptive churn work primarily optimises economic targeting and incentives [4,5,15,17]. Comparatively little work formulates strategic customer-value coverage and economic utility as non-identical objectives under a shared budget with scarce human review. Two operational facts motivate that formulation. First, economic targeting based on monthly charges is not equivalent to strategic targeting based on lifetime-value knowledge: in the IBM Telco workbook, CLTV and the simulated annual-margin proxy have Pearson correlation 0.10. Second, routing every uncertain case to review can consume budget without improving strategic allocation.",
    )
    add_text(
        doc,
        'This short communication separates prediction from allocation. A calibrated ensemble estimates churn probability. CLTV supplies strategic-value knowledge, while a compact service-contract graph provides an auxiliary fold-local structural diagnostic. Here, "knowledge-guided" refers primarily to CLTV and explicit operational constraints; the graph is not claimed as the source of the primary allocation gain. The proposed knowledge-guided decision intelligence (KGDI) policy combines majority economic allocation with a minority review reserve for high-value customers. High-value churn reach under a shared expected-cost budget is the main strategic endpoint; overall churn reach and an outcome-anchored net-benefit proxy are secondary. Because the policy uses the same CLTV quantile that defines high-value evaluation, the coverage result is achievement of an intended objective rather than discovery of a hidden segment. A matched CLTV-reserved offer-only comparator and an exact efficiency benchmark test mechanism and efficiency. Only two tables and two figures are reported.',
    )

    add_h1(doc, "2. Proposed Framework")
    add_h2(doc, "2.1 Predictive model")
    add_text(
        doc,
        "Leakage-safe repeated stratified out-of-fold prediction is used (5 folds × 5 repeats). Predictors are restricted to an explicit allowlist of service, contract, billing, and tenure fields. Direct leakage fields (churn label, churn score, churn reason) are excluded; gender and senior-citizen status are excluded from the model and policy and retained only for audit. Logistic regression, random forest, XGBoost, and LightGBM are calibrated with a sigmoid map fitted on a dedicated calibration split inside each training fold [11]. The ensemble probability pᵢ is the equal-weight mean of the four calibrated probabilities. Normalised binary entropy of pᵢ is the primary uncertainty signal; between-model disagreement is retained only as a diagnostic. No synthetic oversampling is applied; imbalance is handled through cost-aware utilities [6-8]. Ranking and calibration, not F1, feed the decision equations.",
    )

    add_h2(doc, "2.2 Knowledge representation")
    add_text(
        doc,
        "A heterogeneous service-contract graph is built from observed IBM fields. Customer nodes link to contract type, internet service, payment method, and add-on or protection services. Motifs were specified from service-contract semantics-month-to-month fibre, electronic-check payment, unprotected internet, and short tenure-before fold-local rate estimation; they were not selected by scanning full-dataset outcomes. One-hop knowledge risk kᵢ for a held-out customer is the training-fold empirical churn rate of incident motifs, averaged out-of-fold. Test-fold labels never enter motif rates. Knowledge risk is a structural prior (AUROC 0.81) rather than a replacement for the ensemble (AUROC 0.86; correlation of kᵢ with pᵢ = 0.76). The residual kᵢ − pᵢ is an optional routing diagnostic; no incremental effect is assumed a priori. This construction is a compact domain graph, not an industrial ontology or graph-reasoning engine [3,13,14].",
    )
    add_text(
        doc,
        "IBM CLTV is a customer-node attribute converted to a percentile rank and is never treated as currency. Economic value in the utility is the simulated margin proxy Vᵢ = monthly charges × 12 × 0.60.",
    )

    add_h2(doc, "2.3 Decision model")
    add_text(
        doc,
        "Let cₒ be offer cost, s the assumed retention-success probability, cₕ the review cost, and (Se, Sp) reviewer sensitivity and specificity. Expected offer and review utilities are",
        first_line=True,
    )
    add_text(doc, "Uᵢ^offer = pᵢ s Vᵢ − cₒ                                              (1)", first_line=False, align="center")
    add_text(
        doc,
        "Uᵢ^review = pᵢ Se s Vᵢ − (cₕ + cₒ [pᵢ Se + (1 − pᵢ)(1 − Sp)])     (2)",
        first_line=False, align="center",
    )
    add_text(
        doc,
        "A reviewed true churner proceeds to a retention offer with probability Se, whereas a reviewed non-churner receives an unnecessary follow-up offer with probability 1 − Sp; expected review cost therefore includes the review cost and the probability-weighted downstream offer cost. The planning budget is B = N cₒ × 0.20 and is enforced on ex-ante expected selected cost. Three primary baselines receive the same B: (i) probability-threshold ranking among pᵢ ≥ 0.5; (ii) cost-aware greedy ranking by expected offer utility among pᵢ ≥ 0.20 and positive expected offer utility; and (iii) uncertainty review, which routes eligible customers with normalised entropy ≥ 0.65 to human review and otherwise offers [4,5,12]. The cost-aware baseline is the primary economic comparator.",
    )
    add_text(
        doc,
        "The proposed KGDI policy uses a nominal operating point of 85% of B on cost-aware offers and 15% reserved for human review. The minority review reserve is an interpretable capacity assumption, not a uniquely optimal split. Review eligibility requires pᵢ ≥ 0.20, CLTV at or above the 75th percentile, positive review utility, and either normalised entropy ≥ 0.45 or kᵢ − pᵢ > 0. Eligible review candidates are ranked by expected review utility divided by expected review cost. Any leftover budget returns to cost-aware offers. A robustness path over economic shares {0.70, 0.75, 0.80, 0.85, 0.90, 1.00} is evaluated. Observed churn is never used to choose actions.",
    )
    add_text(
        doc,
        "Default values are cₒ = $100, s = 0.35, cₕ = $25, Se = Sp = 0.85. Inference versus cost-aware targeting uses 1000 paired customer bootstrap resamples with policy rerouting and budget reallocation inside every resample. The bootstrap conditions on previously generated out-of-fold risk estimates and therefore quantifies decision-allocation uncertainty rather than full model-training uncertainty. Reported quantities are observed deltas, percentile 95% intervals, and the fraction of resamples with a positive delta; those fractions are not classical p-values.",
    )

    add_h1(doc, "3. Experimental Design")
    add_text(
        doc,
        "The IBM Telco Customer Churn workbook contains 7043 customers and a 26.5% churn rate. High-value churners, used only for evaluation, are observed churners with CLTV percentile at least 0.75 (n = 365). Because the policy uses the same 0.75 CLTV quantile as a review gate, high-value reach is an intended coverage objective. All primary policies share planning budget $140,860. Action rate may exceed 20% when cheaper reviews replace offers while expected selected cost remains at B. Public telecommunications tables of this type remain a standard test-bed [1,2,7].",
    )
    add_text(
        doc,
        "Two post hoc analyses probe mechanism and efficiency without refitting the models. A matched CLTV-reserved offer-only comparator uses the same 85/15 budget structure, but spends the reserve on eligible high-CLTV direct offers rather than review. An exact binary epsilon-constraint programme on frozen out-of-fold scores traces the frontier between expected economic utility and expected high-value churn-risk mass, subject to the same budget and one action per customer. Observed churn labels are excluded from that optimisation.",
    )

    add_h1(doc, "4. Results and Discussion")
    add_text(
        doc,
        "Table 1 shows that the ensemble is well calibrated (Brier 0.130 ± 0.001; expected calibration error 0.012 ± 0.003) and therefore supports using these probabilities in Eqs. (1)-(2). No predictive state of the art is claimed [1,2,7,9]. Logistic regression has a higher F1 than the ensemble; the policy uses ranking and calibration rather than F1.",
    )

    add_caption(
        doc,
        "Table 1. Repeated out-of-fold predictive performance (mean ± sd over five repeats).",
    )
    fill_table(
        doc,
        ["Model", "AUROC", "PR-AUC", "F1", "Brier ↓", "ECE ↓"],
        [
            ["Logistic regression", "0.856 ± 0.000", "0.669 ± 0.002", "0.616 ± 0.003", "0.132 ± 0.000", "0.013 ± 0.003"],
            ["Random forest", "0.849 ± 0.002", "0.656 ± 0.008", "0.563 ± 0.011", "0.135 ± 0.001", "0.015 ± 0.002"],
            ["XGBoost", "0.859 ± 0.001", "0.679 ± 0.005", "0.600 ± 0.006", "0.131 ± 0.000", "0.011 ± 0.003"],
            ["LightGBM", "0.851 ± 0.001", "0.665 ± 0.005", "0.586 ± 0.006", "0.134 ± 0.000", "0.014 ± 0.003"],
            ["Ensemble", "0.860 ± 0.001", "0.680 ± 0.005", "0.601 ± 0.005", "0.130 ± 0.001", "0.012 ± 0.003"],
        ],
        [1.6, 1.15, 1.15, 1.15, 1.15, 1.15],
    )
    add_text(
        doc,
        "Notes: Brier and ECE are lower-is-better. ECE: expected calibration error.",
        size=10, italic=True, first_line=False, after=12,
    )

    add_text(
        doc,
        "Table 2 summarises the decision results. Cost-aware targeting is the strongest offer-only economic baseline ($65,209). Uncertainty review raises high-value reach to 54.2% with 1650 reviews, but the proxy falls to $59,253. Proposed KGDI issues 1297 offers and 166 reviews (20.8% action rate; expected selected cost remains within B). High-value churn reach is 56.4% versus 43.8% for cost-aware targeting (+12.6 percentage points; 1000/1000 primary bootstrap resamples positive; 95% interval [9.3, 16.5] pp). Overall churn reach is essentially unchanged (50.6% versus 49.9%). The net-benefit proxy is $67,529 (+$2320), but its 95% interval [-$92, $5397] includes a small negative value, so economic dominance is not confirmed. The matched CLTV-reserved offer-only comparator reaches only 44.1% high-value churn at $65,157. In a paired mechanism bootstrap between the CLTV-only review-reserve policy-which is action-identical to full KGDI at the nominal point-and the matched offer-only reserve, high-value reach improves by 12.3 percentage points (1000/1000; 95% interval [8.8, 15.6] pp); the net-benefit difference is +$2372 (967/1000; 95% interval [-$149, $4875]). Under the assumed reviewer model, review routing, not CLTV reservation alone, produces the coverage gain.",
    )

    add_caption(
        doc,
        "Table 2. Budget-matched decision performance. Primary deltas for KGDI are versus cost-aware targeting; the CLTV-reserved offer-only row is a post hoc matched mechanism comparator.",
    )
    fill_table(
        doc,
        [
            "Strategy", "Action %", "Churn reach %", "HV reach %",
            "Offers / reviews", "Net-benefit ($)", "Expected cost ($)",
        ],
        [
            ["Probability threshold", "20.0", "51.7", "43.0", "1408 / 0", "53,786", "140,800"],
            ["Cost-aware", "20.0", "49.9", "43.8", "1408 / 0", "65,209", "140,800"],
            ["CLTV-reserved offer-only", "20.0", "49.9", "44.1", "1408 / 0", "65,157", "140,800"],
            ["Uncertainty review", "24.5", "56.4", "54.2", "79 / 1650", "59,253", "140,811"],
            ["Proposed KGDI", "20.8", "50.6", "56.4", "1297 / 166", "67,529", "140,839"],
        ],
        [1.85, 0.75, 1.05, 0.85, 1.15, 1.15, 1.15],
    )
    add_text(
        doc,
        "Notes: Action %: proportion assigned either a retention offer or human review. HV: high-value churn reach. CI: percentile interval. Resample fractions are not classical p-values. Primary KGDI versus cost-aware HV-delta 95% interval: [9.3, 16.5] pp. Matched CLTV-review versus CLTV-offer-only HV-delta interval: [8.8, 15.6] pp.",
        size=10, italic=True, first_line=False, after=12,
    )

    add_text(
        doc,
        "Component ablation under the same 85/15 split identifies the operative mechanism. A CLTV-only review reserve produces the same 1297 offers, 166 reviews, 50.6% overall reach, 56.4% high-value reach, and $67,529 proxy as full KGDI (0 of 7043 actions differ). Adding entropy does not change that set. CLTV plus graph residual without entropy yields 101 reviews and 51.8% high-value reach. Removing the CLTV gate increases reviews to 305 and drops high-value reach to 47.1%. Setting the economic share to 1.00 returns high-value reach to 44.1%. The coverage gain is therefore the reserved high-CLTV review slice, not incremental graph-residual routing. Positive-residual terciles have low mean predicted probability (0.11-0.17) and observed churn slightly below pᵢ. Residual and model disagreement are distinct (correlation -0.29; top-decile Jaccard overlap 0.012). The graph is retained as a leak-safe structural diagnostic, not as the source of the decision gain.",
    )
    add_text(
        doc,
        "Expected utility is the ex-ante optimisation objective; the net-benefit proxy is a separate ex-post quantity. The exact epsilon-constraint maximum-strategic endpoint reaches 53.48% of expected high-value churn-risk mass while retaining 99.68% of the utility optimum (ex-post high-value reach 56.4%; net-benefit $69,280). The 85/15 heuristic matches that strategic coverage but retains only 93.02% of the utility optimum ($67,529). At the same strategic target the expected-utility gap is about $4260 (63,809 versus 59,549). Across 11 scenario perturbations the high-value coverage advantage remains positive, but it narrows when review is expensive or reviewer accuracy is low, and point-estimate economic superiority changes sign in several scenarios. Setting Se = Sp = 0.50 eliminates all reviews and reverts to the cost-aware offer-only result, showing that the gain depends on informative reviewer performance under the assumed model.",
    )

    add_text(
        doc,
        "Fig. 1 shows offers concentrated at higher calibrated risk, with human review almost entirely above the 75th CLTV percentile. Marker size encodes graph motif risk as an auxiliary structural diagnostic, not as the allocation engine. Fig. 2 reports the post hoc efficiency benchmark on frozen out-of-fold scores. The 85/15 strategic-review heuristic reaches the same maximum strategic-risk coverage shown by the exact endpoint, but with lower expected-utility retention.",
        first_line=True,
    )

    add_picture(doc, FIG1, 3.15)
    add_caption(
        doc,
        "Fig. 1. Strategic-value-guided decision map. Horizontal axis: calibrated ensemble churn probability. Vertical axis: CLTV percentile. Marker size encodes graph motif risk as an auxiliary structural diagnostic. The dotted line is the high-value CLTV quantile (0.75).",
    )
    add_picture(doc, FIG2, 3.15)
    add_caption(
        doc,
        "Fig. 2. Exact epsilon-constraint frontier on frozen out-of-fold scores: expected high-value churn-risk mass versus expected utility retained. Observed labels are not used in the optimisation.",
    )

    add_h1(doc, "5. Limitations")
    add_text(
        doc,
        "Business parameters are simulated, reviewers are represented by fixed sensitivity and specificity, and only one public cohort is analysed. High-value reach uses the same CLTV threshold that the policy gates on. The matched CLTV-offer analysis and the epsilon-constraint benchmark are post hoc mechanism and efficiency checks. The graph residual is non-binding at the nominal point. The bootstrap conditions on frozen out-of-fold scores. The 85/15 heuristic is not Pareto-efficient. Reviewer performance, intervention success, and costs would require empirical estimation in deployment. External validity requires another cohort or a temporal split [1,2].",
        first_line=False,
    )

    add_h1(doc, "6. Conclusion")
    add_text(
        doc,
        "This short communication formulates customer-churn intervention as a constrained decision problem rather than a classification endpoint. A calibrated ensemble supplies risk; CLTV supplies strategic-value knowledge; scarce human review and a shared expected-cost budget define the operational decision space; and a compact service-contract graph supplies auxiliary structural context [3-5,11,12,14]. On IBM Telco data, the 85/15 strategic-review policy raises high-value churn reach from 43.8% under cost-aware targeting to 56.4% while leaving overall churn reach essentially unchanged. A matched CLTV-reserved offer-only policy reaches only 44.1%; under the stated reviewer model, review routing is the operative mechanism. The economic point estimate favours the policy but remains interval-qualified. Ablation shows that entropy and graph-residual gates do not add incremental routing at the nominal point. The exact epsilon-constraint benchmark reaches the same strategic coverage with better expected-utility retention. The contribution is a budget-constrained decision-support architecture and mechanism analysis, not a new classifier, a claim of graph reasoning, or a claim of economic optimality.",
        first_line=False,
    )

    add_h1(doc, "Data Availability and Declarations")
    add_text(
        doc,
        "The IBM Telco Customer Churn sample is a public IBM Cognos Analytics benchmark [18]. The raw workbook is not redistributed. Analysis code is available at https://github.com/drsaikrishnathota1/kbs-churn-decision-intelligence. Reproduction requires the IBM sample and main.py to generate leakage-safe out-of-fold scores; experiment_c.py, letter_outputs.py, review_analyses.py, and experiment_b.py regenerate the decision, mechanism, and efficiency analyses. Competing interest: none. Funding: none. Generative AI: AI-assisted tools were used for language drafting, editing, organisation, and formatting; the author reviewed the content and takes full responsibility. Tables and figures were generated from executable code. CRediT: Sai Krishna Thota: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review and editing, Visualization.",
        first_line=False,
    )

    add_h1(doc, "References")
    refs = [
        "[1] Manzoor, A., Qureshi, M. A., Kidney, E., & Longo, L. (2024). A review on machine learning methods for customer churn prediction and recommendations for business practitioners. IEEE Access, 12, 70434-70463. https://doi.org/10.1109/ACCESS.2024.3402092",
        "[2] Imani, M., Joudaki, M., Beikmohammadi, A., & Arabnia, H. R. (2025). Customer churn prediction: A systematic review of recent advances, trends, and challenges in machine learning and deep learning. Machine Learning and Knowledge Extraction, 7(3), 105. https://doi.org/10.3390/make7030105",
        "[3] Shahabikargar, M., Beheshti, A., Mansoor, W., Zhang, X., Foo, E. J., Jolfaei, A., Hanif, A., & Shabani, N. (2025). ChurnKB: A generative AI-enriched knowledge base for customer churn feature engineering. Algorithms, 18(4), 238. https://doi.org/10.3390/a18040238",
        "[4] Jiang, P., Liu, Z., Abedin, M. Z., Wang, J., Yang, W., & Dong, Q. (2024). Profit-driven weighted classifier with interpretable ability for customer churn prediction. Omega, 125, 103034. https://doi.org/10.1016/j.omega.2024.103034",
        "[5] Feng, Y., Yin, Y., Wang, D., Ignatius, J., Cheng, T. C. E., Marra, M., & Guo, Y. (2024). Enhancing e-commerce customer churn management with a profit- and AUC-focused prescriptive analytics approach. Journal of Business Research, 184, 114872. https://doi.org/10.1016/j.jbusres.2024.114872",
        "[6] Rao, C., Xu, Y., Xiao, X., Hu, F., & Goh, M. (2024). Imbalanced customer churn classification using a new multi-strategy collaborative processing method. Expert Systems with Applications, 247, 123251. https://doi.org/10.1016/j.eswa.2024.123251",
        "[7] Wang, C., Rao, C., Hu, F., Xiao, X., & Goh, M. (2024). Risk assessment of customer churn in telco using FCLCNN-LSTM model. Expert Systems with Applications, 248, 123352. https://doi.org/10.1016/j.eswa.2024.123352",
        "[8] Haddadi, S. J., Farshidvard, A., dos Santos Silva, F., dos Reis, J. C., & Reis, M. S. (2024). Customer churn prediction in imbalanced datasets with resampling methods: A comparative study. Expert Systems with Applications, 246, 123086. https://doi.org/10.1016/j.eswa.2023.123086",
        "[9] Joy, U. G., Hoque, K. E., Uddin, M. N., Chowdhury, L., & Park, S.-B. (2024). A big data-driven hybrid model for enhancing streaming service customer retention through churn prediction integrated with explainable AI. IEEE Access, 12, 69130-69150. https://doi.org/10.1109/ACCESS.2024.3401247",
        "[10] Poudel, S. S., Pokharel, S., & Timilsina, M. (2024). Explaining customer churn prediction in telecom industry using tabular machine learning models. Machine Learning with Applications, 17, 100567. https://doi.org/10.1016/j.mlwa.2024.100567",
        "[11] Văduva, A.-G., Oprea, S. V., Niculae, A.-M., Bâra, A., & Andreescu, A.-I. (2024). Improving churn detection in the banking sector: A machine learning approach with probability calibration techniques. Electronics, 13(22), 4527. https://doi.org/10.3390/electronics13224527",
        "[12] Kostopoulos, G., Davrazos, G., & Kotsiantis, S. (2024). Explainable artificial intelligence-based decision support systems: A recent review. Electronics, 13(14), 2842. https://doi.org/10.3390/electronics13142842",
        "[13] Guo, Z., Zhou, D., Yu, D., Zhou, Q., Wu, H., & Hao, A. (2024). An ontology-based method for knowledge reuse in the design for maintenance of complex products. Computers in Industry, 161, 104124. https://doi.org/10.1016/j.compind.2024.104124",
        "[14] Su, C., Jiang, Q., Han, Y., Wang, T., & He, Q. (2025). Knowledge graph-driven decision support for manufacturing process: A graph neural network-based knowledge reasoning approach. Advanced Engineering Informatics, 64, 103098. https://doi.org/10.1016/j.aei.2024.103098",
        "[15] Chu, B.-H., Tsai, M.-S., & Ho, C.-S. (2007). Toward a hybrid data mining model for customer retention. Knowledge-Based Systems, 20(8), 703-718. https://doi.org/10.1016/j.knosys.2006.10.003",
        "[16] Abbaspour Onari, M., Jahangoshai Rezaee, M., Saberi, M., & Nobile, M. S. (2024). An explainable data-driven decision support framework for strategic customer development. Knowledge-Based Systems, 295, 111761. https://doi.org/10.1016/j.knosys.2024.111761",
        "[17] Latorre, P., Meza, A., López-Ospina, H., Verbeke, W., & Pérez, J. (2025). A prescriptive analytics framework for jointly optimizing retention incentives and targeting. Knowledge-Based Systems, 330, 114649. https://doi.org/10.1016/j.knosys.2025.114649",
        "[18] IBM Cognos Analytics Samples Team. (2019). Telco customer churn (11.1.3+). IBM Community. https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113",
    ]
    for r in refs:
        add_text(doc, r, size=9, first_line=False, after=0, space=1.0)

    doc.save(OUT)
    print(f"Wrote {OUT}")
    print("words", word_count(doc))


if __name__ == "__main__":
    build()
