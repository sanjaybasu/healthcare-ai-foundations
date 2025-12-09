---
layout: chapter
title: "Chapter 23: Precision Medicine and Treatment Optimization"
chapter_number: 23
part_number: 6
prev_chapter: /chapters/chapter-22-clinical-decision-support/
next_chapter: /chapters/chapter-24-population-health-screening/
---
# Chapter 23: Precision Medicine and Treatment Optimization

## Learning Objectives

By the end of this chapter, you will be able to:

1. Understand the mathematical foundations of treatment effect heterogeneity estimation and how individual-level treatment effects can be predicted to personalize therapeutic recommendations across diverse populations
2. Implement causal inference methods including meta-learners, doubly robust estimators, and causal forests to identify subgroups who benefit differentially from specific treatments
3. Design recommendation systems that balance multiple competing objectives including clinical efficacy, safety, patient preferences, cost-effectiveness, and equitable access to care
4. Develop medication dosing algorithms that account for pharmacokinetic and pharmacodynamic variability across populations while avoiding algorithmic bias in dosing recommendations
5. Build clinical pathway optimization systems that adapt to resource availability and care setting constraints rather than assuming uniform access to interventions
6. Incorporate patient preferences, values, and shared decision-making principles into automated decision support while respecting cultural differences in health beliefs and treatment goals
7. Identify and mitigate sources of inequity in treatment recommendations including disparities in treatment effect estimation, resource-intensive interventions, and recommendations that fail to account for social determinants of health
8. Validate treatment recommendation systems for fairness across demographic groups, clinical subpopulations, and care settings to ensure equitable benefit from clinical decision support tools

## Introduction

Treatment recommendation systems represent one of the most consequential applications of artificial intelligence in healthcare, directly influencing therapeutic decisions that affect patient outcomes, quality of life, and survival. Unlike diagnostic systems that classify disease states, treatment recommendation systems must navigate the complex landscape of causal inference, individual heterogeneity, competing risks, patient autonomy, and resource constraints. The fundamental challenge is that the optimal treatment for a given patient depends on predicted individual treatment effects, which cannot be directly observed since each patient receives only one treatment at a time. This requires sophisticated causal inference methodology to estimate counterfactual outcomes and identify which patients benefit most from specific interventions.

The equity implications of treatment recommendation systems are profound and multifaceted. First, treatment effects themselves often vary systematically across populations due to biological differences, social determinants of health, environmental exposures, and healthcare access patterns. Second, many advanced treatments require resources, infrastructure, or adherence support that may be inequitably distributed. Third, the data used to train recommendation systems often underrepresents marginalized populations, leading to uncertain or biased predictions for these groups. Fourth, patient preferences and values regarding treatment trade-offs may differ across cultures, yet recommendation systems often embed implicit assumptions about acceptable risks and benefits. Finally, deployment of clinical decision support systems themselves may be inequitable, with advanced tools more readily available in well-resourced healthcare settings.

This chapter develops a comprehensive framework for building treatment recommendation systems that explicitly address these equity challenges while maintaining clinical rigor. We begin with the mathematical foundations of causal inference for treatment effect heterogeneity, including meta-learners, doubly robust estimation, and causal forests. We then examine how to model patient preferences and utilities in ways that respect diverse values and incorporate shared decision-making principles. The chapter proceeds to multi-objective optimization frameworks that balance efficacy, safety, equity, and cost considerations. We provide detailed implementations of medication dosing systems that account for pharmacokinetic variability, treatment selection systems that identify optimal therapies for individual patients, and clinical pathway optimization that adapts to resource constraints. Throughout, we emphasize validation approaches that assess fairness across populations and strategies to avoid perpetuating or amplifying treatment disparities. The goal is to equip healthcare data scientists with the technical tools and ethical frameworks necessary to build recommendation systems that improve outcomes equitably across all patient populations.

## Mathematical Foundations of Treatment Effect Heterogeneity

The fundamental challenge in treatment recommendation is estimating individual treatment effects when each patient receives only one treatment. For patient $$ i $$, we define the potential outcomes $$ Y_i(1) $$ under treatment and $$ Y_i(0) $$ under control. The individual treatment effect is $$ \tau_i = Y_i(1) - Y_i(0) $$, but we observe only $$ Y_i = T_i Y_i(1) + (1-T_i)Y_i(0) $$ where $$ T_i \in \{0,1\} $$ indicates treatment assignment. The average treatment effect (ATE) is $$ \tau = \mathbb{E}[\tau_i] $$, but our goal is to estimate the conditional average treatment effect (CATE) $$ \tau(x) = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i = x] $$ to personalize treatment recommendations based on patient characteristics $$ X_i $$.

Under the assumptions of unconfoundedness $$ (Y_i(0), Y_i(1)) \perp T_i \mid X_i $$ and positivity $$ 0 \lt  P(T_i=1\mid X_i=x) \lt  1 $$, the CATE can be identified from observational data. The most direct approach is the S-learner which builds a single model $$ \mu(x,t) $$ to predict outcomes and estimates $$ \hat{\tau}(x) = \hat{\mu}(x,1) - \hat{\mu}(x,0) $$. However, when treatment effects are small relative to baseline risk, the S-learner may perform poorly because it must learn both the prognostic function and treatment effect simultaneously.

The T-learner addresses this by building separate models $$ \mu_0(x) $$ and $$ \mu_1(x) $$ for control and treatment groups, estimating $$ \hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x) $$. This allows each model to focus on prediction within its respective group but may suffer from overfitting when one treatment group is small. The X-learner improves upon the T-learner by incorporating information across treatment groups. After fitting $$ \hat{\mu}_0 $$ and $$ \hat{\mu}_1 $$, it computes imputed treatment effects $$ \tilde{\tau}_1(x) = Y_i - \hat{\mu}_0(X_i) $$ for treated patients and $$ \tilde{\tau}_0(x) = \hat{\mu}_1(X_i) - Y_i $$ for controls, then builds models $$ \tau_1(x) $$ and $$ \tau_0(x) $$ to predict these imputed effects. The final CATE estimate is $$ \hat{\tau}(x) = g(x)\hat{\tau}_0(x) + (1-g(x))\hat{\tau}_1(x) $$ where $$ g(x) = P(T=1\mid X=x) $$ weights the estimates by propensity score.

For more robust estimation, doubly robust methods combine outcome modeling with propensity score weighting. The augmented inverse propensity weighted (AIPW) estimator for the CATE is given by

$$\hat{\tau}(x) = \mathbb{E}\left[\frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)} + \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) \Big\mid X_i = x\right]$$

where $$ \hat{e}(x) = P(T=1\mid X=x) $$ is the estimated propensity score. This estimator is consistent if either the outcome models or the propensity model is correctly specified, providing protection against model misspecification. In practice, we use flexible machine learning methods for both $$ \hat{\mu}_t(x) $$ and $$ \hat{e}(x) $$, combining them through cross-fitting to avoid overfitting bias.

Causal forests extend random forests to estimate CATEs by modifying the splitting criterion to maximize treatment effect heterogeneity. Each tree is built by randomly subsampling the data and features, then recursively partitioning to maximize the difference in treatment effects across resulting nodes. For a given leaf $$ L(x) $$ containing patient $$ i $$, the treatment effect estimate equals

$$\hat{\tau}(x) = \frac{\sum_{i: X_i \in L(x)} T_i Y_i}{\sum_{i: X_i \in L(x)} T_i} - \frac{\sum_{i: X_i \in L(x)} (1-T_i) Y_i}{\sum_{i: X_i \in L(x)} (1-T_i)}$$

Honest causal forests improve upon this by using a sample splitting approach where one subset of data is used to determine the tree structure and another independent subset is used to estimate treatment effects within leaves, reducing overfitting. The resulting estimates have desirable theoretical properties including consistency and asymptotic normality under regularity conditions.

For treatment recommendations, we must account for uncertainty in CATE estimates. The variance of $$ \hat{\tau}(x) $$ depends on both the variance of potential outcomes and the propensity score. Regions of covariate space with low propensity scores have high variance estimates because few patients receive treatment. This has important equity implications: if certain demographic groups are historically undertreated, their CATE estimates will be uncertain, yet they may be the populations that could benefit most from treatment. Confidence intervals for CATEs can be constructed using bootstrap methods for meta-learners or using the asymptotic distribution for causal forests. When making treatment recommendations, we should favor treatments with both high estimated benefit and sufficient precision to distinguish from null effects.

## Patient Preferences and Utility Modeling

Effective treatment recommendations must incorporate patient preferences and values, which often vary across populations and cultures. Utility theory provides a mathematical framework for quantifying how patients value different health outcomes. For patient $$ i $$ with covariates $$ X_i $$, we define a utility function $$ U_i(h) $$ that maps health states $$ h $$ to real numbers representing preference. Under treatment $$ t $$, patient $$ i $$ experiences outcome $$ Y_i(t) $$ with associated health state $$ h_i(t) $$. The optimal treatment maximizes expected utility, given by

$$t^{\star} = \arg\max_{t \in \mathcal{T}} \mathbb{E}[U_i(h_i(t)) \mid X_i]$$

In clinical contexts, health states are typically multidimensional, encompassing survival, quality of life, symptom burden, functional status, and treatment side effects. Quality-adjusted life years (QALYs) provide one approach to aggregate these dimensions, defining $$ U_i(h) = L_i \cdot Q_i $$ where $$ L_i $$ is life years and $$ Q_i \in [0,1] $$ is quality of life. However, the QALY framework embeds assumptions about temporal additivity and constant proportional trade-offs that may not reflect individual preferences. More flexible approaches use multi-attribute utility functions of the form

$$U_i(h) = \sum_{j=1}^J w_{ij} u_{ij}(h_j)$$

where $$ h_j $$ are attributes such as mobility, pain, cognitive function, and $$ w_{ij} $$ are patient-specific weights reflecting the relative importance of each attribute. Eliciting these weights requires careful preference assessment through methods such as time trade-off tasks, standard gambles, or discrete choice experiments.

Crucially, preferences vary systematically across populations. Research has documented cultural differences in preferences for life-extending versus symptom-relieving treatments, willingness to accept treatment side effects, and attitudes toward aggressive interventions. Older adults may prioritize quality over quantity of life, while parents of young children may accept greater treatment burden to maximize survival. Patients from collectivist cultures may weigh family preferences more heavily than those from individualistic cultures. Socioeconomic factors influence preferences through different mechanisms: patients facing financial constraints may discount treatments with high out-of-pocket costs, while those with limited social support may avoid treatments requiring intensive caregiver involvement.

To model preference heterogeneity, we can estimate conditional utility functions $$ U(h; x, p) $$ where $$ x $$ represents clinical characteristics and $$ p $$ represents patient-reported preferences. One approach uses mixed logit models to estimate preference distributions of the form

$$P(y_i = t \mid x_i, p_i) = \frac{\exp(\beta_t^T f(x_i, p_i))}{\sum_{t' \in \mathcal{T}} \exp(\beta_{t'}^T f(x_i, p_i))}$$

where $$ f(x_i, p_i) $$ combines clinical features with preference indicators and $$ \beta_t $$ captures how different attributes influence treatment choice. By allowing coefficients to vary across individuals, we capture heterogeneous preferences while still estimating population-level preference distributions.

An equity-focused approach to preference modeling recognizes that stated preferences may be constrained by past experiences and structural inequities. Patients who have experienced discrimination in healthcare may distrust aggressive treatments or clinical trials. Those who have faced financial toxicity from previous treatments may undervalue effective but expensive therapies. Simply eliciting and implementing stated preferences may perpetuate these constraints rather than expand patient choice. Instead, we should present information about treatment options in ways that help patients understand their full range of choices, explicitly discuss how resource constraints could be addressed, and distinguish between inherent preferences versus preferences shaped by structural barriers.

## Multi-Objective Optimization for Treatment Recommendations

Treatment recommendations must balance multiple competing objectives including clinical efficacy, safety, cost, patient preferences, and equity considerations. We formalize this as a multi-objective optimization problem where we seek treatment recommendations that optimize a vector of objectives $$ \mathbf{f}(t, x) = (f_1(t,x), \ldots, f_K(t,x))^T $$. Representative objectives include

- Efficacy — Expected improvement in clinical outcome $$ f_1(t,x) = \mathbb{E}[Y_i(t) - Y_i(0) \mid X_i=x] $$
- Safety — Negative expected adverse events $$ f_2(t,x) = -\mathbb{E}[\text{AE}_i(t) \mid X_i=x] $$
- Cost — Negative total healthcare costs $$ f_3(t,x) = -\mathbb{E}[C_i(t) \mid X_i=x] $$
- Equity — Reduction in outcome disparities $$ f_4(t,x) = -\text{Var}_{g \in \mathcal{G}}[\mathbb{E}[Y_i(t) \mid G_i=g]] $$

where $$ g $$ indexes demographic or social groups. A solution $$ t $$ is Pareto optimal if no other treatment is better on all objectives simultaneously. The set of Pareto optimal solutions forms the Pareto frontier, representing optimal trade-offs between competing goals.

To identify a single recommended treatment, we must aggregate objectives using a scalarization function. The weighted sum approach defines $$ f_{\text{agg}}(t,x) = \sum_{k=1}^K \omega_k f_k(t,x) $$ where $$ \omega_k \geq 0 $$ are weights reflecting relative importance. By varying weights, we trace out the Pareto frontier. However, this approach requires objectives to be commensurate (same units) and may miss non-convex regions of the Pareto frontier.

The constraint-based approach instead treats some objectives as hard constraints: maximize $$ f_1(t,x) $$ subject to $$ f_k(t,x) \geq \epsilon_k $$ for $$ k=2,\ldots,K $$. This is appropriate when certain objectives represent minimum acceptable thresholds. For example, we might maximize efficacy subject to the constraints that adverse event rate is below 10%, total cost is below a budget threshold, and the treatment effect is positive for all demographic subgroups.

From an equity perspective, several optimization formulations explicitly promote fairness. The maximin approach seeks to maximize the minimum outcome across groups, yielding

$$t^{\star} = \arg\max_t \min_{g \in \mathcal{G}} \mathbb{E}[Y_i(t) \mid G_i = g, X_i]$$

This prioritizes improvements for the worst-off group but may sacrifice overall efficacy. The Nash social welfare approach instead maximizes the product of group outcomes, expressed as

$$t^{\star} = \arg\max_t \prod_{g \in \mathcal{G}} \mathbb{E}[Y_i(t) \mid G_i = g, X_i]$$

providing a balance between efficiency and equity. Disparate impact constraints require that treatment effects do not differ substantially across groups, enforcing

$$\frac{\mathbb{E}[Y_i(t) \mid G_i = g_1, X_i]}{\mathbb{E}[Y_i(t) \mid G_i = g_2, X_i]} \geq 1 - \delta$$

for all pairs of groups $$ (g_1, g_2) $$ and some tolerance $$ \delta $$. These constraints ensure that recommendations do not exacerbate existing disparities.

In practice, we solve multi-objective treatment recommendation problems using mixed-integer programming when the treatment space is discrete, or constrained optimization when continuous. For complex problems with many objectives and constraints, evolutionary algorithms such as NSGA-II can approximate the Pareto frontier by maintaining a population of candidate solutions and iteratively selecting for non-dominated solutions with high diversity.

A critical equity consideration is that trade-offs between objectives may differ across populations. For low-income patients, cost may be a binding constraint that dramatically narrows the feasible treatment set. For patients in under-resourced care settings, treatments requiring frequent monitoring or specialized administration may be infeasible regardless of efficacy. The optimization framework must explicitly model these resource constraints as population-specific constraints rather than assuming uniform treatment availability.

## Medication Dosing and Pharmacometric Models

Precision dosing represents a critical application of treatment recommendation systems where the goal is to identify optimal drug dosing regimens that maximize efficacy while minimizing toxicity. Pharmacometric models describe how drug concentrations change over time (pharmacokinetics) and how concentrations relate to effects (pharmacodynamics). These models form the basis for dose individualization based on patient characteristics.

The standard pharmacokinetic model describes drug concentration $$ C(t) $$ over time following a dose $$ D $$ using compartmental models. A one-compartment model with first-order elimination is

$$C(t) = \frac{D}{V} e^{-k_e t}$$

where $$ V $$ is volume of distribution and $$ k_e $$ is elimination rate constant. Multi-compartment models extend this to describe distribution across tissue compartments. Population pharmacokinetic models allow parameters to vary across individuals according to

$$\theta_i = \theta_{\text{pop}} \cdot e^{\eta_i}$$

where $$ \theta_i $$ is an individual parameter (e.g., clearance), $$ \theta_{\text{pop}} $$ is the population mean, and $$ \eta_i \sim N(0, \omega^2) $$ captures between-subject variability. Covariates such as age, weight, renal function, and genetic polymorphisms can be incorporated through

$$\theta_i = \theta_{\text{pop}} \cdot \prod_{j} \left(\frac{X_{ij}}{X_{\text{ref},j}}\right)^{\beta_j} \cdot e^{\eta_i}$$

allowing dose adjustment based on patient characteristics. For example, for drugs cleared renally, clearance scales with creatinine clearance: $$ CL_i = CL_{\text{pop}} \cdot (CrCl_i/CrCl_{\text{ref}})^{0.75} $$.

Pharmacodynamic models link concentration to effect. The sigmoid Emax model describes how concentration $$ C $$ produces effect $$ E $$ via

$$E(C) = E_0 + \frac{E_{\max} \cdot C^\gamma}{EC_{50}^\gamma + C^\gamma}$$

where $$ E_0 $$ is baseline effect, $$ E_{\max} $$ is maximum effect, $$ EC_{50} $$ is concentration producing half-maximal effect, and $$ \gamma $$ is Hill coefficient describing sigmoidicity. For many drugs, toxicity also increases with concentration, creating a therapeutic window between minimum effective concentration and maximum safe concentration.

To recommend optimal doses, we use model predictive control which repeatedly solves the optimization problem

$$\max_{d_1, \ldots, d_T} \sum_{t=1}^T U(E(C_t), A(C_t))$$

subject to pharmacokinetic model constraints linking doses $$ d_t $$ to concentrations $$ C_t $$, where $$ U(E,A) $$ is a utility function reflecting the benefit of therapeutic effect $$ E $$ and harm of adverse effects $$ A $$. This is solved at each time point using the current measured concentration and patient state, implementing only the first dose and then re-optimizing at the next time point once new measurements are available.

Equity issues in precision dosing arise through multiple mechanisms. First, population pharmacokinetic models are typically developed using data from phase 2 and 3 clinical trials that systematically underrepresent certain populations including racial and ethnic minorities, older adults, patients with multimorbidity, pregnant women, and children. When these populations are excluded from model development, parameter estimates may be biased and dose recommendations may be suboptimal or unsafe. Second, some dosing algorithms have explicitly incorporated race as a covariate, most notoriously in nephrology with eGFR equations that adjusted for race. These adjustments often lack biological justification and can lead to underdosing or overdosing of specific populations. Third, implementing precision dosing requires therapeutic drug monitoring infrastructure that may not be available in under-resourced settings, creating disparities in who benefits from dose optimization.

Best practices for equitable dosing algorithms include: developing models using diverse, representative populations; carefully evaluating whether demographic variables serve as valid proxies for biological mechanisms or instead reflect structural inequities; validating models separately within demographic subgroups; and designing dosing strategies that can be implemented across a range of care settings rather than assuming universal access to therapeutic drug monitoring.

## Treatment Selection and Ranking Systems

Treatment selection systems recommend which among multiple treatment options is most appropriate for a given patient. When treatments are mutually exclusive (e.g., choosing between chemotherapy regimens), this is a classification problem with treatments as classes. When treatments can be combined, this becomes a set recommendation problem identifying optimal treatment combinations.

For single treatment selection, we estimate the CATE for each available treatment $$ \tau_t(x) = \mathbb{E}[Y_i(t) - Y_i(0) \mid X_i=x] $$ and recommend

$$t^{\star}(x) = \arg\max_{t \in \mathcal{T}} \hat{\tau}_t(x)$$

However, this greedy approach ignores uncertainty in CATE estimates. A more robust approach uses Thompson sampling where we randomly sample from the posterior distribution of treatment effects and recommend the treatment with the highest sampled effect, namely

$$t^{\star}(x) \sim P(t = \arg\max_{t'} \tilde{\tau}_{t'}(x))$$

where $$ \tilde{\tau}_t(x) \sim P(\tau_t(x) \mid \mathcal{D}) $$ is sampled from the posterior given data $$ \mathcal{D} $$. This naturally incorporates exploration: treatments with uncertain effects have higher probability of being recommended, allowing the system to learn from experience.

For treatment combinations, we must consider interactions between therapies. Let $$ t = (t_1, \ldots, t_J) $$ be a vector indicating which treatments are administered. The outcome with combination treatment may exhibit synergistic effects described by

$$Y_i(\mathbf{t}) = \mu_0(X_i) + \sum_{j=1}^{J} \tau_j(X_i)\, t_j + \sum_{1 \le j < k \le J} \tau_{jk}(X_i)\, t_j t_k + \cdots$$

where $$ \tau_j(X_i) $$ are main effects and $$ \tau_{jk}(X_i) $$ are pairwise interaction effects. Estimating higher-order interactions requires substantial sample sizes. In practice, we often assume limited interactions and use regularization to select sparse models.

An alternative approach uses reinforcement learning to learn optimal treatment policies through sequential decision-making. We model treatment selection as a contextual bandit problem where at each decision point we observe patient state $$ X_i $$, choose action (treatment) $$ A_i $$, and observe reward $$ R_i $$. The goal is to learn a policy $$ \pi(a\mid x) $$ that maximizes expected reward $$ \mathbb{E}_{x,a \sim \pi}[R(x,a)] $$. Contextual bandit algorithms such as LinUCB maintain uncertainty estimates for each action and select actions optimistically according to

$$a^*(x) = \arg\max_a \left[\hat{Q}(x,a) + \beta \sqrt{\text{Var}[\hat{Q}(x,a)]}\right]$$

where $$ \hat{Q}(x,a) $$ is the estimated expected reward and $$ \beta $$ controls exploration. This UCB approach has theoretical guarantees on regret bounds and naturally handles the exploration-exploitation trade-off.

From an equity perspective, treatment selection systems must explicitly evaluate whether recommendations differ systematically across groups and whether these differences reflect true treatment effect heterogeneity or algorithmic bias. We should stratify validation analyses by demographic groups and clinical subpopulations, examining

- Recommendation rates — Are treatments recommended at different rates across groups?
- Treatment effect estimates — Do estimated benefits differ across groups, and are these differences clinically plausible?
- Uncertainty quantification — Are confidence intervals wider for underrepresented groups?
- Outcome disparities — Do implemented recommendations reduce or exacerbate disparities?

When treatment recommendations differ across groups, we must distinguish between appropriate individualization based on biological or clinical factors versus inappropriate disparity driven by biased data or models. This requires clinical review by diverse teams who can assess whether differences are medically justified.

## Clinical Pathway Optimization

Clinical pathways are structured multidisciplinary care plans that specify the sequence of interventions, timing of assessments, and decision points for managing specific conditions. Optimizing clinical pathways involves identifying the sequence of interventions that maximizes patient outcomes while respecting resource constraints, care setting capabilities, and patient preferences.

We model a clinical pathway as a finite-horizon Markov decision process where patient state $$ S_t $$ evolves over discrete time steps, actions $$ A_t $$ represent clinical interventions, and the transition dynamics $$ P(S_{t+1}\mid S_t, A_t) $$ describe disease progression and treatment response. The reward function $$ R(S_t, A_t) $$ captures both intermediate outcomes (symptom relief, functional improvement) and terminal outcomes (survival, quality of life). The optimal policy $$ \pi^{\star}(s) = \arg\max_a Q^{\star}(s,a) $$ maximizes expected cumulative reward defined by

$$Q^{\star}(s,a) = \mathbb{E}\left[\sum_{t=0}^T \gamma^t R(S_t, A_t) \Big\mid S_0=s, A_0=a, \pi^{\star}\right]$$

where $$ \gamma \in [0,1] $$ is a discount factor reflecting the relative value of immediate versus future rewards.

For pathways with moderate state and action spaces, we can solve for optimal policies using dynamic programming methods including value iteration or policy iteration. For complex pathways with high-dimensional state spaces, we use reinforcement learning with function approximation. Fitted Q-iteration builds a sequence of Q-functions specified by

$$\hat{Q}_{k+1}(s,a) = r(s,a) + \gamma \max_{a'} \hat{Q}_k(s', a')$$

estimated using supervised learning on transition data $$ (s, a, r, s') $$. Deep Q-networks use neural networks to approximate $$ Q^{\star}(s,a) $$, enabling application to high-dimensional state representations.

A critical challenge in clinical pathway optimization is that the state space includes not only clinical variables but also resources, care setting, and patient circumstances. For a patient in a rural setting without access to daily infusion centers, pathways requiring frequent IV therapy are infeasible regardless of clinical superiority. For patients facing transportation barriers, pathways demanding frequent clinic visits impose severe burdens. For patients with limited English proficiency, pathways requiring complex self-management may be inappropriate without language-concordant support.

To incorporate these constraints, we extend the MDP formulation to include resource availability $$ C \subseteq \mathcal{C} $$ as part of the state space, where $$ \mathcal{C} $$ is the set of all possible resources (specialist access, imaging modalities, medication formularies, social support). The action space becomes context-dependent: $$ \mathcal{A}(C) = \{a \in \mathcal{A} : \text{requirements}(a) \subseteq C\} $$ includes only actions whose required resources are available. The transition dynamics may depend on resource availability: a treatment requiring strict adherence may be less effective when administered without adherence support.

Equity-focused pathway optimization explicitly considers disparities in access and outcomes. We can formulate this as constrained optimization

$$\max_\pi \mathbb{E}_{s,a \sim \pi}[Q^\pi(s,a)]$$

subject to the constraints

$$\min_{g \in \mathcal{G}} \mathbb{E}_{s,a \sim \pi \mid G=g}[Q^\pi(s,a)] \geq \theta$$

ensuring that expected outcomes for all groups exceed a minimum threshold $$ \theta $$. Alternatively, we can incorporate equity directly into the reward function via

$$R_{\text{equity}}(s,a) = R(s,a) - \lambda \cdot \text{Var}_g[\mathbb{E}[Q^\pi(s,a) \mid G=g]]$$

penalizing policies that produce disparate outcomes across groups. This formulation incentivizes identifying pathways that work well across diverse care settings and patient circumstances.

## Shared Decision Making and Preference Elicitation

Shared decision making (SDM) is an approach to clinical consultation where clinicians and patients jointly deliberate about treatment options, considering best evidence alongside patient values and preferences. Implementing SDM through clinical decision support requires systems that elicit preferences, communicate uncertainty, and support deliberation rather than dictating recommendations.

Preference elicitation methods vary in cognitive demand and information requirements. Discrete choice experiments present patients with choice sets of treatment profiles defined by multiple attributes (efficacy, side effects, administration burden, cost). For two treatment options described by attributes $$ x_1 $$ and $$ x_2 $$, the probability patient $$ i $$ chooses option 1 is modeled as

$$P(y_i = 1) = \frac{\exp(\beta_i^T x_1)}{\exp(\beta_i^T x_1) + \exp(\beta_i^T x_2)}$$

where $$ \beta_i $$ represents individual-specific preference weights. By presenting multiple choice tasks with systematically varied attribute levels, we can estimate $$ \beta_i $$ and predict preferred treatment. Time trade-off (TTO) methods ask patients how many years of life in a disease state they would trade for fewer years in full health, quantifying quality-of-life weights. Standard gambles ask patients what mortality risk they would accept for a treatment that could restore full health, eliciting risk preferences.

However, these methods assume patients have stable, well-formed preferences that can be elicited through structured tasks. Behavioral economics research demonstrates that preferences are often constructed during elicitation, influenced by framing effects, default options, and the order in which information is presented. Patients may have limited understanding of probabilistic information, particularly when communicated using percentages rather than natural frequencies. For example, saying "20 out of 100 patients experience this side effect" is more interpretable than "there is a 20% chance of this side effect."

Cultural factors profoundly influence decision-making processes. In many cultures, family involvement in medical decisions is expected, yet standard SDM frameworks often assume individual patient autonomy. Some patients prefer physicians to make recommendations based on medical expertise rather than requiring patients to navigate complex trade-offs. For patients with limited health literacy, structured preference elicitation tasks may be overwhelming or misunderstood. Language barriers can prevent meaningful engagement with decision support tools.

An equity-focused approach to decision support acknowledges these cultural and contextual factors. Rather than imposing a single decision-making model, the system should adapt to patient and family preferences about involvement in decisions. Information should be communicated in multiple formats (verbal, visual, numerical) and languages, with complexity tailored to health literacy. Default recommendations can be provided for patients who prefer physician guidance, while detailed trade-off tools are available for those wanting greater control. Crucially, the system should help patients understand not just clinical trade-offs but also how social services, financial assistance, or care coordination could address barriers that constrain choices.

## Fairness Constraints and Equity Metrics

To ensure treatment recommendation systems promote rather than undermine health equity, we must define and operationalize fairness criteria. Multiple definitions of algorithmic fairness have been proposed, each capturing different ethical intuitions about equitable treatment.

Demographic parity requires that treatment recommendations are independent of protected attributes, imposing

$$P(T=1\mid G=g_1) = P(T=1 \mid G=g_2)$$

for all groups $$ g_1, g_2 $$. This ensures equal treatment rates but may be inappropriate if treatment needs genuinely differ across populations. Equalized odds requires that, conditional on the outcome, recommendations are independent of group membership, meaning

$$P(T=1\mid Y=y, G=g_1) = P(T=1 \mid Y=y, G=g_2)$$

This allows recommendation rates to differ across groups if base rates differ, but requires equal true positive and false positive rates. Predictive parity requires that the positive predictive value of recommendations is equal across groups, captured by

$$P(Y=1\mid T=1, G=g_1) = P(Y=1 \mid T=1, G=g_2)$$

ensuring that a recommendation means the same thing for all groups.

For treatment recommendations specifically, we should focus on outcome fairness by asking whether recommended treatments lead to equitable outcomes. This requires that expected outcomes conditional on covariates are similar across groups, so that

$$\mathbb{E}[Y_i(t^{\star}(X_i)) \mid X_i, G_i=g_1] \approx \mathbb{E}[Y_i(t^{\star}(X_i)) \mid X_i, G_i=g_2]$$

for recommended treatments $$ t^{\star}(X_i) $$. Note that this differs from requiring equal outcomes unconditionally, which would ignore that patients present with different clinical needs. Instead, we require that among patients with similar clinical presentations, recommended treatments lead to similar expected benefits regardless of group membership.

Calibration within groups is also critical. A recommendation system is calibrated if the predicted benefit of treatment matches the realized benefit, i.e.

$$\mathbb{E}[Y_i(t) - Y_i(0) \mid \hat{\tau}(X_i) = \tau, G_i=g] = \tau$$

for all groups $$ g $$. Poor calibration can lead to over-treatment or under-treatment of specific populations. We should validate calibration separately within each demographic subgroup and clinical population of interest.

To implement fairness constraints, we can modify the optimization objective when learning recommendation policies. For example, to satisfy equalized odds constraints in a treatment recommendation setting, we add Lagrange multipliers so that

$$\mathcal{L}(\pi, \lambda) = -\mathbb{E}[Y_i(t^\pi(X_i))] + \sum_{g,y} \lambda_{g,y} \lvert P(T=1 \mid Y=y,G=g) - P(T=1\mid Y=y) \rvert$$

and solve

$$\max_\pi \min_\lambda \mathcal{L}(\pi, \lambda)$$

The resulting policy balances outcome maximization with fairness constraints. Fair representation learning provides an alternative approach: learn representations $$ Z_i = h(X_i) $$ that are predictive of outcomes but statistically independent of sensitive attributes $$ G_i $$. Treatment recommendations based on $$ Z_i $$ cannot encode information about group membership beyond what is predictively relevant for outcomes.

Importantly, satisfying mathematical fairness constraints does not guarantee equitable outcomes if the data used to train systems embeds historical inequities. If certain populations have historically received worse care, their outcomes under historical treatment patterns may be poor, and a system trained to replicate these patterns will perpetuate disparities even if it satisfies fairness metrics. This highlights the need for causal, not just associative, modeling: we need to estimate what outcomes would be under equitable care, not just reproduce patterns from inequitable history.

## Implementation Considerations

```python
"""
Treatment Recommendation System with Equity-Aware Components
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TreatmentOption:
    """Represents a treatment option with attributes."""
    name: str
    efficacy_mean: float
    efficacy_std: float
    adverse_event_rate: float
    cost: float
    resource_requirements: List[str]
    administration_frequency: str


class CATEEstimator:
    """
    Estimates Conditional Average Treatment Effects using meta-learners.
    Implements S-learner, T-learner, and X-learner approaches.
    """

    def __init__(
        self,
        method: str = "x_learner",
        base_learner: Optional[object] = None,
        propensity_learner: Optional[object] = None,
        n_folds: int = 5
    ):
        """
        Initialize CATE estimator.

        Args:
            method: One of "s_learner", "t_learner", "x_learner"
            base_learner: sklearn-compatible regressor for outcome modeling
            propensity_learner: sklearn-compatible classifier for propensity modeling
            n_folds: Number of folds for cross-fitting to avoid overfitting
        """
        if method not in ["s_learner", "t_learner", "x_learner"]:
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.base_learner = base_learner or RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=20,
            random_state=42
        )
        self.propensity_learner = propensity_learner or GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
        self.n_folds = n_folds
        self.models_control: List[object] = []
        self.models_treatment: List[object] = []
        self.propensity_model: Optional[object] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
        sensitive_features: Optional[np.ndarray] = None
    ) -> 'CATEEstimator':
        """
        Fit CATE model using specified meta-learner.

        Args:
            X: Covariate matrix (n_samples, n_features)
            y: Observed outcomes (n_samples,)
            treatment: Treatment indicators (n_samples,), binary 0/1
            sensitive_features: Optional sensitive attributes for fairness monitoring

        Returns:
            Fitted estimator
        """
        if X.shape[0] != len(y) or X.shape[0] != len(treatment):
            raise ValueError("X, y, and treatment must have same number of samples")

        treatment = treatment.astype(int)

        if self.method == "s_learner":
            self._fit_s_learner(X, y, treatment)
        elif self.method == "t_learner":
            self._fit_t_learner(X, y, treatment)
        elif self.method == "x_learner":
            self._fit_x_learner(X, y, treatment)

        # Fit propensity model for uncertainty quantification
        self.propensity_model = type(self.propensity_learner)()
        self.propensity_model.fit(X, treatment)

        # Validate calibration within groups if sensitive features provided
        if sensitive_features is not None:
            self._validate_group_calibration(X, y, treatment, sensitive_features)

        return self

    def _fit_s_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray
    ) -> None:
        """Fit S-learner by building single model with treatment as feature."""
        X_augmented = np.column_stack([X, treatment])
        model = type(self.base_learner)()
        model.fit(X_augmented, y)
        self.models_treatment = [model]

    def _fit_t_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray
    ) -> None:
        """Fit T-learner by building separate models for each treatment group."""
        # Model for control group
        control_mask = treatment == 0
        model_control = type(self.base_learner)()
        model_control.fit(X[control_mask], y[control_mask])
        self.models_control = [model_control]

        # Model for treatment group
        treatment_mask = treatment == 1
        model_treatment = type(self.base_learner)()
        model_treatment.fit(X[treatment_mask], y[treatment_mask])
        self.models_treatment = [model_treatment]

    def _fit_x_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray
    ) -> None:
        """
        Fit X-learner using cross-fitting to avoid overfitting bias.
        """
        # Step 1: Fit outcome models for each treatment group
        control_mask = treatment == 0
        treatment_mask = treatment == 1

        model_control = type(self.base_learner)()
        model_control.fit(X[control_mask], y[control_mask])

        model_treatment = type(self.base_learner)()
        model_treatment.fit(X[treatment_mask], y[treatment_mask])

        # Step 2: Impute counterfactual outcomes and compute treatment effects
        # For treated: tau = Y - mu_0(X)
        y_imputed_control = model_control.predict(X[treatment_mask])
        tau_treatment = y[treatment_mask] - y_imputed_control

        # For controls: tau = mu_1(X) - Y
        y_imputed_treatment = model_treatment.predict(X[control_mask])
        tau_control = y_imputed_treatment - y[control_mask]

        # Step 3: Build models to predict imputed treatment effects
        model_tau_treatment = type(self.base_learner)()
        model_tau_treatment.fit(X[treatment_mask], tau_treatment)

        model_tau_control = type(self.base_learner)()
        model_tau_control.fit(X[control_mask], tau_control)

        self.models_treatment = [model_tau_treatment]
        self.models_control = [model_tau_control]

    def predict_cate(
        self,
        X: np.ndarray,
        return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict Conditional Average Treatment Effect.

        Args:
            X: Covariate matrix (n_samples, n_features)
            return_std: Whether to return standard errors

        Returns:
            CATE predictions and optionally standard errors
        """
        if self.method == "s_learner":
            X_treatment = np.column_stack([X, np.ones(len(X))])
            X_control = np.column_stack([X, np.zeros(len(X))])
            mu_1 = self.models_treatment[0].predict(X_treatment)
            mu_0 = self.models_treatment[0].predict(X_control)
            cate = mu_1 - mu_0

        elif self.method == "t_learner":
            mu_1 = self.models_treatment[0].predict(X)
            mu_0 = self.models_control[0].predict(X)
            cate = mu_1 - mu_0

        elif self.method == "x_learner":
            tau_1 = self.models_treatment[0].predict(X)
            tau_0 = self.models_control[0].predict(X)
            # Weight by propensity score
            propensity = self.propensity_model.predict(X)
            propensity = np.clip(propensity, 0.01, 0.99)  # Avoid extreme values
            cate = propensity * tau_0 + (1 - propensity) * tau_1

        if not return_std:
            return cate

        # Estimate standard errors using bootstrap or asymptotic approximation
        # Simplified: use propensity-based variance estimate
        propensity = self.propensity_model.predict(X)
        propensity = np.clip(propensity, 0.01, 0.99)

        # Variance scales with 1/(p*(1-p))
        variance = 1.0 / (propensity * (1 - propensity))
        std = np.sqrt(variance)

        return cate, std

    def _validate_group_calibration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
        sensitive_features: np.ndarray
    ) -> None:
        """Validate calibration within demographic groups."""
        unique_groups = np.unique(sensitive_features)

        logger.info("Validating CATE calibration within groups:")

        for group in unique_groups:
            group_mask = sensitive_features == group
            X_group = X[group_mask]
            y_group = y[group_mask]
            treatment_group = treatment[group_mask]

            # Predict CATE for group
            cate_pred = self.predict_cate(X_group)

            # Estimate realized treatment effect
            treated_mask = treatment_group == 1
            control_mask = treatment_group == 0

            if np.sum(treated_mask) > 0 and np.sum(control_mask) > 0:
                realized_effect = np.mean(y_group[treated_mask]) - np.mean(y_group[control_mask])
                predicted_effect = np.mean(cate_pred)

                logger.info(
                    f"Group {group}: Predicted CATE = {predicted_effect:.3f}, "
                    f"Realized ATE = {realized_effect:.3f}"
                )

class EquityAwareTreatmentRecommender:
    """
    Treatment recommendation system with explicit equity constraints.
    """

    def __init__(
        self,
        cate_estimator: CATEEstimator,
        treatments: List[TreatmentOption],
        fairness_constraint: str = "equalized_odds",
        fairness_tolerance: float = 0.1
    ):
        """
        Initialize recommender.

        Args:
            cate_estimator: Fitted CATE estimator
            treatments: List of available treatments
            fairness_constraint: Type of fairness constraint to enforce
            fairness_tolerance: Maximum allowable disparity
        """
        self.cate_estimator = cate_estimator
        self.treatments = treatments
        self.fairness_constraint = fairness_constraint
        self.fairness_tolerance = fairness_tolerance

    def recommend(
        self,
        X: np.ndarray,
        sensitive_features: np.ndarray,
        resource_constraints: Optional[List[str]] = None,
        patient_preferences: Optional[Dict[str, float]] = None
    ) -> Tuple[List[int], np.ndarray]:
        """
        Recommend treatments with equity constraints.

        Args:
            X: Patient covariates (n_patients, n_features)
            sensitive_features: Sensitive attributes for fairness monitoring
            resource_constraints: List of available resources
            patient_preferences: Dictionary of preference weights

        Returns:
            Recommended treatment indices and confidence scores
        """
        n_patients = X.shape[0]

        # Predict CATE for each treatment
        cate_predictions = []
        cate_std = []

        for treatment in self.treatments:
            cate, std = self.cate_estimator.predict_cate(X, return_std=True)
            cate_predictions.append(cate)
            cate_std.append(std)

        cate_predictions = np.array(cate_predictions).T  # (n_patients, n_treatments)
        cate_std = np.array(cate_std).T

        # Filter by resource constraints
        feasible_treatments = self._filter_by_resources(resource_constraints)

        # Apply patient preferences if provided
        if patient_preferences is not None:
            cate_predictions = self._apply_preferences(
                cate_predictions,
                patient_preferences
            )

        # Select treatments maximizing expected benefit
        recommendations = np.argmax(cate_predictions[:, feasible_treatments], axis=1)
        recommendations = feasible_treatments[recommendations]

        # Check fairness constraints
        self._check_fairness_constraints(
            recommendations,
            cate_predictions,
            sensitive_features
        )

        # Compute confidence scores
        confidence = self._compute_confidence(
            recommendations,
            cate_predictions,
            cate_std
        )

        return recommendations, confidence

    def _filter_by_resources(
        self,
        available_resources: Optional[List[str]]
    ) -> List[int]:
        """Filter treatments by available resources."""
        if available_resources is None:
            return list(range(len(self.treatments)))

        feasible = []
        for idx, treatment in enumerate(self.treatments):
            if all(req in available_resources for req in treatment.resource_requirements):
                feasible.append(idx)

        if not feasible:
            logger.warning("No treatments feasible with available resources")
            # Fall back to treatments with minimal requirements
            min_requirements = min(
                len(t.resource_requirements) for t in self.treatments
            )
            feasible = [
                idx for idx, t in enumerate(self.treatments)
                if len(t.resource_requirements) == min_requirements
            ]

        return feasible

    def _apply_preferences(
        self,
        cate_predictions: np.ndarray,
        preferences: Dict[str, float]
    ) -> np.ndarray:
        """
        Adjust CATE predictions based on patient preferences.

        Preferences might include:
        - side_effect_weight: Penalty for adverse events
        - cost_weight: Penalty for out-of-pocket costs
        - convenience_weight: Penalty for frequent administration
        """
        adjusted = cate_predictions.copy()

        for idx, treatment in enumerate(self.treatments):
            # Penalty for adverse events
            if "side_effect_weight" in preferences:
                adjusted[:, idx] -= (
                    preferences["side_effect_weight"] * treatment.adverse_event_rate
                )

            # Penalty for cost
            if "cost_weight" in preferences:
                adjusted[:, idx] -= preferences["cost_weight"] * treatment.cost

            # Penalty for inconvenient administration
            if "convenience_weight" in preferences:
                frequency_penalty = {
                    "daily": 0.3,
                    "weekly": 0.1,
                    "monthly": 0.0
                }.get(treatment.administration_frequency, 0.0)
                adjusted[:, idx] -= preferences["convenience_weight"] * frequency_penalty

        return adjusted

    def _check_fairness_constraints(
        self,
        recommendations: np.ndarray,
        cate_predictions: np.ndarray,
        sensitive_features: np.ndarray
    ) -> None:
        """Check if recommendations satisfy fairness constraints."""
        unique_groups = np.unique(sensitive_features)

        if self.fairness_constraint == "equalized_odds":
            # Check that treatment recommendation rates are similar across groups
            # conditional on predicted benefit

            # Discretize predicted benefit into quartiles
            benefit_quartiles = np.percentile(
                cate_predictions[np.arange(len(recommendations)), recommendations],
                [25, 50, 75]
            )

            for q_low, q_high in zip(
                [-np.inf] + benefit_quartiles.tolist(),
                benefit_quartiles.tolist() + [np.inf]
            ):
                rates = {}
                for group in unique_groups:
                    group_mask = sensitive_features == group
                    benefit_mask = (
                        (cate_predictions[np.arange(len(recommendations)), recommendations] >= q_low) &
                        (cate_predictions[np.arange(len(recommendations)), recommendations] < q_high)
                    )
                    mask = group_mask & benefit_mask

                    if np.sum(mask) > 0:
                        # Calculate rate of recommending most beneficial treatment
                        rates[group] = np.mean(
                            recommendations[mask] == np.argmax(
                                cate_predictions[mask], axis=1
                            )
                        )

                if len(rates) > 1:
                    max_disparity = max(rates.values()) - min(rates.values())
                    if max_disparity > self.fairness_tolerance:
                        logger.warning(
                            f"Fairness constraint violation: disparity = {max_disparity:.3f} "
                            f"in benefit range [{q_low:.2f}, {q_high:.2f})"
                        )

        elif self.fairness_constraint == "outcome_parity":
            # Check that expected outcomes are similar across groups
            expected_outcomes = {}

            for group in unique_groups:
                group_mask = sensitive_features == group
                group_cate = cate_predictions[
                    group_mask,
                    recommendations[group_mask]
                ]
                expected_outcomes[group] = np.mean(group_cate)

            if len(expected_outcomes) > 1:
                max_disparity = (
                    max(expected_outcomes.values()) -
                    min(expected_outcomes.values())
                )

                if max_disparity > self.fairness_tolerance:
                    logger.warning(
                        f"Outcome disparity detected: {max_disparity:.3f} "
                        f"across groups"
                    )

    def _compute_confidence(
        self,
        recommendations: np.ndarray,
        cate_predictions: np.ndarray,
        cate_std: np.ndarray
    ) -> np.ndarray:
        """
        Compute confidence scores for recommendations.

        Confidence is high when:
        1. Predicted benefit is large
        2. Uncertainty is low
        3. Recommended treatment is clearly better than alternatives
        """
        n_patients = len(recommendations)
        confidence = np.zeros(n_patients)

        for i in range(n_patients):
            rec = recommendations[i]

            # Predicted benefit
            benefit = cate_predictions[i, rec]

            # Uncertainty
            uncertainty = cate_std[i, rec]

            # Margin over second-best treatment
            sorted_benefits = np.sort(cate_predictions[i])
            margin = sorted_benefits[-1] - sorted_benefits[-2]

            # Confidence is high when benefit/uncertainty ratio is large
            # and margin is large
            confidence[i] = (benefit / (uncertainty + 1e-6)) * margin

        # Normalize to [0, 1]
        confidence = (confidence - confidence.min()) / (
            confidence.max() - confidence.min() + 1e-6
        )

        return confidence

class PharmacokineticDosingOptimizer:
    """
    Optimize medication dosing using pharmacokinetic models.
    """

    def __init__(
        self,
        pk_model: Callable,
        pd_model: Callable,
        target_concentration: float,
        safety_margin: float = 1.5
    ):
        """
        Initialize dosing optimizer.

        Args:
            pk_model: Function mapping (dose, patient_params, time) -> concentration
            pd_model: Function mapping concentration -> effect
            target_concentration: Target therapeutic concentration
            safety_margin: Factor above target that triggers toxicity concern
        """
        self.pk_model = pk_model
        self.pd_model = pd_model
        self.target_concentration = target_concentration
        self.safety_margin = safety_margin

    def optimize_dose(
        self,
        patient_params: Dict[str, float],
        dosing_interval: float = 24.0,
        n_doses: int = 7
    ) -> Dict[str, float]:
        """
        Optimize dosing regimen for individual patient.

        Args:
            patient_params: Patient-specific PK/PD parameters
            dosing_interval: Time between doses (hours)
            n_doses: Number of doses to optimize

        Returns:
            Dictionary with optimal dose and predicted concentrations
        """

        def objective(dose: float) -> float:
            """
            Objective penalizes deviation from target and risk of toxicity.
            """
            times = np.linspace(0, n_doses * dosing_interval, 100)
            concentrations = np.array([
                self.pk_model(dose, patient_params, t) for t in times
            ])

            # Penalty for deviation from target
            target_penalty = np.mean((concentrations - self.target_concentration) ** 2)

            # Penalty for exceeding safety threshold
            max_safe_concentration = self.target_concentration * self.safety_margin
            toxicity_penalty = np.sum(
                np.maximum(0, concentrations - max_safe_concentration) ** 2
            )

            return target_penalty + 10.0 * toxicity_penalty

        # Optimize dose
        result = minimize(
            objective,
            x0=100.0,  # Initial guess
            bounds=[(10.0, 1000.0)],  # Reasonable dose range
            method='L-BFGS-B'
        )

        optimal_dose = result.x[0]

        # Compute steady-state concentrations
        times = np.linspace(0, n_doses * dosing_interval, 100)
        concentrations = np.array([
            self.pk_model(optimal_dose, patient_params, t) for t in times
        ])

        return {
            'optimal_dose': optimal_dose,
            'mean_concentration': np.mean(concentrations),
            'max_concentration': np.max(concentrations),
            'min_concentration': np.min(concentrations),
            'time_above_target': np.mean(concentrations >= self.target_concentration),
            'toxicity_risk': np.mean(
                concentrations >= self.target_concentration * self.safety_margin
            )
        }

def example_pk_model(
    dose: float,
    params: Dict[str, float],
    time: float
) -> float:
    """
    Example one-compartment PK model with first-order elimination.

    Args:
        dose: Administered dose (mg)
        params: Dictionary with 'volume' (L) and 'clearance' (L/h)
        time: Time since dose (hours)

    Returns:
        Plasma concentration (mg/L)
    """
    volume = params.get('volume', 50.0)  # Default 50L
    clearance = params.get('clearance', 5.0)  # Default 5 L/h

    # Elimination rate constant
    k_e = clearance / volume

    # Concentration at time t
    c_0 = dose / volume  # Initial concentration
    concentration = c_0 * np.exp(-k_e * time)

    return concentration

def example_pd_model(concentration: float) -> float:
    """
    Example PD model: Emax model relating concentration to effect.

    Args:
        concentration: Drug concentration (mg/L)

    Returns:
        Effect (0 to 1 scale)
    """
    e_max = 1.0  # Maximum effect
    ec_50 = 10.0  # Concentration producing 50% of maximum effect
    gamma = 2.0  # Hill coefficient

    effect = e_max * (concentration ** gamma) / (ec_50 ** gamma + concentration ** gamma)

    return effect

def evaluate_treatment_recommendations(
    recommendations: np.ndarray,
    ground_truth_outcomes: np.ndarray,
    sensitive_features: np.ndarray,
    treatment_costs: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive evaluation of treatment recommendations.

    Args:
        recommendations: Recommended treatment indices (n_patients,)
        ground_truth_outcomes: Actual outcomes for each patient-treatment pair
            (n_patients, n_treatments)
        sensitive_features: Sensitive attributes (n_patients,)
        treatment_costs: Costs for each treatment (n_treatments,)

    Returns:
        Dictionary of evaluation metrics
    """
    n_patients = len(recommendations)

    # Regret: difference between outcome of optimal treatment and recommended treatment
    optimal_treatments = np.argmax(ground_truth_outcomes, axis=1)
    optimal_outcomes = ground_truth_outcomes[
        np.arange(n_patients),
        optimal_treatments
    ]
    recommended_outcomes = ground_truth_outcomes[
        np.arange(n_patients),
        recommendations
    ]

    regret = optimal_outcomes - recommended_outcomes
    mean_regret = np.mean(regret)

    # Accuracy: fraction of times recommending optimal treatment
    accuracy = np.mean(recommendations == optimal_treatments)

    # Cost analysis
    recommended_costs = treatment_costs[recommendations]
    optimal_costs = treatment_costs[optimal_treatments]

    # Group-specific metrics
    unique_groups = np.unique(sensitive_features)
    group_metrics = {}

    for group in unique_groups:
        group_mask = sensitive_features == group
        group_regret = np.mean(regret[group_mask])
        group_accuracy = np.mean(
            recommendations[group_mask] == optimal_treatments[group_mask]
        )
        group_cost = np.mean(recommended_costs[group_mask])

        group_metrics[f"group_{group}_regret"] = group_regret
        group_metrics[f"group_{group}_accuracy"] = group_accuracy
        group_metrics[f"group_{group}_cost"] = group_cost

    # Fairness metrics
    group_regrets = [
        np.mean(regret[sensitive_features == g]) for g in unique_groups
    ]
    regret_disparity = max(group_regrets) - min(group_regrets)

    group_accuracies = [
        np.mean(recommendations[sensitive_features == g] == optimal_treatments[sensitive_features == g])
        for g in unique_groups
    ]
    accuracy_disparity = max(group_accuracies) - min(group_accuracies)

    return {
        "mean_regret": mean_regret,
        "accuracy": accuracy,
        "mean_recommended_cost": np.mean(recommended_costs),
        "mean_optimal_cost": np.mean(optimal_costs),
        "regret_disparity": regret_disparity,
        "accuracy_disparity": accuracy_disparity,
        **group_metrics
    }

# Example usage demonstrating complete workflow
if __name__ == "__main__":
    # Simulate dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Patient covariates
    X = np.random.randn(n_samples, n_features)

    # Sensitive attribute (e.g., demographic group)
    sensitive_features = np.random.choice([0, 1, 2], size=n_samples)

    # Treatment assignment (binary for this example)
    treatment = np.random.binomial(1, 0.5, size=n_samples)

    # Simulate outcomes with heterogeneous treatment effects
    # Treatment effect depends on covariates
    baseline = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_samples) * 0.5
    treatment_effect = 0.5 + 0.3 * X[:, 2] - 0.2 * X[:, 3]
    outcomes = baseline + treatment * treatment_effect

    # Fit CATE estimator
    print("Fitting CATE estimator...")
    cate_estimator = CATEEstimator(method="x_learner")
    cate_estimator.fit(X, outcomes, treatment, sensitive_features)

    # Predict treatment effects
    cate_pred, cate_std = cate_estimator.predict_cate(X, return_std=True)

    print(f"\nMean predicted CATE: {np.mean(cate_pred):.3f}")
    print(f"Mean CATE uncertainty: {np.mean(cate_std):.3f}")

    # Define treatment options
    treatments = [
        TreatmentOption(
            name="Treatment A",
            efficacy_mean=0.5,
            efficacy_std=0.1,
            adverse_event_rate=0.1,
            cost=100.0,
            resource_requirements=["basic_lab"],
            administration_frequency="weekly"
        ),
        TreatmentOption(
            name="Treatment B",
            efficacy_mean=0.7,
            efficacy_std=0.15,
            adverse_event_rate=0.2,
            cost=500.0,
            resource_requirements=["basic_lab", "specialist"],
            administration_frequency="daily"
        )
    ]

    # Initialize recommender
    print("\nInitializing treatment recommender...")
    recommender = EquityAwareTreatmentRecommender(
        cate_estimator=cate_estimator,
        treatments=treatments,
        fairness_constraint="outcome_parity",
        fairness_tolerance=0.15
    )

    # Generate recommendations
    print("\nGenerating treatment recommendations...")
    recommendations, confidence = recommender.recommend(
        X=X,
        sensitive_features=sensitive_features,
        resource_constraints=["basic_lab", "specialist"],
        patient_preferences={
            "side_effect_weight": 0.3,
            "cost_weight": 0.001,
            "convenience_weight": 0.2
        }
    )

    print(f"Mean confidence score: {np.mean(confidence):.3f}")
    print(f"Treatment A recommended for {np.mean(recommendations == 0):.1%} of patients")
    print(f"Treatment B recommended for {np.mean(recommendations == 1):.1%} of patients")

    # Evaluate equity
    for group in np.unique(sensitive_features):
        group_mask = sensitive_features == group
        print(f"\nGroup {group}:")
        print(f"  Treatment A rate: {np.mean(recommendations[group_mask] == 0):.1%}")
        print(f"  Treatment B rate: {np.mean(recommendations[group_mask] == 1):.1%}")
        print(f"  Mean confidence: {np.mean(confidence[group_mask]):.3f}")

    # Demonstrate PK/PD dosing optimization
    print("\n" + "="*60)
    print("Pharmacokinetic Dosing Optimization Example")
    print("="*60)

    # Patient parameters varying by demographic factors
    # For example, clearance may vary with renal function
    patient_params_examples = [
        {"volume": 50.0, "clearance": 5.0},  # Normal renal function
        {"volume": 50.0, "clearance": 2.5},  # Reduced renal function
        {"volume": 70.0, "clearance": 7.0},  # Larger patient
    ]

    dosing_optimizer = PharmacokineticDosingOptimizer(
        pk_model=example_pk_model,
        pd_model=example_pd_model,
        target_concentration=15.0,
        safety_margin=1.5
    )

    for i, params in enumerate(patient_params_examples):
        print(f"\nPatient {i+1}: Volume={params['volume']}L, Clearance={params['clearance']}L/h")
        result = dosing_optimizer.optimize_dose(
            patient_params=params,
            dosing_interval=24.0,
            n_doses=7
        )
        print(f"  Optimal dose: {result['optimal_dose']:.1f} mg")
        print(f"  Mean concentration: {result['mean_concentration']:.2f} mg/L")
        print(f"  Time above target: {result['time_above_target']:.1%}")
        print(f"  Toxicity risk: {result['toxicity_risk']:.1%}")
```

## Regulatory Considerations and Validation

Treatment recommendation systems are considered Software as a Medical Device (SaMD) by regulatory agencies including the FDA, EMA, and other national authorities. The regulatory pathway depends on the risk classification of the device, determined by the severity of the condition being treated and the degree of autonomy of the system. Systems that recommend treatments for serious or life-threatening conditions, or that operate with minimal clinician oversight, are classified as higher risk and require more stringent validation and approval processes.

The FDA's predetermined change control plans provide a framework for continuously updating AI-based medical devices while maintaining regulatory compliance. This is particularly important for treatment recommendation systems given the need to incorporate new evidence, update to reflect changing treatment standards, and address identified biases or safety concerns. The predetermined change control plan specifies the types of modifications that can be implemented without requiring new regulatory submission (SaMD Pre-Specifications) and the protocol for validating these modifications (Algorithm Change Protocol).

Validation of treatment recommendation systems must demonstrate both clinical efficacy and fairness across populations. Clinical validation requires showing that implemented recommendations lead to improved patient outcomes compared to standard care, ideally through randomized controlled trials or high-quality observational studies with appropriate causal inference methods. Importantly, trials must include adequate representation of underserved populations to ensure that benefits are realized equitably. Post-market surveillance is essential to monitor for differential performance across populations and to detect if the system performs worse than anticipated in real-world settings with greater patient diversity than development datasets.

Fairness validation requires demonstrating that the system does not produce discriminatory recommendations. This includes testing for disparate impact (differential recommendation rates across groups after accounting for clinical need), disparate treatment (use of protected attributes in making recommendations), and disparate outcomes (differential clinical outcomes across groups among patients receiving recommendations). Importantly, fairness must be assessed not just in development data but in external validation cohorts representing the populations where the system will be deployed.

Documentation requirements for treatment recommendation systems include comprehensive description of the training data including demographic composition and any known limitations in population representation, detailed explanation of the algorithms and models including how treatment effects are estimated and uncertainty is quantified, validation results stratified by demographic groups and clinical subpopulations, intended use statement clearly specifying the populations and care settings where the system should be used, and risk mitigation strategies addressing identified limitations or biases. For systems deployed in under-resourced settings, documentation must address how the system accounts for resource constraints and whether recommendations remain appropriate when certain interventions are unavailable.

## Ethical Frameworks and Societal Implications

Treatment recommendation systems raise profound ethical questions about the appropriate role of AI in medical decision-making, patient autonomy, and health equity. The principle of beneficence requires that recommendations promote patient welfare, while non-maleficence demands that systems do not cause harm through inappropriate recommendations or by perpetuating treatment disparities. Respect for autonomy requires that systems support rather than replace patient decision-making, providing information to enable informed choices rather than dictating treatment paths. Justice requires that systems promote equitable access to effective treatments and do not systematically disadvantage marginalized populations.

A significant ethical concern is that optimization of treatment recommendations at the individual level may conflict with population-level equity goals. For example, precision medicine approaches that tailor treatments to individual genetic profiles may be most effective for populations with extensive genetic data, potentially widening disparities if these populations already have better health outcomes. Systems that optimize for overall population benefit may inadvertently allocate more effective treatments to patients with higher baseline prognosis, a phenomenon known as "predictive equality" where those predicted to benefit most receive superior interventions. Ethical treatment recommendation systems must explicitly weigh individual versus collective welfare and incorporate constraints that prevent optimizing individual outcomes at the expense of exacerbating population-level inequities.

The transparency and explainability of treatment recommendations is essential for respecting patient autonomy and enabling informed consent. Patients should understand why a treatment was recommended, what alternatives exist, and how the recommendation relates to their values and preferences. However, complex machine learning models may obscure the reasoning behind recommendations, reducing them to opaque "black box" predictions. This is particularly problematic when recommendations differ from what clinicians would suggest or when they incorporate non-clinical factors such as cost or resource availability. Efforts to develop interpretable recommendation systems include using inherently interpretable models (decision trees, rule lists, linear models), post-hoc explanation methods (SHAP values, counterfactual explanations), or hybrid approaches where AI provides decision support but final decisions remain with clinicians who can explain their reasoning to patients.

The social determinants of health fundamentally shape which treatments are accessible and feasible for different populations, yet treatment recommendation systems often focus narrowly on clinical factors while ignoring social context. A treatment requiring daily clinic visits is infeasible for patients facing transportation barriers or employment inflexibility. Medications with high out-of-pocket costs are inaccessible regardless of efficacy for patients facing financial constraints. Dietary interventions assume access to healthy foods, and exercise recommendations assume safe spaces for physical activity. Recommendation systems that ignore these structural barriers effectively recommend treatments that cannot be implemented, creating a gap between algorithmic prescriptions and lived reality. Ethical systems must explicitly model resource availability, financial constraints, and social support, adapting recommendations to what is actually feasible rather than optimizing for an idealized clinical scenario divorced from patients' lives.

The deployment of treatment recommendation systems may itself be inequitable, with advanced decision support tools more readily available in well-resourced healthcare settings while under-resourced safety-net institutions lack the technical infrastructure to implement them. This creates a perverse situation where populations most likely to benefit from decision support due to healthcare access barriers are least likely to receive it. Addressing this requires intentional efforts to ensure equitable deployment including open-source implementations, partnerships with safety-net institutions, and funding mechanisms that support implementation in resource-constrained settings.

## Future Directions and Research Opportunities

Treatment recommendation represents a rapidly evolving field with numerous opportunities for methodological innovation and real-world impact. Several key research directions merit attention from both technical and equity perspectives.

First, causal inference methods for treatment effect heterogeneity estimation continue to improve, with recent developments in doubly robust machine learning, targeted minimum loss estimation, and causal forests providing more accurate and efficient CATE estimates. Future work should extend these methods to handle multiple competing treatments, continuous treatment dosages, and longitudinal treatment sequences while maintaining valid causal inference in the presence of time-varying confounding. Particular attention is needed for settings with limited randomized trial data, developing methods to combine observational data with experimental evidence and to extrapolate from well-studied populations to underrepresented groups.

Second, multi-objective optimization frameworks that explicitly balance clinical efficacy, safety, equity, cost, and patient preferences remain underdeveloped. Most existing work optimizes single objectives or requires specifying preference weights a priori. Future research should develop adaptive methods that learn optimal trade-offs from patient choices and clinician decisions, allowing the system to reflect diverse values without imposing a single preference structure. Evolutionary multi-objective optimization and Pareto frontier estimation methods from operations research could be adapted to the clinical context, generating a set of non-dominated recommendations and supporting deliberation about trade-offs rather than prescribing a single optimal choice.

Third, incorporating uncertainty quantification throughout the treatment recommendation pipeline is essential but technically challenging. Uncertainty arises from finite sample sizes in training data, model misspecification, unmeasured confounding, and extrapolation to novel populations. Bayesian methods provide a principled framework for uncertainty quantification but can be computationally intensive for complex models. Conformal prediction offers a distribution-free alternative for constructing prediction sets with coverage guarantees. Future work should develop computationally efficient uncertainty quantification methods tailored to causal inference tasks and determine how to communicate uncertainty to clinicians and patients in ways that support decision-making without overwhelming with information.

Fourth, adaptive treatment strategies and reinforcement learning for sequential treatment decisions represent promising but underutilized approaches in clinical medicine. Most chronic diseases involve sequences of treatment decisions made over months or years, with each decision informed by response to prior treatments and evolution of disease state. Optimal treatment sequences can be learned using Q-learning, policy gradient methods, or offline reinforcement learning from electronic health record data. However, challenges include credit assignment in sparse reward settings, off-policy evaluation to assess proposed policies using data from different policies, and ensuring safety constraints are satisfied throughout learning. Research is needed to develop reinforcement learning methods specifically designed for clinical contexts with high stakes, delayed and censored outcomes, and strong safety requirements.

Fifth, preference elicitation and shared decision-making remain difficult to implement at scale despite their recognized importance. Most patients receive little time with clinicians to discuss treatment options, and preference elicitation tools are rarely integrated into clinical workflows. Natural language processing and conversational AI could enable asynchronous preference elicitation where patients interact with an AI system before clinical visits to explore treatment options, express values, and identify priority questions. However, this requires ensuring that AI systems present information fairly without steering patients toward preferred choices and that they are accessible to patients with varying literacy levels and language backgrounds. Research should focus on developing conversational agents for preference elicitation that are validated for accuracy, cultural responsiveness, and equity across populations.

Finally, the integration of social determinants of health into treatment recommendation systems is at an early stage. While there is widespread recognition that social factors shape treatment access and effectiveness, few systems explicitly model these factors or adapt recommendations accordingly. Future work should develop data collection methods to systematically capture social determinants including housing stability, food security, social support, transportation access, and financial constraints. Recommendation algorithms should then incorporate these factors through constraint-based optimization (excluding infeasible treatments), outcome adjustment (predicting effectiveness conditional on social context), or intervention expansion (recommending social services alongside clinical treatments). This requires interdisciplinary collaboration between data scientists, clinicians, and social scientists to ensure that social determinants are represented accurately and respectfully.

## Conclusion

Treatment recommendation systems represent the application of artificial intelligence most directly linked to clinical outcomes, with the potential to personalize treatments, optimize dosing, improve adherence to evidence-based guidelines, and reduce unwarranted variation in care quality. However, the high-stakes nature of treatment decisions demands that these systems be developed with rigorous attention to causal inference, uncertainty quantification, patient preferences, and health equity. The technical challenges are formidable: estimating individual treatment effects from observational data, balancing multiple competing objectives, incorporating social determinants and resource constraints, and validating fairness across diverse populations. Yet these challenges are surmountable through careful application of causal inference methods, multi-objective optimization, preference modeling, and equity-focused validation.

The key principle is that equity cannot be an afterthought but must be central to system design from the outset. This requires diverse, representative training data; algorithms that explicitly model treatment effect heterogeneity across populations; fairness constraints and outcome parity metrics; preference elicitation that respects cultural diversity; resource-aware optimization that adapts to care setting constraints; and comprehensive validation stratified by demographic groups and clinical subpopulations. Treatment recommendation systems should expand rather than constrain patient choice, support rather than replace clinician expertise, and reduce rather than perpetuate health disparities.

As these systems move from research prototypes to clinical deployment, the data science community has an opportunity and obligation to ensure they serve all populations equitably. This requires technical innovation in causal inference and optimization, but also humility about the limits of algorithmic decision-making, engagement with affected communities, and commitment to ongoing monitoring and improvement. The goal is not to replace human judgment but to augment it, providing clinicians and patients with evidence-based recommendations that respect autonomy, promote justice, and ultimately improve health outcomes for all.

## Bibliography

Abadie A, Cattaneo MD. (2018). Econometric methods for program evaluation. *Annual Review of Economics*, 10:465-503.

Abernethy AP, Etheredge LM, Ganz PA, et al. (2010). Rapid-learning system for cancer care. *Journal of Clinical Oncology*, 28(27):4268-4274.

Aggarwal CC. (2016). *Recommender Systems: The Textbook*. Springer.

Agrawal R, Prabakaran S. (2020). Big data in digital healthcare: lessons learnt and recommendations for general practice. *Heredity*, 124(4):525-534.

Ahmed I, Deb K, Jindal A, et al. (2016). Multi-objective optimization and decision making approaches to cricket team selection. *Applied Soft Computing*, 13(1):402-414.

Angrist JD, Pischke JS. (2008). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.

Ashoori M, Weisz JD. (2019). In AI we trust? Factors that influence trustworthiness of AI-infused decision-making processes. *arXiv preprint* arXiv:1912.02675.

Athey S, Imbens GW. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27):7353-7360.

Athey S, Tibshirani J, Wager S. (2019). Generalized random forests. *Annals of Statistics*, 47(2):1148-1178.

Barocas S, Hardt M, Narayanan A. (2019). *Fairness and Machine Learning: Limitations and Opportunities*. fairmlbook.org.

Bennett CC, Hauser K. (2013). Artificial intelligence framework for simulating clinical decision-making: A Markov decision process approach. *Artificial Intelligence in Medicine*, 57(1):9-19.

Berlin JA, Santanna J, Schmid CH, et al. (2002). Individual patient-versus group-level data meta-regressions for the investigation of treatment effect modifiers. *Statistics in Medicine*, 21(18):2619-2631.

Beygelzimer A, Langford J. (2009). The offset tree for learning with partial labels. *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 129-138.

Bibault JE, Giraud P, Burgun A. (2016). Big data and machine learning in radiation oncology. *Cancer Letters*, 382(1):110-117.

Bottou L, Peters J, Quiñonero-Candela J, et al. (2013). Counterfactual reasoning and learning systems: The example of computational advertising. *Journal of Machine Learning Research*, 14(1):3207-3260.

Brock DW, Wartman SA. (2009). When competent patients make irrational choices. *New England Journal of Medicine*, 322(22):1595-1599.

Buolamwini J, Gebru T. (2018). Gender shades: Intersectional accuracy disparities in commercial gender classification. *Proceedings of Machine Learning Research*, 81:77-91.

Char DS, Shah NH, Magnus D. (2018). Implementing machine learning in health care - addressing ethical challenges. *New England Journal of Medicine*, 378(11):981-983.

Chernozhukov V, Chetverikov D, Demirer M, et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1):C1-C68.

Coiera E. (2015). *Guide to Health Informatics* (3rd ed.). CRC Press.

Collins FS, Varmus H. (2015). A new initiative on precision medicine. *New England Journal of Medicine*, 372(9):793-795.

Darcy AM, Louie AK, Roberts LW. (2016). Machine learning and the profession of medicine. *JAMA*, 315(6):551-552.

Deb K, Pratap A, Agarwal S, Meyarivan T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2):182-197.

Doshi-Velez F, Kim B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv preprint* arXiv:1702.08608.

Dudley JT, Listgarten J, Stegle O, et al. (2010). Personalized medicine: From genotypes, molecular phenotypes and the quantified self, towards improved medicine. *Pacific Symposium on Biocomputing*, 15:342-346.

Elwyn G, Frosch D, Thomson R, et al. (2012). Shared decision making: A model for clinical practice. *Journal of General Internal Medicine*, 27(10):1361-1367.

Emanuel EJ, Wendler D, Grady C. (2000). What makes clinical research ethical? *JAMA*, 283(20):2701-2711.

Esteva A, Kuprel B, Novoa RA, et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639):115-118.

Ferlay J, Soerjomataram I, Dikshit R, et al. (2015). Cancer incidence and mortality worldwide: Sources, methods and major patterns in GLOBOCAN 2012. *International Journal of Cancer*, 136(5):E359-E386.

Forrow L, Taylor WC, Arnold RM. (1992). Absolutely relative: How research results are summarized can affect treatment decisions. *American Journal of Medicine*, 92(2):121-124.

Foster JC, Taylor JMG, Ruberg SJ. (2011). Subgroup identification from randomized clinical trial data. *Statistics in Medicine*, 30(24):2867-2880.

Frieden TR. (2017). Evidence for health decision making - beyond randomized, controlled trials. *New England Journal of Medicine*, 377(5):465-475.

Futoma J, Simons M, Panch T, et al. (2020). The myth of generalisability in clinical research and machine learning in health care. *Lancet Digital Health*, 2(9):e489-e492.

Gianfrancesco MA, Tamang S, Yazdany J, Schmajuk G. (2018). Potential biases in machine learning algorithms using electronic health record data. *JAMA Internal Medicine*, 178(11):1544-1547.

Glasgow RE, Lichtenstein E, Marcus AC. (2003). Why don't we see more translation of health promotion research to practice? Rethinking the efficacy-to-effectiveness transition. *American Journal of Public Health*, 93(8):1261-1267.

Gottesman O, Johansson F, Komorowski M, et al. (2019). Guidelines for reinforcement learning in healthcare. *Nature Medicine*, 25(1):16-18.

Gulshan V, Peng L, Coram M, et al. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*, 316(22):2402-2410.

Hamburg MA, Collins FS. (2010). The path to personalized medicine. *New England Journal of Medicine*, 363(4):301-304.

Hardt M, Price E, Srebro N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems*, 29:3315-3323.

Hernán MA, Robins JM. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.

Hill JL, Su YS. (2013). Assessing lack of common support in causal inference using Bayesian nonparametrics. *Annals of Applied Statistics*, 7(3):1386-1420.

Holland PW. (1986). Statistics and causal inference. *Journal of the American Statistical Association*, 81(396):945-960.

Holzinger A, Langs G, Denk H, et al. (2019). Causability and explainability of artificial intelligence in medicine. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 9(4):e1312.

Horvitz DG, Thompson DJ. (1952). A generalization of sampling without replacement from a finite universe. *Journal of the American Statistical Association*, 47(260):663-685.

Huang Y, Li W, Macheret F, et al. (2018). A tutorial on calibration measurements and calibration models for clinical prediction models. *Journal of the American Medical Informatics Association*, 27(4):621-633.

Imai K, Ratkovic M. (2013). Estimating treatment effect heterogeneity in randomized program evaluation. *Annals of Applied Statistics*, 7(1):443-470.

Imbens GW, Rubin DB. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.

Institute of Medicine. (2001). *Crossing the Quality Chasm: A New Health System for the 21st Century*. National Academies Press.

Jha S, Topol EJ. (2016). Adapting to artificial intelligence: Radiologists and pathologists as information specialists. *JAMA*, 316(22):2353-2354.

Johansson F, Shalit U, Sontag D. (2016). Learning representations for counterfactual inference. *Proceedings of the 33rd International Conference on Machine Learning*, 48:3020-3029.

Kaelbling LP, Littman ML, Cassandra AR. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101(1-2):99-134.

Kaplan B. (2016). Evaluating informatics applications - clinical decision support systems literature review. *International Journal of Medical Informatics*, 64(1):15-37.

Kattan MW, O'Rourke C, Yu C, Chagin K. (2016). The wisdom of crowds of doctors: Their average predictions outperform their individual ones. *Medical Decision Making*, 36(4):536-540.

Katzman JL, Shaham U, Cloninger A, et al. (2018). DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18(1):24.

Kennedy EH, Ma Z, McHugh MD, Small DS. (2017). Non-parametric methods for doubly robust estimation of continuous treatment effects. *Journal of the Royal Statistical Society: Series B*, 79(4):1229-1245.

Kent DM, Rothwell PM, Ioannidis JP, et al. (2010). Assessing and reporting heterogeneity in treatment effects in clinical trials: A proposal. *Trials*, 11(1):85.

Kleinberg J, Mullainathan S, Raghavan M. (2016). Inherent trade-offs in the fair determination of risk scores. *Proceedings of Innovations in Theoretical Computer Science*, 43.

Kosorok MR, Moodie EEM. (2015). *Adaptive Treatment Strategies in Practice: Planning Trials and Analyzing Data for Personalized Medicine*. SIAM.

Krumholz HM. (2014). Big data and new knowledge in medicine: The thinking, training, and tools needed for a learning health system. *Health Affairs*, 33(7):1163-1170.

Künzel SR, Sekhon JS, Bickel PJ, Yu B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10):4156-4165.

Kuperman GJ, Bobb A, Payne TH, et al. (2007). Medication-related clinical decision support in computerized provider order entry systems: A review. *Journal of the American Medical Informatics Association*, 14(1):29-40.

Langford J, Zhang T. (2007). The epoch-greedy algorithm for contextual multi-armed bandits. *Advances in Neural Information Processing Systems*, 20:817-824.

Lattimore T, Szepesvári C. (2020). *Bandit Algorithms*. Cambridge University Press.

Liao P, Klasnja P, Murphy SA. (2020). Off-policy estimation of long-term average outcomes with applications to mobile health. *Journal of the American Statistical Association*, 116(533):382-391.

Liu NT, Holcomb JB, Wade CE, Salinas J. (2015). Development and validation of a machine learning algorithm and hybrid system to predict the need for life-saving interventions in trauma patients. *Medical and Biological Engineering and Computing*, 52(2):193-203.

London AJ. (2019). Artificial intelligence and black-box medical decisions: Accuracy versus explainability. *Hastings Center Report*, 49(1):15-21.

Lunceford JK, Davidian M. (2004). Stratification and weighting via the propensity score in estimation of causal treatment effects. *Statistics in Medicine*, 23(19):2937-2960.

McCulloch P, Altman DG, Campbell WB, et al. (2009). No surgical innovation without evaluation: The IDEAL recommendations. *Lancet*, 374(9695):1105-1112.

McNeil BJ, Pauker SG, Sox HC, Tversky A. (1982). On the elicitation of preferences for alternative therapies. *New England Journal of Medicine*, 306(21):1259-1262.

Mehrabi N, Morstatter F, Saxena N, et al. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6):1-35.

Mitchell M, Wu S, Zaldivar A, et al. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229.

Murphy SA. (2003). Optimal dynamic treatment regimes. *Journal of the Royal Statistical Society: Series B*, 65(2):331-355.

Murray CJ, Lauer JA, Hutubessy RC, et al. (2003). Effectiveness and costs of interventions to lower systolic blood pressure and cholesterol: A global and regional analysis. *Lancet*, 361(9359):717-725.

Naglie G, Krahn MD, Naimark D, et al. (1997). Primer on medical decision analysis: Part 3 - Estimating probabilities and utilities. *Medical Decision Making*, 17(2):136-141.

Obermeyer Z, Powers B, Vogeli C, Mullainathan S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464):447-453.

Osoba OA, Welser W. (2017). *An Intelligence in Our Image: The Risks of Bias and Errors in Artificial Intelligence*. RAND Corporation.

Pearl J. (2009). *Causality: Models, Reasoning and Inference* (2nd ed.). Cambridge University Press.

Pham T, Tran T, Phung D, Venkatesh S. (2017). Predicting healthcare trajectories from medical records: A deep learning approach. *Journal of Biomedical Informatics*, 69:218-229.

Rajkomar A, Hardt M, Howell MD, et al. (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12):866-872.

Robins JM, Hernan MA, Brumback B. (2000). Marginal structural models and causal inference in epidemiology. *Epidemiology*, 11(5):550-560.

Rosenbaum PR, Rubin DB. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1):41-55.

Rosenbaum PR. (2002). *Observational Studies* (2nd ed.). Springer.

Rubin DB. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5):688-701.

Sackett DL, Rosenberg WM, Gray JA, et al. (1996). Evidence based medicine: What it is and what it isn't. *BMJ*, 312(7023):71-72.

Schneeweiss S. (2007). Developments in post-marketing comparative effectiveness research. *Clinical Pharmacology and Therapeutics*, 82(2):143-156.

Schulam P, Saria S. (2017). Reliable decision support using counterfactual models. *Advances in Neural Information Processing Systems*, 30:1697-1708.

Sendak MP, Gao M, Brajer N, Balu S. (2020). Presenting machine learning model information to clinical end users with model facts labels. *npj Digital Medicine*, 3(1):41.

Shah ND, Steyerberg EW, Kent DM. (2018). Big data and predictive analytics: Recalibrating expectations. *JAMA*, 320(1):27-28.

Shalit U, Johansson FD, Sontag D. (2017). Estimating individual treatment effect: Generalization bounds and algorithms. *Proceedings of the 34th International Conference on Machine Learning*, 70:3076-3085.

Sheiner LB, Steimer JL. (2000). Pharmacokinetic/pharmacodynamic modeling in drug development. *Annual Review of Pharmacology and Toxicology*, 40:67-95.

Shortreed SM, Laber E, Lizotte DJ, et al. (2011). Informing sequential clinical decision-making through reinforcement learning: An empirical study. *Machine Learning*, 84(1-2):109-136.

Simon N, Friedman J, Hastie T, Tibshirani R. (2011). Regularization paths for Cox's proportional hazards model via coordinate descent. *Journal of Statistical Software*, 39(5):1-13.

Sondhi A, Arbour D, Dimmery D. (2020). Balanced off-policy evaluation in general action spaces. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics*, 2413-2421.

Sox HC, Higgins MC, Owens DK. (2013). *Medical Decision Making* (2nd ed.). Wiley-Blackwell.

Steyerberg EW, Vergouwe Y. (2014). Towards better clinical prediction models: Seven steps for development and an ABCD for validation. *European Heart Journal*, 35(29):1925-1931.

Stiglic G, Kocbek P, Fijacko N, et al. (2020). Interpretability of machine learning-based prediction models in healthcare. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 10(5):e1379.

Stuart EA, Lee BK, Leacy FP. (2013). Prognostic score-based balance measures can be a useful diagnostic for propensity score methods in comparative effectiveness research. *Journal of Clinical Epidemiology*, 66(8):S84-S90.

Sutton RS, Barto AG. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Swaminathan A, Joachims T. (2015). The self-normalized estimator for counterfactual learning. *Advances in Neural Information Processing Systems*, 28:3231-3239.

Thomas P, Brunskill E. (2016). Data-efficient off-policy policy evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning*, 48:2139-2148.

Topol EJ. (2019). High-performance medicine: The convergence of human and artificial intelligence. *Nature Medicine*, 25(1):44-56.

Tsoukalas A, Albertson T, Tagkopoulos I. (2015). From data to optimal decision: A data-driven, probabilistic machine learning approach to decision support for patients with sepsis. *JMIR Medical Informatics*, 3(1):e11.

Tversky A, Kahneman D. (1981). The framing of decisions and the psychology of choice. *Science*, 211(4481):453-458.

van der Laan MJ, Rose S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer.

Vayena E, Blasimme A, Cohen IG. (2018). Machine learning in medicine: Addressing ethical challenges. *PLoS Medicine*, 15(11):e1002689.

Verghese A, Shah NH, Harrington RA. (2018). What this computer needs is a physician: Humanism and artificial intelligence. *JAMA*, 319(1):19-20.

Wager S, Athey S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523):1228-1242.

Wang C, Zhu X, Qi Hong H, et al. (2015). Personalized risk prediction of symptomatic excessive daytime sleepiness in OSAHS. *Medicine*, 94(51):e2049.

Weinstein MC, Torrance G, McGuire A. (2009). QALYs: The basics. *Value in Health*, 12(s1):S5-S9.

Wiens J, Saria S, Sendak M, et al. (2019). Do no harm: A roadmap for responsible machine learning for health care. *Nature Medicine*, 25(9):1337-1340.

Williamson EJ, Morkoc IH. (2018). Propensity score: From naive enthusiasm to intuitive understanding. *Statistical Methods in Medical Research*, 27(8):2499-2525.

Xu Y, Ignatowicz A, Cheung CR, et al. (2020). Identifying subgroups of patients with type 2 diabetes who respond heterogeneously to SGLT2 inhibitors. *BMJ Open Diabetes Research and Care*, 8(1):e001410.

Yoon J, Jordon J, van der Schaar M. (2018). GANITE: Estimation of individualized treatment effects using generative adversarial nets. *Proceedings of the International Conference on Learning Representations*.

Zhang B, Tsiatis AA, Laber EB, Davidian M. (2012). A robust method for estimating optimal treatment regimes. *Biometrics*, 68(4):1010-1018.

Zhao Y, Zeng D, Rush AJ, Kosorok MR. (2012). Estimating individualized treatment rules using outcome weighted learning. *Journal of the American Statistical Association*, 107(499):1106-1118.

Zhou Z, Athey S, Wager S. (2018). Offline multi-action policy learning: Generalization and optimization. *arXiv preprint* arXiv:1810.04778.
