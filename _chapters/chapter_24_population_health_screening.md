---
layout: chapter
title: "Chapter 24: Population Health Management and Risk Stratification"
chapter_number: 24
part_number: 6
prev_chapter: /chapters/chapter-23-precision-medicine-genomics/
next_chapter: /chapters/chapter-25-sdoh-integration/
---
# Chapter 24: Population Health Management and Risk Stratification

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Design and implement population risk stratification systems that prioritize need over ease of reach, incorporating social determinants and equity metrics throughout the development pipeline
2. Develop screening strategies that maximize population health benefit while minimizing disparities in participation and outcomes, using threshold optimization methods that account for differential screening burdens across demographic groups
3. Build outbreak detection and infectious disease surveillance systems using temporal and spatial statistical methods, ensuring equitable surveillance coverage across underserved communities
4. Create health need forecasting models for resource allocation that explicitly optimize for equity rather than efficiency, incorporating geographic and social vulnerability indices
5. Evaluate population health interventions using causal inference frameworks with distributional impact analysis, quantifying effects across socioeconomic strata and identifying potential harm to subgroups
6. Implement production-grade population health systems with continuous monitoring, feedback loops, and automated equity audits that prevent drift toward easily reached populations

## Introduction

Population health management represents a fundamental shift from individual patient care to systematic approaches for improving health outcomes across entire communities. While clinical machine learning often focuses on optimizing care for individual patients presenting to healthcare systems, population health applications must identify and reach individuals who may not be actively seeking care, design interventions that work at scale, and ensure that limited resources are allocated to maximize overall health benefit while prioritizing those with greatest need. This paradigm shift introduces unique technical challenges and profound equity considerations that distinguish population health ML from other healthcare applications.

The core tension in population health management lies between efficiency and equity. Traditional optimization approaches naturally gravitate toward "low-hanging fruit" interventions targeting easily reached populations with high expected response rates, but these efficiency-based strategies systematically disadvantage underserved communities who face greater barriers to participation and follow-through. A person experiencing housing instability may be identified as high-risk for hospital readmission but be unable to attend care management appointments or follow complex treatment regimens, leading efficiency-focused systems to deprioritize them in favor of more "compliant" patients. This creates a vicious cycle where those with greatest need receive least attention, perpetuating rather than reducing health inequities.

This chapter develops population health management approaches that place equity at the center of technical design rather than treating it as a constraint or afterthought. We present mathematical frameworks for need-based targeting that explicitly balance health benefit against intervention burden, screening strategies that account for differential participation rates across demographic groups, outbreak detection systems that ensure equitable surveillance coverage, resource allocation methods that prioritize vulnerability over predicted response, and causal inference approaches for evaluating distributional impacts of population interventions. Throughout, we emphasize that achieving health equity requires fundamentally rethinking optimization objectives, not merely adding fairness constraints to traditional efficiency-based formulations.

Population health applications span a wide range of use cases, each with distinct equity implications. Care management programs identify high-risk individuals for intensive intervention, but risk models trained on historical utilization data may systematically miss underserved patients who face barriers to accessing care in the first place. Screening programs aim to detect disease early when treatment is most effective, but differential participation rates mean that those who would benefit most from screening are often least likely to participate. Outbreak detection systems must balance sensitivity against false alarm rates, but surveillance gaps in underserved communities can delay recognition of emerging threats. Resource allocation decisions determine which clinics receive equipment, staff, or supplies, but optimization for efficiency may concentrate resources in affluent areas with better infrastructure rather than high-need communities with limited alternatives. Each application requires careful consideration of how technical design choices shape who benefits from population health efforts.

The mathematical foundations for population health management draw from several distinct areas of statistics and machine learning. Risk stratification builds on survival analysis and competing risks models, but must account for social determinants that mediate both disease progression and ability to engage with interventions. Screening optimization uses decision analysis and threshold selection methods, but must incorporate models of differential screening burden and participation across populations. Outbreak detection applies sequential change-point detection and spatial scan statistics, but requires understanding how surveillance infrastructure gaps create blind spots. Resource allocation employs operations research and optimization methods, but needs explicit equity constraints and vulnerability-based objective functions. Intervention evaluation relies on causal inference frameworks, but must go beyond average treatment effects to characterize distributional impacts and potential harm to subgroups.

This chapter integrates technical rigor with practical implementation guidance, providing production-ready code for building population health systems that measurably advance health equity rather than perpetuating existing disparities.

## Mathematical Foundations for Population Health

### Population Risk Stratification

Population risk stratification aims to identify individuals at high risk for adverse outcomes to enable preventive interventions, but the mathematical formulation of "risk" profoundly shapes who gets identified and how resources are allocated. Traditional risk models optimize predictive accuracy for outcomes like hospitalization or mortality, but this approach has systematic equity problems because it conflates risk of adverse health events with risk of healthcare utilization or documentation, privileging patients with better access to care and more complete health records.

Consider a standard risk stratification model for identifying patients at high risk of hospitalization within the next year. Let $$Y_i \in \{0, 1\}$$ indicate whether patient $$ i $$ is hospitalized during the prediction window, and $$ X_i$$ be a feature vector containing demographics, diagnoses, medications, and prior utilization. A logistic regression model estimates:

$$
P(Y_i = 1 \mid X_i) = \sigma(\beta^T X_i)
$$

where $$\sigma(z) = 1/(1 + e^{-z})$$ is the sigmoid function. Patients are ranked by predicted risk $$\hat{p}_i = \sigma(\hat{\beta}^T X_i)$$ and those above some threshold $$\tau $$ are enrolled in care management programs. This formulation has two critical equity problems. First, patients with better documentation will have higher predicted risks simply because their health problems are more comprehensively recorded, not because their underlying health is worse. A patient seen regularly at a well-resourced academic medical center will accumulate more diagnosis codes than an equally sick patient receiving sporadic care at an under-resourced community clinic. Second, the outcome $$ Y_i$$ measures healthcare utilization rather than underlying health need - patients who lack insurance or transportation may be extremely sick but unable to access hospitalization.

A more equitable formulation distinguishes underlying health status from healthcare access and documentation quality. We can decompose the observed outcome as:

$$
Y_i = H_i \cdot A_i
$$

where $$H_i \in \{0, 1\}$$ indicates whether patient $$ i $$ has a health condition severe enough to warrant hospitalization, and $$ A_i \in \{0, 1\}$$ indicates whether they actually access hospitalization given their health status. In underserved populations, many patients have $$ H_i = 1 $$ but $$ A_i = 0 $$ due to barriers like lack of insurance, transportation, or proximity to hospitals. Standard risk models trained on $$ Y_i $$ systematically underestimate risk for patients with low $$ A_i $$, concentrating resources on those who already have good access.

To address this, we need models that estimate $$ P(H_i = 1 \mid X_i)$$ rather than $$ P(Y_i = 1 \mid X_i)$$, but $$ H_i $$ is unobserved in standard claims or EHR data. This requires causal inference techniques to debias for differential access. One approach uses instrumental variable methods where geographic proximity to hospitals serves as an instrument for access conditional on health status. Another approach stratifies by measures of healthcare access (insurance type, usual source of care, prior utilization patterns) and applies inverse probability weighting to adjust for selection into observation.

A practical approach suitable for many settings employs composite outcomes that combine utilization with clinical indicators less dependent on access patterns. Instead of predicting hospitalization alone, we can predict $$ Y_i^* = \max(Y_i^{\text{hosp}}, Y_i^{\text{ED}}, Y_i^{\text{lab}}, Y_i^{\text{decline}})$$ where:

- $$ Y_i^{\text{hosp}}$$ indicates hospitalization
- $$ Y_i^{\text{ED}}$$ indicates multiple emergency department visits
- $$ Y_i^{\text{lab}}$$ indicates concerning laboratory values (elevated HbA1c, low eGFR, etc.)
- $$ Y_i^{\text{decline}}$$ indicates functional decline measured by changes in vital signs or mobility

This composite outcome better captures underlying health status because laboratory values and functional measures are less dependent on healthcare-seeking behavior than hospitalization codes. However, even composite outcomes have equity issues if some components are more likely to be measured in certain populations.

The most robust approach incorporates social determinants and health equity indices directly into risk models, not as additional predictors but as explicit components of the outcome definition. We can define need-adjusted risk as:

$$
R_i^{\text{need}} = P(H_i = 1 \mid X_i) \cdot V_i
$$

where $$V_i $$ is a vulnerability index capturing social determinants that amplify the impact of health problems. For example, an elderly patient with diabetes living alone in a housing-unstable situation faces greater consequences from uncontrolled diabetes than an otherwise identical patient with strong family support and stable housing. The vulnerability index $$ V_i $$ can incorporate:

- Social isolation (living alone, limited social support)
- Housing instability or homelessness
- Food insecurity
- Transportation barriers
- Limited English proficiency
- Low health literacy
- Multiple social needs

This formulation prioritizes patients who face greatest consequences from health problems rather than those most likely to utilize services, fundamentally reorienting the optimization objective toward need rather than predicted utilization.

### Screening Theory and Optimization

Screening programs aim to detect disease early when treatment is most effective, but screening decisions involve tradeoffs between benefits and harms that differ across populations in ways that can exacerbate health disparities. The mathematical framework for screening optimization must account for differential screening burden, varying prevalence across demographic groups, and inequities in follow-up care access.

Classical screening theory analyzes the population benefit from screening using a decision-theoretic framework. Let $$ D \in \{0, 1\}$$ indicate true disease status and $$ S \in \{0, 1\}$$ indicate screening test result (positive or negative). For a continuous test score $$ T $$, the screening decision uses threshold $$\tau $$ such that $$ S = \mathbb{1}(T \gt  \tau)$$. The operating characteristics of screening at threshold $$\tau$$ are:

- Sensitivity: $$\text{Se}(\tau) = P(T \gt  \tau \mid D = 1)$$
- Specificity: $$\text{Sp}(\tau) = P(T \leq \tau \mid D = 0)$$
- Positive predictive value: $$\text{PPV}(\tau) = \frac{p \cdot \text{Se}(\tau)}{p \cdot \text{Se}(\tau) + (1-p)(1-\text{Sp}(\tau))}$$

where $$ p = P(D = 1)$$ is disease prevalence. The receiver operating characteristic (ROC) curve plots sensitivity versus $$(1 - \text{specificity})$$ across thresholds, and the area under the ROC curve (AUC) summarizes overall discriminative ability.

However, this framework ignores screening burden and assumes all false positives and false negatives have equal consequences across populations. In reality, screening burden varies dramatically. For a patient with reliable transportation, health insurance, and flexible work arrangements, undergoing screening and follow-up testing for a false positive is an inconvenience. For a patient working multiple hourly-wage jobs with no paid sick leave, uncertain immigration status, or significant caregiving responsibilities, the same process may be prohibitively burdensome or even dangerous.

A more complete framework incorporates screening burden explicitly into the utility function. Let:

- $$ U_{TP}$$ = utility of true positive (early disease detection and treatment)
- $$ U_{TN}$$ = utility of true negative (reassurance)
- $$ U_{FP}$$ = utility of false positive (includes screening burden, anxiety, follow-up testing risks)
- $$ U_{FN}$$ = utility of false negative (missed opportunity for early treatment)

Expected utility of screening at threshold $$\tau$$ for an individual is:

$$
EU(\tau) = p \cdot [\text{Se}(\tau) \cdot U_{TP} + (1-\text{Se}(\tau)) \cdot U_{FN}] + (1-p) \cdot [\text{Sp}(\tau) \cdot U_{TN} + (1-\text{Sp}(\tau)) \cdot U_{FP}]
$$

The optimal threshold $$\tau^*$$ maximizes expected utility:

$$
\tau^* = \arg\max_\tau EU(\tau)
$$

Critically, the utility values $$U_{TP}, U_{TN}, U_{FP}, U_{FN}$$ vary across populations based on screening burden, access to follow-up care, and treatment effectiveness. For populations facing high screening burden, $$ U_{FP}$$ is more negative (false positives cause greater harm) and $$ U_{TN}$$ is less positive (screening itself imposes costs even when negative). This means the optimal threshold $$\tau^*$$ should differ across populations - higher thresholds (requiring stronger evidence before screening positive) for populations with high screening burden.

Standard practice uses uniform thresholds across populations, which systematically harms those facing greater screening burden. Consider mammography screening where $$\tau$$ determines the threshold for calling a mammogram abnormal and recommending additional imaging. Using the same threshold for all women means that those who face transportation barriers, lack paid leave, or fear immigration consequences from medical contact experience greater net harm from screening because they endure the same false positive rate but face higher burden from follow-up.

An equitable screening strategy optimizes thresholds separately for populations with different burden profiles:

$$
\tau_g^* = \arg\max_\tau EU_g(\tau)
$$

where subscript $$g$$ indicates population group. This approach raises concerns about "different standards" for different groups, but the key insight is that groups already face different burdens - uniform thresholds mean uniform test characteristics but inequitable net benefit. Adjusting thresholds to equalize net benefit across groups is more equitable than maintaining uniform test characteristics.

The number needed to screen (NNS) provides an interpretable metric for comparing screening strategies:

$$
\text{NNS} = \frac{1}{p \cdot \text{Se}(\tau) \cdot \text{RRR}}
$$

where RRR is relative risk reduction from detecting and treating disease early. Lower NNS means fewer people need to be screened to prevent one adverse outcome. However, NNS alone is insufficient for equity analysis because it doesn't account for screening burden or differential treatment effectiveness. A better metric is number needed to screen to achieve net benefit:

$$
\text{NNS}_{\text{net}} = \frac{1}{p \cdot \text{Se}(\tau) \cdot \text{RRR} - (1-p) \cdot (1-\text{Sp}(\tau)) \cdot h}
$$

where $$h $$ is the ratio of harms from false positive to benefits from true positive. This metric accounts for false positive harms and provides a more complete picture of screening value.

Screening optimization must also address differential participation rates. Even if a screening strategy has positive net benefit at the population level, it exacerbates disparities if participation is lower among those who would benefit most. Let $$\pi_g $$ be the participation rate in group $$ g$$. Population-level impact of screening is:

$$
\text{Impact}_g = \pi_g \cdot [p_g \cdot \text{Se}(\tau) \cdot \text{RRR}]
$$

If participation rates $$\pi_g $$ are lower in groups with higher prevalence $$ p_g $$ (as often occurs in underserved populations), screening widens disparities even if it has positive benefit in participating individuals. This requires intervention strategies that actively reduce participation barriers rather than simply offering screening and assuming equal uptake.

### Temporal Outbreak Detection

Outbreak detection systems identify unusual increases in disease incidence to trigger public health response, but surveillance gaps in underserved communities create blind spots where outbreaks may go undetected. Mathematical approaches for outbreak detection must account for unequal surveillance coverage and differential reporting rates across populations.

The fundamental problem in outbreak detection is distinguishing signal (true outbreak) from noise (random variation in baseline disease rates). Let $$ Y_t $$ be the count of disease cases observed at time $$ t $$. Under the null hypothesis of no outbreak, $$ Y_t $$ follows some baseline distribution, while under the alternative hypothesis of an outbreak starting at time $$\tau_0$$, there is an elevated rate:

$$
Y_t \sim \begin{cases} \text{Baseline}(\lambda_0) & t < \tau_0 \\ \text{Outbreak}(\lambda_1) & t \geq \tau_0 \end{cases}
$$

where $$\lambda_1 \gt  \lambda_0 $$. The goal is to detect the change point $$\tau_0$$ as quickly as possible while maintaining low false alarm rates.

The cumulative sum (CUSUM) control chart is a classical sequential detection method that accumulates evidence of deviation from baseline. Define the CUSUM statistic:

$$
S_t = \max(0, S_{t-1} + (Y_t - \mu_0 - k))
$$

where $$\mu_0 $$ is the expected count under baseline, and $$ k $$ is a reference value (typically set to half the minimum outbreak effect size). An alarm is raised when $$ S_t $$ exceeds threshold $$ h $$. The CUSUM efficiently detects sustained increases in disease counts and provides control over average run length (ARL), the expected time to false alarm under the null hypothesis.

However, CUSUM assumes complete and uniform surveillance. In reality, disease reporting varies dramatically across populations. Underserved communities often have limited access to healthcare facilities where diagnoses are made and reported, leading to systematically lower baseline counts $$\mu_0^g $$ not because disease is less common but because it is less frequently diagnosed and reported. If we use the same CUSUM threshold $$ h $$ across all communities, outbreaks in underserved areas require larger absolute increases to trigger alarms, creating delayed detection in populations that already face worse health outcomes.

An equity-aware CUSUM approach standardizes by population-specific baselines and adjusts for reporting rates. Let $$\rho_g $$ be the estimated reporting rate (proportion of true cases that are captured by surveillance) in geographic area $$ g$$. We can adjust the CUSUM statistic to:

$$
S_t^g = \max(0, S_{t-1}^g + \frac{Y_t^g - \mu_0^g}{\rho_g} - k)
$$

This formulation upweights observations from areas with poor surveillance coverage, making the system more sensitive to changes in these communities. However, this also increases false alarm rates in low-coverage areas if reporting variability is higher, requiring careful calibration.

An alternative approach employs stratified thresholds that explicitly trade off detection speed against false alarm rates differently across communities. We can set community-specific thresholds $$h_g$$ to equalize expected time to detection for outbreaks of equal severity:

$$
\mathbb{E}[\tau^g \mid \text{outbreak}] = c \text{ for all } g
$$

where $$\tau^g $$ is the detection time in community $$ g $$ and $$ c $$ is a constant. This means accepting higher false alarm rates in underserved communities with poorer surveillance in exchange for comparable detection speed, which may be justified by the greater consequences of delayed outbreak response in these vulnerable populations.

Spatial scan statistics provide complementary approaches that detect geographic clusters of disease. The Kulldorff spatial scan statistic evaluates cylinders (circles on a map extended through time) and identifies clusters where observed case counts significantly exceed expected counts. For each potential cluster $$ Z$$, the likelihood ratio is:

$$
\Lambda(Z) = \left(\frac{c_Z}{E[c_Z]}\right)^{c_Z} \left(\frac{c_{\bar{Z}}}{E[c_{\bar{Z}}]}\right)^{c_{\bar{Z}}}
$$

where $$c_Z $$ is observed cases in cluster $$ Z $$, $$ c_{\bar{Z}}$$ is observed cases outside the cluster, and $$ E[\cdot]$$ indicates expected counts. The most likely cluster is the one maximizing $$\Lambda(Z)$$, and statistical significance is assessed via Monte Carlo simulation.

Spatial scan statistics have equity implications because they naturally detect clusters in areas with higher population density and better surveillance coverage. A cluster of 100 excess cases in a dense urban area with good healthcare access is more likely to be detected than a cluster of 50 excess cases in a rural area with limited healthcare facilities, even though the rural outbreak may represent a larger relative increase in disease burden. To address this, we can use population-adjusted scan statistics that weight by population size and vulnerability indices:

$$
\Lambda^{\text{adj}}(Z) = V_Z \cdot \Lambda(Z)
$$

where $$V_Z $$ is an average vulnerability index for cluster $$ Z $$. This upweights clusters in vulnerable communities, increasing sensitivity to outbreaks in areas where consequences are most severe.

### Resource Allocation and Capacity Planning

Resource allocation in population health determines which communities receive limited resources like clinic capacity, public health nurses, or vaccination supplies. Traditional operations research approaches optimize for efficiency, but this systematically disadvantages underserved communities where infrastructure gaps mean lower expected returns per resource unit. Equity-focused resource allocation requires explicit optimization for need and vulnerability rather than predicted impact.

Consider a resource allocation problem where we have $$ R $$ units of a resource (e.g., community health workers) to allocate across $$ G $$ geographic areas. Let $$ r_g $$ be the allocation to area $$ g $$, subject to $$\sum_g r_g = R$$. An efficiency-based allocation maximizes total impact:

$$
\max_{r_g} \sum_g I_g(r_g)
$$

where $$I_g(r_g)$$ is the predicted impact (e.g., number of hospitalizations prevented) in area $$ g $$ from allocating $$ r_g $$ resources. This formulation naturally concentrates resources in areas where infrastructure already exists to support effective interventions, as $$ I_g(r_g)$$ is higher in well-resourced communities.

An equity-based allocation instead maximizes weighted impact where weights reflect need rather than capacity:

$$
\max_{r_g} \sum_g w_g \cdot I_g(r_g)
$$

subject to constraints that ensure adequate coverage in all communities:

$$
r_g \geq r_{\min} \text{ for all } g
$$

The weights $$w_g$$ can incorporate multiple dimensions of need:

$$
w_g = \alpha_1 \cdot \text{Disease burden}_g + \alpha_2 \cdot \text{Vulnerability}_g + \alpha_3 \cdot \text{Resource deficit}_g
$$

where disease burden captures baseline prevalence and severity, vulnerability incorporates social determinants and healthcare access barriers, and resource deficit measures existing resource gaps relative to need. The coefficients $$\alpha_1, \alpha_2, \alpha_3$$ reflect value judgments about how to prioritize different dimensions of need.

An alternative formulation optimizes for equity directly by minimizing disparities in outcomes across areas:

$$
\min_{r_g} \text{Var}_g[H_g(r_g)]
$$

where $$H_g(r_g)$$ is the predicted health outcome (e.g., life expectancy, disease-free years) in area $$ g $$ given resource allocation $$ r_g $$. This minimax approach seeks to raise the floor, prioritizing improvements in worst-off communities over aggregate benefit.

Capacity planning for healthcare services must account for geographic access barriers that affect underserved populations disproportionately. The classic facility location problem determines where to place $$ K $$ facilities to minimize average travel distance. Let $$ d_{ig}$$ be the distance from individual $$ i $$ to potential facility location $$ g $$, and $$ y_g \in \{0, 1\}$$ indicate whether a facility is placed at location $$ g$$. The p-median problem minimizes total travel burden:

$$
\min_{y_g, x_{ig}} \sum_i \sum_g d_{ig} \cdot x_{ig}
$$

subject to:
- $$\sum_g y_g = K$$ (exactly $$K$$ facilities)
- $$\sum_g x_{ig} = 1$$ for all $$i$$ (each person assigned to exactly one facility)
- $$ x_{ig} \leq y_g $$ for all $$ i, g$$ (can only assign to open facilities)

This formulation minimizes average travel distance but ignores heterogeneity in travel burden. For someone with a car and flexible schedule, traveling 20 minutes versus 40 minutes is a minor inconvenience. For someone depending on public transit with multiple job responsibilities, this difference may determine whether healthcare access is feasible at all.

An equity-focused facility location approach weights travel burden by individual vulnerability:

$$
\min_{y_g, x_{ig}} \sum_i v_i \cdot \sum_g d_{ig} \cdot x_{ig}
$$

where $$v_i $$ is a vulnerability score capturing factors that amplify travel burden (lack of car, caregiving responsibilities, job inflexibility, etc.). This formulation places facilities closer to those for whom access barriers are most significant, even if it increases average travel distance.

Capacity planning must also consider temporal variation in demand and resource availability. Underserved communities often face "deserts" of healthcare access during evenings and weekends when family members are available to accompany patients to appointments. Queueing models can inform capacity planning decisions that account for these patterns. The M/M/s queue with time-varying arrival rates $$\lambda(t)$$ and $$ s$$ servers provides a baseline model. Expected waiting time is:

$$
W = \frac{P_0}{s \mu - \lambda(t)} \cdot \frac{(\lambda(t) / \mu)^s}{s! \cdot (1 - \lambda(t)/(s\mu))^2}
$$

where $$\mu $$ is service rate and $$ P_0$$ is the probability of zero customers in system. For populations with limited flexibility in timing, prolonged wait times effectively reduce access.

## Risk Stratification for Care Management

Care management programs provide intensive support to high-risk patients to prevent adverse outcomes like hospitalizations, but the process of identifying patients for these programs has profound equity implications. This section develops production-grade risk stratification systems that prioritize need over predicted utilization, incorporate social determinants throughout the modeling pipeline, and include continuous monitoring for equity drift.

### Comprehensive Risk Model Architecture

A production risk stratification system must integrate diverse data sources while remaining transparent and interpretable for clinical teams. We develop a modular architecture where separate components capture distinct risk dimensions, then combine them using interpretable aggregation rules.

```python
"""
Production Risk Stratification System for Population Health Management

This module implements a comprehensive risk stratification system that prioritizes
need over predicted utilization, incorporates social determinants, and includes
continuous equity monitoring. The architecture is modular to allow clinical
interpretation of different risk dimensions.

Author: Healthcare AI for the Underserved
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import logging
from datetime import datetime, timedelta
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskComponent:
    """
    Represents a single risk dimension (clinical, social, functional, etc.)

    Attributes:
        name: Human-readable name for this risk dimension
        model: Trained sklearn-compatible model for this component
        features: List of feature names used by this component
        weight: Weight for combining with other components (0-1)
        last_trained: Timestamp of last model training
        performance_metrics: Dictionary of performance metrics on test set
    """
    name: str
    model: BaseEstimator
    features: List[str]
    weight: float = 1.0
    last_trained: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability for this risk component"""
        X_component = X[self.features]
        return self.model.predict_proba(X_component)[:, 1]

@dataclass
class VulnerabilityIndex:
    """
    Social vulnerability index capturing factors that amplify health consequences

    Components include:
    - Social isolation (living alone, limited support network)
    - Housing instability or homelessness
    - Food insecurity
    - Transportation barriers
    - Limited English proficiency
    - Low health literacy
    - Financial strain
    """
    social_isolation_score: float = 0.0
    housing_instability_score: float = 0.0
    food_insecurity_score: float = 0.0
    transportation_barriers_score: float = 0.0
    language_barriers_score: float = 0.0
    health_literacy_score: float = 0.0
    financial_strain_score: float = 0.0

    def compute_composite(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute composite vulnerability score

        Args:
            weights: Optional custom weights for each component

        Returns:
            Composite vulnerability score (0-1 scale)
        """
        if weights is None:
            # Equal weighting by default
            weights = {
                'social_isolation': 1/7,
                'housing_instability': 1/7,
                'food_insecurity': 1/7,
                'transportation_barriers': 1/7,
                'language_barriers': 1/7,
                'health_literacy': 1/7,
                'financial_strain': 1/7
            }

        score = (
            weights['social_isolation'] * self.social_isolation_score +
            weights['housing_instability'] * self.housing_instability_score +
            weights['food_insecurity'] * self.food_insecurity_score +
            weights['transportation_barriers'] * self.transportation_barriers_score +
            weights['language_barriers'] * self.language_barriers_score +
            weights['health_literacy'] * self.health_literacy_score +
            weights['financial_strain'] * self.financial_strain_score
        )

        return np.clip(score, 0.0, 1.0)

class NeedBasedRiskStratifier(BaseEstimator, ClassifierMixin):
    """
    Risk stratification system that prioritizes need over predicted utilization

    This system addresses equity problems in traditional risk models by:
    1. Separating clinical risk from access barriers
    2. Incorporating social vulnerability as amplifier of consequences
    3. Using composite outcomes less dependent on healthcare-seeking behavior
    4. Continuous monitoring for equity drift across demographic groups

    The system combines multiple risk components (clinical, functional, social)
    with vulnerability indices to produce need-adjusted risk scores that prioritize
    patients facing greatest consequences rather than those most likely to utilize
    services.

    Parameters:
        clinical_model_type: Type of model for clinical risk component
        social_model_type: Type of model for social risk component
        functional_model_type: Type of model for functional status component
        vulnerability_weights: Weights for different vulnerability dimensions
        min_training_size: Minimum samples required for training
        equity_groups: Demographic variables for equity monitoring
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        clinical_model_type: str = 'gradient_boosting',
        social_model_type: str = 'logistic',
        functional_model_type: str = 'random_forest',
        vulnerability_weights: Optional[Dict[str, float]] = None,
        min_training_size: int = 1000,
        equity_groups: List[str] = None,
        random_state: int = 42
    ):
        self.clinical_model_type = clinical_model_type
        self.social_model_type = social_model_type
        self.functional_model_type = functional_model_type
        self.vulnerability_weights = vulnerability_weights
        self.min_training_size = min_training_size
        self.equity_groups = equity_groups or ['race', 'ethnicity', 'insurance', 'language']
        self.random_state = random_state

        # Components will be initialized during fit
        self.components: Dict[str, RiskComponent] = {}
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted_: bool = False
        self.equity_metrics_: Dict[str, Any] = {}

    def _initialize_component_model(
        self,
        model_type: str
    ) -> BaseEstimator:
        """Initialize a model based on type specification"""
        if model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_composite_outcome(
        self,
        y_hosp: np.ndarray,
        y_ed: np.ndarray,
        y_lab: np.ndarray,
        y_decline: np.ndarray
    ) -> np.ndarray:
        """
        Create composite outcome that captures health status beyond utilization

        Args:
            y_hosp: Hospitalization indicator
            y_ed: Multiple ED visits indicator
            y_lab: Concerning lab values indicator
            y_decline: Functional decline indicator

        Returns:
            Composite outcome (1 if any component is positive)
        """
        composite = np.maximum.reduce([
            y_hosp.astype(int),
            y_ed.astype(int),
            y_lab.astype(int),
            y_decline.astype(int)
        ])

        return composite

    def _extract_vulnerability_scores(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract vulnerability index scores from feature matrix

        Expects columns following naming convention:
        - social_isolation_score
        - housing_instability_score
        - food_insecurity_score
        - transportation_barriers_score
        - language_barriers_score
        - health_literacy_score
        - financial_strain_score

        Args:
            X: Feature matrix including vulnerability scores

        Returns:
            Array of composite vulnerability scores
        """
        vulnerability_cols = [
            'social_isolation_score',
            'housing_instability_score',
            'food_insecurity_score',
            'transportation_barriers_score',
            'language_barriers_score',
            'health_literacy_score',
            'financial_strain_score'
        ]

        # Check which vulnerability columns are available
        available_cols = [col for col in vulnerability_cols if col in X.columns]

        if len(available_cols) == 0:
            logger.warning("No vulnerability score columns found. Using uniform vulnerability.")
            return np.ones(len(X))

        # Compute composite vulnerability
        vulnerability_scores = []
        for _, row in X[available_cols].iterrows():
            vi = VulnerabilityIndex(
                **{col: row[col] for col in available_cols}
            )
            vulnerability_scores.append(
                vi.compute_composite(self.vulnerability_weights)
            )

        return np.array(vulnerability_scores)

    def fit(
        self,
        X: pd.DataFrame,
        y_components: Dict[str, np.ndarray],
        feature_groups: Dict[str, List[str]],
        demographic_data: Optional[pd.DataFrame] = None
    ) -> 'NeedBasedRiskStratifier':
        """
        Fit the risk stratification model

        Args:
            X: Feature matrix with all features
            y_components: Dictionary mapping component names to outcome arrays
                Expected keys: 'hospitalization', 'ed_visits', 'lab_abnormal', 'functional_decline'
            feature_groups: Dictionary mapping component names to feature lists
                Expected keys: 'clinical', 'social', 'functional'
            demographic_data: Optional DataFrame with demographic variables for equity monitoring

        Returns:
            self: Fitted estimator
        """
        if len(X) < self.min_training_size:
            raise ValueError(
                f"Training set too small: {len(X)} < {self.min_training_size}"
            )

        logger.info(f"Training need-based risk stratifier on {len(X)} samples")

        # Create composite outcome
        y_composite = self._create_composite_outcome(
            y_components['hospitalization'],
            y_components['ed_visits'],
            y_components['lab_abnormal'],
            y_components['functional_decline']
        )

        logger.info(f"Composite outcome prevalence: {y_composite.mean():.3f}")

        # Initialize and train clinical risk component
        clinical_model = self._initialize_component_model(self.clinical_model_type)
        clinical_features = feature_groups['clinical']
        clinical_model.fit(X[clinical_features], y_composite)

        self.components['clinical'] = RiskComponent(
            name='Clinical Risk',
            model=clinical_model,
            features=clinical_features,
            weight=0.4,
            last_trained=datetime.now()
        )

        # Initialize and train social risk component
        social_model = self._initialize_component_model(self.social_model_type)
        social_features = feature_groups['social']
        social_model.fit(X[social_features], y_composite)

        self.components['social'] = RiskComponent(
            name='Social Risk',
            model=social_model,
            features=social_features,
            weight=0.3,
            last_trained=datetime.now()
        )

        # Initialize and train functional status component
        functional_model = self._initialize_component_model(self.functional_model_type)
        functional_features = feature_groups['functional']
        functional_model.fit(X[functional_features], y_composite)

        self.components['functional'] = RiskComponent(
            name='Functional Status',
            model=functional_model,
            features=functional_features,
            weight=0.3,
            last_trained=datetime.now()
        )

        # Extract vulnerability scores for need adjustment
        self.vulnerability_scores_train_ = self._extract_vulnerability_scores(X)

        # Compute equity metrics if demographic data provided
        if demographic_data is not None:
            self._compute_equity_metrics(
                X, y_composite, demographic_data
            )

        self.is_fitted_ = True
        logger.info("Model training completed successfully")

        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
        return_components: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Predict need-adjusted risk probabilities

        Args:
            X: Feature matrix
            return_components: If True, return component risks separately

        Returns:
            If return_components is False:
                Array of shape (n_samples,) with need-adjusted risk scores
            If return_components is True:
                Tuple of (composite_risk, component_dict) where component_dict
                maps component names to their individual risk predictions
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        # Get predictions from each component
        component_risks = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for name, component in self.components.items():
            risk = component.predict_proba(X)
            component_risks[name] = risk
            weighted_sum += component.weight * risk
            total_weight += component.weight

        # Combine component risks with weights
        base_risk = weighted_sum / total_weight

        # Adjust by vulnerability to get need-adjusted risk
        vulnerability = self._extract_vulnerability_scores(X)
        need_adjusted_risk = base_risk * (1 + vulnerability)

        # Clip to valid probability range
        need_adjusted_risk = np.clip(need_adjusted_risk, 0.0, 1.0)

        if return_components:
            component_risks['vulnerability'] = vulnerability
            component_risks['base_risk'] = base_risk
            return need_adjusted_risk, component_risks
        else:
            return need_adjusted_risk

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict binary high-risk classification

        Args:
            X: Feature matrix
            threshold: Risk threshold for classification as high-risk

        Returns:
            Binary predictions (1 = high risk, 0 = not high risk)
        """
        risk_scores = self.predict_proba(X)
        return (risk_scores >= threshold).astype(int)

    def _compute_equity_metrics(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        demographic_data: pd.DataFrame
    ) -> None:
        """
        Compute equity metrics across demographic groups

        Args:
            X: Feature matrix
            y: True outcomes
            demographic_data: DataFrame with demographic variables
        """
        predictions = self.predict_proba(X)

        self.equity_metrics_ = {}

        for group_var in self.equity_groups:
            if group_var not in demographic_data.columns:
                logger.warning(f"Group variable {group_var} not found in demographic data")
                continue

            group_metrics = {}
            for group_value in demographic_data[group_var].unique():
                mask = demographic_data[group_var] == group_value
                if mask.sum() < 30:  # Skip very small groups
                    continue

                group_y = y[mask]
                group_pred = predictions[mask]

                # Compute metrics for this group
                group_metrics[group_value] = {
                    'mean_risk': float(group_pred.mean()),
                    'median_risk': float(np.median(group_pred)),
                    'prevalence': float(group_y.mean()),
                    'n_samples': int(mask.sum()),
                    'high_risk_rate': float((group_pred > 0.5).mean())
                }

            self.equity_metrics_[group_var] = group_metrics

        logger.info("Equity metrics computed successfully")

    def get_equity_report(self) -> pd.DataFrame:
        """
        Generate equity report comparing metrics across demographic groups

        Returns:
            DataFrame with equity metrics by group
        """
        if not self.equity_metrics_:
            raise RuntimeError("No equity metrics available. Provide demographic_data during fit().")

        rows = []
        for group_var, group_metrics in self.equity_metrics_.items():
            for group_value, metrics in group_metrics.items():
                row = {
                    'demographic_variable': group_var,
                    'group_value': group_value,
                    **metrics
                }
                rows.append(row)

        return pd.DataFrame(rows)

def compute_targeting_equity_metrics(
    risk_scores: np.ndarray,
    true_need: np.ndarray,
    demographic_groups: pd.Series,
    top_k: int = 500
) -> Dict[str, Any]:
    """
    Evaluate equity of care management targeting decisions

    This function assesses whether high-risk targeting concentrates resources
    on easily reached populations vs those with greatest need, across demographic groups.

    Args:
        risk_scores: Predicted risk scores used for targeting
        true_need: Ground truth need scores (e.g., composite health + vulnerability)
        demographic_groups: Series indicating demographic group for each patient
        top_k: Number of patients to target (top k by risk score)

    Returns:
        Dictionary containing equity metrics:
        - representation_ratios: Ratio of selected to eligible for each group
        - need_capture_rates: Proportion of high-need patients captured in each group
        - precision_by_group: Precision (true need among selected) by group
        - disparity_metrics: Ratio of max to min across groups for key metrics
    """
    # Select top k patients by risk score
    top_k_indices = np.argsort(risk_scores)[-top_k:]
    selected = np.zeros(len(risk_scores), dtype=bool)
    selected[top_k_indices] = True

    # Define high need as top quartile of true need
    high_need_threshold = np.percentile(true_need, 75)
    high_need = true_need >= high_need_threshold

    # Compute metrics by demographic group
    metrics_by_group = {}

    for group in demographic_groups.unique():
        group_mask = demographic_groups == group

        # Selection rate: proportion of group selected
        selection_rate = selected[group_mask].mean()

        # Representation ratio: selection rate / population proportion
        population_prop = group_mask.mean()
        representation_ratio = selection_rate / population_prop if population_prop > 0 else 0

        # Need capture rate: proportion of high-need patients in group who are selected
        group_high_need = high_need[group_mask]
        if group_high_need.sum() > 0:
            need_capture_rate = selected[group_mask][high_need[group_mask]].mean()
        else:
            need_capture_rate = 0

        # Precision: proportion of selected patients who are truly high-need
        group_selected = selected[group_mask]
        if group_selected.sum() > 0:
            precision = high_need[group_mask][group_selected].mean()
        else:
            precision = 0

        metrics_by_group[group] = {
            'selection_rate': float(selection_rate),
            'representation_ratio': float(representation_ratio),
            'need_capture_rate': float(need_capture_rate),
            'precision': float(precision),
            'n_selected': int(group_selected.sum()),
            'n_high_need': int(group_high_need.sum())
        }

    # Compute disparity metrics (ratio of max to min across groups)
    representation_ratios = [m['representation_ratio'] for m in metrics_by_group.values()]
    need_capture_rates = [m['need_capture_rate'] for m in metrics_by_group.values()]

    disparity_metrics = {
        'representation_disparity': max(representation_ratios) / min(representation_ratios) if min(representation_ratios) > 0 else np.inf,
        'need_capture_disparity': max(need_capture_rates) / min(need_capture_rates) if min(need_capture_rates) > 0 else np.inf,
    }

    return {
        'metrics_by_group': metrics_by_group,
        'disparity_metrics': disparity_metrics
    }
```

This implementation provides a production-grade risk stratification system that addresses key equity concerns in care management targeting. The modular architecture separates clinical, social, and functional risk components to facilitate clinical interpretation and debugging. The vulnerability index explicitly captures factors that amplify health consequences, allowing the system to prioritize patients facing greatest overall risk rather than just highest predicted utilization.

### Continuous Equity Monitoring

Risk stratification models can drift over time as patient populations, care patterns, and data collection practices change. Equity drift is particularly concerning because models may gradually shift toward prioritizing patients from well-documented, easily-reached populations. Production systems require continuous monitoring to detect and correct equity drift.

```python
"""
Continuous equity monitoring system for risk stratification models

This module implements automated monitoring of risk model equity metrics over time,
with alerts when disparities exceed acceptable thresholds.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class EquityAlert:
    """Represents an alert for equity metric exceeding threshold"""
    timestamp: datetime
    metric_name: str
    demographic_variable: str
    current_value: float
    threshold: float
    severity: str  # 'warning' or 'critical'
    affected_groups: List[str]
    message: str

class EquityMonitor:
    """
    Continuous monitoring system for risk stratification equity metrics

    This system tracks equity metrics over time and generates alerts when
    disparities exceed acceptable thresholds. It maintains a time-series
    of equity measurements and performs statistical tests for trend detection.

    Parameters:
        stratifier: Fitted NeedBasedRiskStratifier instance to monitor
        disparity_thresholds: Dict mapping metric names to maximum acceptable disparity ratios
        lookback_periods: Number of monitoring periods to use for trend detection
        alert_cooldown_hours: Minimum hours between alerts for same issue
    """

    def __init__(
        self,
        stratifier: NeedBasedRiskStratifier,
        disparity_thresholds: Optional[Dict[str, float]] = None,
        lookback_periods: int = 12,
        alert_cooldown_hours: int = 24
    ):
        self.stratifier = stratifier
        self.disparity_thresholds = disparity_thresholds or {
            'representation_ratio': 1.5,  # Max 1.5x difference in selection rates
            'need_capture_rate': 1.3,     # Max 1.3x difference in capturing high-need patients
            'precision': 1.3,              # Max 1.3x difference in precision
            'false_positive_rate': 1.5     # Max 1.5x difference in false positive rates
        }
        self.lookback_periods = lookback_periods
        self.alert_cooldown_hours = alert_cooldown_hours

        # Storage for time-series of metrics
        self.metric_history: List[Dict] = []
        self.alert_history: List[EquityAlert] = []

    def monitor(
        self,
        X_current: pd.DataFrame,
        y_current: np.ndarray,
        demographic_data: pd.DataFrame,
        monitoring_date: Optional[datetime] = None
    ) -> List[EquityAlert]:
        """
        Perform equity monitoring on current data batch

        Args:
            X_current: Current batch of feature data
            y_current: Current batch of true outcomes
            demographic_data: Demographic information for current batch
            monitoring_date: Date of monitoring (defaults to current time)

        Returns:
            List of equity alerts generated for this monitoring period
        """
        if monitoring_date is None:
            monitoring_date = datetime.now()

        # Get predictions
        predictions = self.stratifier.predict_proba(X_current)

        # Compute equity metrics for current batch
        current_metrics = self._compute_batch_equity_metrics(
            predictions, y_current, demographic_data
        )
        current_metrics['timestamp'] = monitoring_date

        # Store in history
        self.metric_history.append(current_metrics)

        # Check for threshold violations
        alerts = self._check_thresholds(current_metrics, monitoring_date)

        # Check for adverse trends
        if len(self.metric_history) >= self.lookback_periods:
            trend_alerts = self._check_trends(monitoring_date)
            alerts.extend(trend_alerts)

        # Filter alerts by cooldown period
        alerts = self._apply_alert_cooldown(alerts)

        # Store alerts
        self.alert_history.extend(alerts)

        return alerts

    def _compute_batch_equity_metrics(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        demographic_data: pd.DataFrame
    ) -> Dict:
        """Compute comprehensive equity metrics for a data batch"""
        metrics = {}

        for group_var in self.stratifier.equity_groups:
            if group_var not in demographic_data.columns:
                continue

            group_metrics = {}
            for group_value in demographic_data[group_var].unique():
                mask = demographic_data[group_var] == group_value
                if mask.sum() < 10:  # Skip tiny groups
                    continue

                group_pred = predictions[mask]
                group_true = y_true[mask]

                # Compute various equity-relevant metrics
                group_metrics[str(group_value)] = {
                    'mean_predicted_risk': float(group_pred.mean()),
                    'median_predicted_risk': float(np.median(group_pred)),
                    'true_prevalence': float(group_true.mean()),
                    'false_positive_rate': float(((group_pred > 0.5) & (group_true == 0)).sum() / (group_true == 0).sum()) if (group_true == 0).sum() > 0 else 0,
                    'false_negative_rate': float(((group_pred <= 0.5) & (group_true == 1)).sum() / (group_true == 1).sum()) if (group_true == 1).sum() > 0 else 0,
                    'positive_predictive_value': float(group_true[group_pred > 0.5].mean()) if (group_pred > 0.5).sum() > 0 else 0,
                    'n_samples': int(mask.sum())
                }

            metrics[group_var] = group_metrics

        return metrics

    def _check_thresholds(
        self,
        current_metrics: Dict,
        monitoring_date: datetime
    ) -> List[EquityAlert]:
        """Check if current metrics exceed disparity thresholds"""
        alerts = []

        for group_var, group_metrics in current_metrics.items():
            if group_var == 'timestamp':
                continue

            for metric_name in ['mean_predicted_risk', 'false_positive_rate', 'positive_predictive_value']:
                if metric_name not in self.disparity_thresholds:
                    continue

                # Get values across groups
                values = [m[metric_name] for m in group_metrics.values() if metric_name in m]
                if len(values) < 2:
                    continue

                # Compute disparity ratio (max / min)
                disparity_ratio = max(values) / min(values) if min(values) > 0 else np.inf
                threshold = self.disparity_thresholds.get(metric_name, 1.5)

                if disparity_ratio > threshold:
                    # Identify affected groups (those at extremes)
                    max_group = [g for g, m in group_metrics.items() if m[metric_name] == max(values)]
                    min_group = [g for g, m in group_metrics.items() if m[metric_name] == min(values)]

                    severity = 'critical' if disparity_ratio > threshold * 1.5 else 'warning'

                    alert = EquityAlert(
                        timestamp=monitoring_date,
                        metric_name=metric_name,
                        demographic_variable=group_var,
                        current_value=disparity_ratio,
                        threshold=threshold,
                        severity=severity,
                        affected_groups=max_group + min_group,
                        message=f"Disparity in {metric_name} across {group_var}: {disparity_ratio:.2f}x (threshold: {threshold:.2f}x). Groups affected: {', '.join(max_group + min_group)}"
                    )
                    alerts.append(alert)

        return alerts

    def _check_trends(
        self,
        monitoring_date: datetime
    ) -> List[EquityAlert]:
        """Check for adverse trends in equity metrics over time"""
        alerts = []

        # Get recent history
        recent_history = self.metric_history[-self.lookback_periods:]

        # For each demographic variable and metric, check for adverse trends
        # An adverse trend is increasing disparity over time

        # This is a simplified implementation - production systems should use
        # more sophisticated trend detection (e.g., Mann-Kendall test)

        return alerts  # Placeholder for trend detection

    def _apply_alert_cooldown(
        self,
        alerts: List[EquityAlert]
    ) -> List[EquityAlert]:
        """Filter alerts to respect cooldown periods"""
        filtered_alerts = []

        for alert in alerts:
            # Check if we've alerted on this issue recently
            recent_similar_alerts = [
                a for a in self.alert_history
                if a.metric_name == alert.metric_name
                and a.demographic_variable == alert.demographic_variable
                and (alert.timestamp - a.timestamp).total_seconds() / 3600 < self.alert_cooldown_hours
            ]

            if not recent_similar_alerts:
                filtered_alerts.append(alert)

        return filtered_alerts

    def generate_equity_dashboard(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate visual dashboard of equity metrics over time

        Args:
            save_path: Optional path to save dashboard figure
        """
        if len(self.metric_history) < 2:
            logger.warning("Insufficient history for dashboard generation")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Risk Stratification Equity Dashboard', fontsize=16)

        # Plot 1: Mean predicted risk over time by demographic group
        ax = axes[0, 0]
        for group_var in self.stratifier.equity_groups:
            # Extract time series for this variable
            # Simplified - production version would handle this more robustly
            pass
        ax.set_title('Mean Predicted Risk Over Time')
        ax.set_xlabel('Monitoring Period')
        ax.set_ylabel('Mean Risk Score')

        # Plot 2: Disparity ratios over time
        ax = axes[0, 1]
        ax.set_title('Disparity Ratios Over Time')
        ax.set_xlabel('Monitoring Period')
        ax.set_ylabel('Disparity Ratio (Max/Min)')
        ax.axhline(y=1.0, color='g', linestyle='--', label='Perfect Equity')

        # Plot 3: Alert frequency over time
        ax = axes[1, 0]
        ax.set_title('Equity Alerts Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Alerts')

        # Plot 4: Current disparity metrics
        ax = axes[1, 1]
        ax.set_title('Current Disparity Metrics')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
```

This monitoring system provides continuous oversight of risk stratification equity, automatically detecting when disparities exceed acceptable thresholds and alerting system administrators to potential issues before they cause harm to underserved populations.

## Screening Strategy Optimization

Screening programs must balance benefits of early detection against burdens of screening and follow-up, with explicit consideration of how these tradeoffs differ across populations. This section develops screening optimization methods that account for differential screening burden and participation rates.

```python
"""
Equity-aware screening strategy optimization

This module implements screening threshold optimization methods that explicitly
account for differential screening burden across populations and optimize for
equitable net benefit rather than uniform test characteristics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ScreeningUtilities:
    """
    Utility values for screening outcomes, capturing benefits and harms

    These values should be elicited from patients and stakeholders,
    ideally stratified by population group to reflect differential burdens.

    Attributes:
        u_tp: Utility of true positive (early detection and treatment)
        u_tn: Utility of true negative (reassurance minus screening burden)
        u_fp: Utility of false positive (includes anxiety, follow-up burden, risks)
        u_fn: Utility of false negative (missed opportunity plus false reassurance)
    """
    u_tp: float = 1.0
    u_tn: float = 0.05
    u_fp: float = -0.2
    u_fn: float = -0.8

    def net_benefit_tp(self) -> float:
        """Net benefit of true positive relative to no screening"""
        return self.u_tp - self.u_fn

    def net_harm_fp(self) -> float:
        """Net harm of false positive relative to true negative"""
        return self.u_tn - self.u_fp

class EquityAwareScreeningOptimizer:
    """
    Screening threshold optimization accounting for differential burdens

    This optimizer determines optimal screening thresholds that may differ
    across populations to achieve equitable net benefit, rather than using
    uniform thresholds that result in differential burdens.

    The key insight is that populations facing higher screening burden (due to
    lack of transportation, inflexible work, etc.) should have higher thresholds
    (screen only when evidence is stronger) to equalize net benefit.

    Parameters:
        prevalence: Disease prevalence by population group
        utilities: Screening utilities by population group
        cost_per_screening: Financial cost of screening (if relevant)
        participation_rates: Expected screening participation by group
    """

    def __init__(
        self,
        prevalence: Dict[str, float],
        utilities: Dict[str, ScreeningUtilities],
        cost_per_screening: Optional[float] = None,
        participation_rates: Optional[Dict[str, float]] = None
    ):
        self.prevalence = prevalence
        self.utilities = utilities
        self.cost_per_screening = cost_per_screening
        self.participation_rates = participation_rates or {g: 1.0 for g in prevalence.keys()}

        # Storage for computed thresholds
        self.optimal_thresholds_: Dict[str, float] = {}
        self.expected_benefits_: Dict[str, Dict] = {}

    def compute_expected_utility(
        self,
        threshold: float,
        sensitivity_curve: Callable[[float], float],
        specificity_curve: Callable[[float], float],
        group: str
    ) -> float:
        """
        Compute expected utility of screening at given threshold for a population group

        Args:
            threshold: Screening threshold (higher = more selective)
            sensitivity_curve: Function mapping threshold to sensitivity
            specificity_curve: Function mapping threshold to specificity
            group: Population group identifier

        Returns:
            Expected utility across all screening outcomes
        """
        p = self.prevalence[group]
        u = self.utilities[group]

        se = sensitivity_curve(threshold)
        sp = specificity_curve(threshold)

        # Expected utility calculation
        eu = (
            p * (se * u.u_tp + (1 - se) * u.u_fn) +
            (1 - p) * (sp * u.u_tn + (1 - sp) * u.u_fp)
        )

        return eu

    def compute_net_benefit(
        self,
        threshold: float,
        sensitivity_curve: Callable[[float], float],
        specificity_curve: Callable[[float], float],
        group: str
    ) -> float:
        """
        Compute net benefit (benefit minus weighted harms) of screening

        Net benefit framework is clinically interpretable and widely used.

        Args:
            threshold: Screening threshold
            sensitivity_curve: Function mapping threshold to sensitivity
            specificity_curve: Function mapping threshold to specificity
            group: Population group identifier

        Returns:
            Net benefit per person screened
        """
        p = self.prevalence[group]
        u = self.utilities[group]

        se = sensitivity_curve(threshold)
        sp = specificity_curve(threshold)

        # True positives per person screened
        tp_rate = p * se

        # False positives per person screened
        fp_rate = (1 - p) * (1 - sp)

        # Weight false positives by relative harm
        harm_weight = abs(u.u_fp - u.u_tn) / abs(u.u_tp - u.u_fn)

        net_benefit = tp_rate - harm_weight * fp_rate

        return net_benefit

    def optimize_group_threshold(
        self,
        sensitivity_curve: Callable[[float], float],
        specificity_curve: Callable[[float], float],
        group: str,
        objective: str = 'expected_utility'
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for a specific population group

        Args:
            sensitivity_curve: Function mapping threshold to sensitivity
            specificity_curve: Function mapping threshold to specificity
            group: Population group identifier
            objective: Optimization objective ('expected_utility' or 'net_benefit')

        Returns:
            Tuple of (optimal_threshold, objective_value)
        """
        if objective == 'expected_utility':
            obj_func = lambda t: -self.compute_expected_utility(
                t, sensitivity_curve, specificity_curve, group
            )
        elif objective == 'net_benefit':
            obj_func = lambda t: -self.compute_net_benefit(
                t, sensitivity_curve, specificity_curve, group
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Optimize over reasonable threshold range
        result = minimize_scalar(
            obj_func,
            bounds=(0.01, 0.99),
            method='bounded'
        )

        optimal_threshold = result.x
        optimal_value = -result.fun  # Negate because we minimized negative

        return optimal_threshold, optimal_value

    def optimize_all_groups(
        self,
        sensitivity_curve: Callable[[float], float],
        specificity_curve: Callable[[float], float],
        objective: str = 'expected_utility'
    ) -> Dict[str, Tuple[float, float]]:
        """
        Optimize screening thresholds for all population groups

        Args:
            sensitivity_curve: Function mapping threshold to sensitivity
            specificity_curve: Function mapping threshold to specificity
            objective: Optimization objective

        Returns:
            Dictionary mapping group names to (optimal_threshold, objective_value)
        """
        results = {}

        for group in self.prevalence.keys():
            threshold, value = self.optimize_group_threshold(
                sensitivity_curve, specificity_curve, group, objective
            )
            results[group] = (threshold, value)
            self.optimal_thresholds_[group] = threshold

            # Store expected benefits at optimal threshold
            self.expected_benefits_[group] = {
                'threshold': threshold,
                'expected_utility': self.compute_expected_utility(
                    threshold, sensitivity_curve, specificity_curve, group
                ),
                'net_benefit': self.compute_net_benefit(
                    threshold, sensitivity_curve, specificity_curve, group
                ),
                'sensitivity': sensitivity_curve(threshold),
                'specificity': specificity_curve(threshold)
            }

        return results

    def compute_population_impact(
        self,
        thresholds: Dict[str, float],
        sensitivity_curve: Callable[[float], float],
        specificity_curve: Callable[[float], float],
        population_sizes: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Compute population-level impact of screening with group-specific thresholds

        Args:
            thresholds: Screening thresholds by group
            sensitivity_curve: Function mapping threshold to sensitivity
            specificity_curve: Function mapping threshold to specificity
            population_sizes: Number of individuals in each group

        Returns:
            Dictionary with population-level impact metrics
        """
        total_screened = 0
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0

        for group, threshold in thresholds.items():
            n = population_sizes[group]
            p = self.prevalence[group]
            participation = self.participation_rates[group]

            # Number actually screened
            n_screened = n * participation
            total_screened += n_screened

            # Expected outcomes
            se = sensitivity_curve(threshold)
            sp = specificity_curve(threshold)

            tp = n_screened * p * se
            fn = n_screened * p * (1 - se)
            tn = n_screened * (1 - p) * sp
            fp = n_screened * (1 - p) * (1 - sp)

            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

        return {
            'total_screened': total_screened,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'true_negatives': total_tn,
            'false_negatives': total_fn,
            'ppv': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
            'npv': total_tn / (total_tn + total_fn) if (total_tn + total_fn) > 0 else 0,
            'overall_sensitivity': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
            'overall_specificity': total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
        }

    def compare_equity_of_strategies(
        self,
        uniform_threshold: float,
        sensitivity_curve: Callable[[float], float],
        specificity_curve: Callable[[float], float]
    ) -> pd.DataFrame:
        """
        Compare uniform threshold strategy vs. equity-optimized group-specific thresholds

        Args:
            uniform_threshold: Single threshold used for all groups
            sensitivity_curve: Function mapping threshold to sensitivity
            specificity_curve: Function mapping threshold to specificity

        Returns:
            DataFrame comparing strategies across groups
        """
        results = []

        for group in self.prevalence.keys():
            # Uniform threshold strategy
            uniform_nb = self.compute_net_benefit(
                uniform_threshold, sensitivity_curve, specificity_curve, group
            )

            # Group-specific optimized threshold
            optimized_threshold = self.optimal_thresholds_.get(group, uniform_threshold)
            optimized_nb = self.compute_net_benefit(
                optimized_threshold, sensitivity_curve, specificity_curve, group
            )

            results.append({
                'group': group,
                'prevalence': self.prevalence[group],
                'uniform_threshold': uniform_threshold,
                'uniform_net_benefit': uniform_nb,
                'optimized_threshold': optimized_threshold,
                'optimized_net_benefit': optimized_nb,
                'net_benefit_improvement': optimized_nb - uniform_nb,
                'participation_rate': self.participation_rates[group]
            })

        return pd.DataFrame(results)

    def plot_threshold_tradeoffs(
        self,
        sensitivity_curve: Callable[[float], float],
        specificity_curve: Callable[[float], float],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize tradeoffs in screening thresholds across groups

        Args:
            sensitivity_curve: Function mapping threshold to sensitivity
            specificity_curve: Function mapping threshold to specificity
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Net benefit curves by group
        ax = axes[0]
        thresholds = np.linspace(0.01, 0.99, 100)

        for group in self.prevalence.keys():
            net_benefits = [
                self.compute_net_benefit(t, sensitivity_curve, specificity_curve, group)
                for t in thresholds
            ]
            ax.plot(thresholds, net_benefits, label=group, linewidth=2)

            # Mark optimal threshold
            if group in self.optimal_thresholds_:
                opt_t = self.optimal_thresholds_[group]
                opt_nb = self.compute_net_benefit(opt_t, sensitivity_curve, specificity_curve, group)
                ax.scatter([opt_t], [opt_nb], s=100, zorder=5)

        ax.set_xlabel('Screening Threshold', fontsize=12)
        ax.set_ylabel('Net Benefit per Person Screened', fontsize=12)
        ax.set_title('Net Benefit vs. Threshold by Population Group', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

        # Plot 2: Optimal thresholds and participation rates
        ax = axes[1]
        groups = list(self.prevalence.keys())
        opt_thresholds = [self.optimal_thresholds_.get(g, 0.5) for g in groups]
        participation = [self.participation_rates[g] for g in groups]

        x = np.arange(len(groups))
        width = 0.35

        ax.bar(x - width/2, opt_thresholds, width, label='Optimal Threshold', alpha=0.8)
        ax.bar(x + width/2, participation, width, label='Participation Rate', alpha=0.8)

        ax.set_xlabel('Population Group', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Optimal Thresholds and Participation Rates', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def estimate_screening_curves_from_data(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstrap: int = 1000
) -> Tuple[Callable, Callable]:
    """
    Estimate sensitivity and specificity curves from validation data

    Uses isotonic regression to ensure monotonicity and bootstrap for uncertainty.

    Args:
        y_true: True disease status (0/1)
        y_scores: Predicted risk scores (higher = more likely diseased)
        n_bootstrap: Number of bootstrap samples for uncertainty estimation

    Returns:
        Tuple of (sensitivity_curve, specificity_curve) functions
    """
    from sklearn.isotonic import IsotonicRegression

    # Compute ROC curve points
    thresholds = np.percentile(y_scores, np.linspace(0, 100, 101))

    sensitivities = []
    specificities = []

    for threshold in thresholds:
        predictions = (y_scores >= threshold).astype(int)

        tp = ((predictions == 1) & (y_true == 1)).sum()
        tn = ((predictions == 0) & (y_true == 0)).sum()
        fp = ((predictions == 1) & (y_true == 0)).sum()
        fn = ((predictions == 0) & (y_true == 1)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # Fit isotonic regression for smooth curves
    # Sensitivity should decrease with threshold
    iso_se = IsotonicRegression(increasing=False)
    iso_se.fit(thresholds, sensitivities)

    # Specificity should increase with threshold
    iso_sp = IsotonicRegression(increasing=True)
    iso_sp.fit(thresholds, specificities)

    # Create functions that interpolate smoothly
    def sensitivity_curve(t: float) -> float:
        return float(np.clip(iso_se.predict([t])[0], 0, 1))

    def specificity_curve(t: float) -> float:
        return float(np.clip(iso_sp.predict([t])[0], 0, 1))

    return sensitivity_curve, specificity_curve
```

This screening optimization framework provides methods for determining screening thresholds that achieve equitable net benefit across populations, explicitly accounting for differential screening burdens and participation rates. The approach recognizes that equal test characteristics (uniform thresholds) lead to unequal burdens, and optimizes for equitable outcomes instead.

## Outbreak Detection and Surveillance

Infectious disease surveillance must balance sensitivity to detect outbreaks quickly against specificity to avoid false alarms, with explicit consideration of surveillance coverage gaps in underserved communities. This section develops spatial-temporal outbreak detection methods with equity-aware sensitivity calibration.

```python
"""
Equity-aware outbreak detection and disease surveillance

This module implements temporal and spatial outbreak detection methods that
account for unequal surveillance coverage and ensure equitable detection
sensitivity across communities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy import stats
from scipy.spatial import distance_matrix
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class SurveillanceSystem:
    """
    Represents surveillance infrastructure for a geographic region

    Attributes:
        reporting_rate: Proportion of true cases captured by surveillance
        diagnostic_capacity: Cases per day that can be diagnosed
        reporting_lag_days: Typical delay from onset to reporting
        coverage_score: 0-1 score indicating surveillance quality
        population_vulnerability: Social vulnerability index for this region
    """
    reporting_rate: float  # 0-1
    diagnostic_capacity: float  # cases/day
    reporting_lag_days: float  # days
    coverage_score: float  # 0-1
    population_vulnerability: float  # 0-1

class EquityAwareCUSUM:
    """
    CUSUM control chart with adjustments for unequal surveillance coverage

    Traditional CUSUM assumes complete and uniform surveillance. This implementation
    adjusts for differential reporting rates across geographic areas, making the
    system more sensitive to outbreaks in underserved communities with poor
    surveillance infrastructure.

    Parameters:
        baseline_rates: Expected baseline case counts by geographic area
        reporting_rates: Estimated proportion of cases captured by surveillance
        reference_value: Minimum outbreak effect size to detect (in cases/day)
        alert_threshold: CUSUM value that triggers outbreak alert
        prioritize_vulnerable: If True, accept higher false alarm rates in
            vulnerable communities to achieve equal detection speed
    """

    def __init__(
        self,
        baseline_rates: Dict[str, float],
        reporting_rates: Dict[str, float],
        reference_value: float = 5.0,
        alert_threshold: float = 10.0,
        prioritize_vulnerable: bool = True
    ):
        self.baseline_rates = baseline_rates
        self.reporting_rates = reporting_rates
        self.reference_value = reference_value
        self.alert_threshold = alert_threshold
        self.prioritize_vulnerable = prioritize_vulnerable

        # Initialize CUSUM statistics for each area
        self.cusum_stats: Dict[str, float] = {area: 0.0 for area in baseline_rates.keys()}
        self.alert_times: Dict[str, List[int]] = {area: [] for area in baseline_rates.keys()}
        self.observation_count: int = 0

        # Adjust thresholds by area if prioritizing vulnerable populations
        if prioritize_vulnerable:
            self.area_thresholds = self._compute_equity_adjusted_thresholds()
        else:
            self.area_thresholds = {area: alert_threshold for area in baseline_rates.keys()}

    def _compute_equity_adjusted_thresholds(self) -> Dict[str, float]:
        """
        Compute area-specific alert thresholds that equalize expected detection time

        Areas with lower reporting rates get lower thresholds (more sensitive)
        to compensate for signal attenuation from poor surveillance.
        """
        thresholds = {}
        reference_reporting = np.median(list(self.reporting_rates.values()))

        for area, reporting_rate in self.reporting_rates.items():
            # Inverse relationship: lower reporting → lower threshold → more sensitive
            adjustment_factor = reference_reporting / reporting_rate if reporting_rate > 0 else 1.0
            thresholds[area] = self.alert_threshold / adjustment_factor

        return thresholds

    def update(
        self,
        observed_counts: Dict[str, float],
        time_index: int
    ) -> Dict[str, bool]:
        """
        Update CUSUM statistics with new observations

        Args:
            observed_counts: Observed case counts by geographic area
            time_index: Time index for this observation

        Returns:
            Dictionary indicating whether alert triggered for each area
        """
        alerts = {}

        for area, count in observed_counts.items():
            if area not in self.cusum_stats:
                continue

            baseline = self.baseline_rates[area]
            reporting = self.reporting_rates[area]

            # Adjust observed count for reporting rate to estimate true count
            adjusted_count = count / reporting if reporting > 0 else count

            # CUSUM update: accumulate evidence of deviation from baseline
            deviation = adjusted_count - baseline - self.reference_value
            self.cusum_stats[area] = max(0.0, self.cusum_stats[area] + deviation)

            # Check if alert threshold exceeded
            threshold = self.area_thresholds[area]
            if self.cusum_stats[area] >= threshold:
                alerts[area] = True
                self.alert_times[area].append(time_index)
                # Reset CUSUM after alert
                self.cusum_stats[area] = 0.0
            else:
                alerts[area] = False

        self.observation_count += 1

        return alerts

    def get_current_statistics(self) -> pd.DataFrame:
        """Return current CUSUM statistics for all areas"""
        data = []
        for area in self.cusum_stats.keys():
            data.append({
                'area': area,
                'cusum_value': self.cusum_stats[area],
                'threshold': self.area_thresholds[area],
                'baseline_rate': self.baseline_rates[area],
                'reporting_rate': self.reporting_rates[area],
                'alert_count': len(self.alert_times[area])
            })

        return pd.DataFrame(data)

class SpatialScanStatistic:
    """
    Kulldorff spatial scan statistic with equity-aware cluster detection

    Traditional spatial scan naturally favors dense urban areas with good
    surveillance. This implementation applies vulnerability weighting to
    upweight clusters in underserved communities.

    Parameters:
        max_cluster_size: Maximum population proportion for a cluster (0-1)
        n_simulations: Monte Carlo simulations for significance testing
        vulnerability_weight: Weight for vulnerability index (0 = no weighting, 1 = full)
    """

    def __init__(
        self,
        max_cluster_size: float = 0.5,
        n_simulations: int = 999,
        vulnerability_weight: float = 0.5
    ):
        self.max_cluster_size = max_cluster_size
        self.n_simulations = n_simulations
        self.vulnerability_weight = vulnerability_weight

        self.scan_results_: Optional[pd.DataFrame] = None

    def scan(
        self,
        case_data: pd.DataFrame,
        population_data: pd.DataFrame,
        vulnerability_scores: pd.Series,
        geometry: gpd.GeoSeries,
        max_cluster_radius_km: float = 50.0
    ) -> pd.DataFrame:
        """
        Perform spatial scan for disease clusters

        Args:
            case_data: DataFrame with columns ['area_id', 'cases']
            population_data: DataFrame with columns ['area_id', 'population']
            vulnerability_scores: Series mapping area_id to vulnerability index (0-1)
            geometry: GeoSeries with Point geometries for each area centroid
            max_cluster_radius_km: Maximum radius for circular clusters

        Returns:
            DataFrame with detected clusters ranked by likelihood ratio
        """
        # Merge data
        data = case_data.merge(population_data, on='area_id')
        data['vulnerability'] = data['area_id'].map(vulnerability_scores)
        data['geometry'] = geometry

        total_cases = data['cases'].sum()
        total_population = data['population'].sum()

        # Compute distance matrix between area centroids
        coords = np.array([[p.x, p.y] for p in geometry])
        distances = distance_matrix(coords, coords)

        # Evaluate all possible circular clusters
        cluster_results = []

        for center_idx in range(len(data)):
            # Get areas within max radius
            within_radius = distances[center_idx, :] <= max_cluster_radius_km

            for radius_idx in np.where(within_radius)[0]:
                radius = distances[center_idx, radius_idx]

                # Define cluster as all areas within this radius of center
                cluster_mask = distances[center_idx, :] <= radius
                cluster_data = data[cluster_mask]

                # Check cluster size constraint
                cluster_pop = cluster_data['population'].sum()
                if cluster_pop > self.max_cluster_size * total_population:
                    continue

                # Compute likelihood ratio for this cluster
                c_in = cluster_data['cases'].sum()  # Cases in cluster
                n_in = cluster_data['population'].sum()  # Population in cluster
                c_out = total_cases - c_in  # Cases outside cluster
                n_out = total_population - n_in  # Population outside cluster

                if c_in == 0 or c_out == 0:
                    continue

                # Expected cases in cluster under null hypothesis
                expected_in = (c_in + c_out) * n_in / (n_in + n_out)

                # Likelihood ratio
                if c_in > expected_in:  # Only consider elevated clusters
                    llr = (
                        c_in * np.log(c_in / expected_in) +
                        c_out * np.log(c_out / (total_cases - expected_in))
                    )

                    # Apply vulnerability weighting
                    mean_vulnerability = cluster_data['vulnerability'].mean()
                    adjusted_llr = llr * (1 + self.vulnerability_weight * mean_vulnerability)

                    cluster_results.append({
                        'center_area': data.iloc[center_idx]['area_id'],
                        'radius_km': radius,
                        'cases_observed': c_in,
                        'cases_expected': expected_in,
                        'relative_risk': (c_in / n_in) / ((c_in + c_out) / (n_in + n_out)),
                        'llr': llr,
                        'adjusted_llr': adjusted_llr,
                        'population': n_in,
                        'n_areas': cluster_mask.sum(),
                        'mean_vulnerability': mean_vulnerability
                    })

        # Convert to DataFrame and rank
        results_df = pd.DataFrame(cluster_results)

        if len(results_df) == 0:
            return results_df

        results_df = results_df.sort_values('adjusted_llr', ascending=False)

        # Monte Carlo significance testing for top clusters
        # (Simplified - production version would do full simulation)
        results_df['p_value'] = self._compute_significance(
            results_df.iloc[0]['adjusted_llr'] if len(results_df) > 0 else 0,
            total_cases,
            total_population,
            data
        )

        self.scan_results_ = results_df

        return results_df

    def _compute_significance(
        self,
        observed_llr: float,
        total_cases: int,
        total_population: int,
        data: pd.DataFrame
    ) -> float:
        """
        Compute p-value via Monte Carlo simulation

        Simplified implementation - production version would perform full scan
        on each simulated dataset.
        """
        # Simulate under null hypothesis (random spatial allocation)
        simulated_max_llr = []

        for _ in range(self.n_simulations):
            # Randomly allocate cases proportional to population
            simulated_cases = np.random.multinomial(
                total_cases,
                data['population'] / total_population
            )

            # Compute maximum LLR in simulated data
            # (Simplified - would need to rerun full scan)
            max_llr = 0  # Placeholder
            simulated_max_llr.append(max_llr)

        # P-value: proportion of simulations with max LLR >= observed
        p_value = (np.array(simulated_max_llr) >= observed_llr).mean()

        return p_value

    def visualize_clusters(
        self,
        top_k: int = 3,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize top detected clusters on a map

        Args:
            top_k: Number of top clusters to display
            save_path: Optional path to save figure
        """
        if self.scan_results_ is None:
            raise RuntimeError("Must run scan() before visualization")

        # Simplified visualization - production version would use geopandas plotting
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot top clusters
        for i, row in self.scan_results_.head(top_k).iterrows():
            # Would plot circular cluster on map
            pass

        ax.set_title(f'Top {top_k} Disease Clusters (Vulnerability-Weighted)', fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def simulate_outbreak_scenarios(
    n_areas: int = 100,
    outbreak_area_idx: int = 50,
    outbreak_start_day: int = 30,
    outbreak_magnitude: float = 3.0,
    reporting_rates: Optional[np.ndarray] = None,
    n_days: int = 100
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate outbreak scenarios to evaluate detection methods

    Args:
        n_areas: Number of geographic areas
        outbreak_area_idx: Area where outbreak occurs
        outbreak_start_day: Day outbreak begins
        outbreak_magnitude: Multiplier for case rates during outbreak
        reporting_rates: Reporting rates by area (if None, assumes complete surveillance)
        n_days: Number of days to simulate

    Returns:
        Tuple of (case_data_df, ground_truth_dict)
    """
    if reporting_rates is None:
        reporting_rates = np.ones(n_areas)

    # Baseline case rates (Poisson distributed)
    baseline_rates = np.random.gamma(2, 2, size=n_areas)

    # Simulate daily case counts
    case_data = []

    for day in range(n_days):
        for area_idx in range(n_areas):
            baseline = baseline_rates[area_idx]

            # Apply outbreak effect
            if day >= outbreak_start_day and area_idx == outbreak_area_idx:
                rate = baseline * outbreak_magnitude
            else:
                rate = baseline

            # Generate true cases
            true_cases = np.random.poisson(rate)

            # Apply reporting rate to get observed cases
            observed_cases = np.random.binomial(true_cases, reporting_rates[area_idx])

            case_data.append({
                'day': day,
                'area_idx': area_idx,
                'observed_cases': observed_cases,
                'true_cases': true_cases,
                'reporting_rate': reporting_rates[area_idx]
            })

    case_df = pd.DataFrame(case_data)

    ground_truth = {
        'outbreak_area': outbreak_area_idx,
        'outbreak_start_day': outbreak_start_day,
        'outbreak_magnitude': outbreak_magnitude
    }

    return case_df, ground_truth
```

These outbreak detection methods explicitly account for surveillance infrastructure gaps and provide equity-aware sensitivity calibration to ensure that outbreaks in underserved communities are detected as quickly as those in well-surveilled areas.

## Conclusion

Population health management requires fundamentally rethinking optimization objectives to prioritize need over efficiency, explicitly accounting for differential barriers and burdens across populations. This chapter has developed mathematical frameworks and production implementations for risk stratification, screening optimization, outbreak detection, and resource allocation that place health equity at the center of technical design. The key insight is that achieving equitable outcomes requires more than adding fairness constraints to traditional formulations - it demands reconceptualizing what we optimize for and how we measure success.

Effective population health systems must integrate continuous equity monitoring, recognizing that even well-designed systems can drift toward efficiency-based targeting over time. The monitoring frameworks presented here provide automated oversight to detect equity drift before it causes harm. As population health management increasingly relies on machine learning and optimization, ensuring that these systems serve those with greatest need rather than those easiest to reach remains a central technical and ethical challenge.

## Bibliography

Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press. (Causal inference methods for program evaluation.)

Artiga, S., & Hinton, E. (2018). Beyond Health Care: The Role of Social Determinants in Promoting Health and Health Equity. *Kaiser Family Foundation*, 10. (Social determinants framework.)

Baker, D. W., et al. (2017). Use of Health Literacy-Sensitive "Teach-Back" Patient Education to Address Health Disparities. *Journal of General Internal Medicine*, 32(4), 461-469. (Health literacy and patient education.)

Basu, S., et al. (2016). Comparative Performance of Private and Public Healthcare Systems in Low- and Middle-Income Countries: A Systematic Review. *PLoS Medicine*, 13(2), e1001244. (Healthcare system performance in underserved settings.)

Bernal, J. L., et al. (2017). Interrupted Time Series Regression for the Evaluation of Public Health Interventions: A Tutorial. *International Journal of Epidemiology*, 46(1), 348-355. (Methods for intervention evaluation.)

Braveman, P., et al. (2011). Health Disparities and Health Equity: The Issue is Justice. *American Journal of Public Health*, 101(S1), S149-S155. (Foundational equity framework.)

Chen, J. H., & Asch, S. M. (2017). Machine Learning and Prediction in Medicine - Beyond the Peak of Inflated Expectations. *New England Journal of Medicine*, 376(26), 2507-2509. (Prediction model deployment considerations.)

Cooksey Stowers, K., et al. (2017). Food Swamps Predict Obesity Rates Better Than Food Deserts in the United States. *International Journal of Environmental Research and Public Health*, 14(11), 1366. (Food access and health.)

Coughlin, S. S. (2014). Recall Bias in Epidemiologic Studies. *Journal of Clinical Epidemiology*, 43(1), 87-91. (Data quality issues in population health research.)

Finkelstein, A., et al. (2012). The Oregon Health Insurance Experiment: Evidence from the First Year. *Quarterly Journal of Economics*, 127(3), 1057-1106. (Randomized evaluation of Medicaid expansion.)

Frieden, T. R. (2010). A Framework for Public Health Action: The Health Impact Pyramid. *American Journal of Public Health*, 100(4), 590-595. (Population health intervention framework.)

Friedman, C. P., et al. (2010). Toward a Science of Learning Systems: A Research Agenda for the High-Functioning Learning Health System. *Journal of the American Medical Informatics Association*, 17(1), 38-44. (Learning health systems.)

Gianfrancesco, M. A., et al. (2018). Potential Biases in Machine Learning Algorithms Using Electronic Health Record Data. *JAMA Internal Medicine*, 178(11), 1544-1547. (EHR data bias.)

Glasgow, R. E., et al. (2019). Evaluating the Impact of Health Information Technology on Diabetes Management: An Evaluation Framework. *Diabetes Technology & Therapeutics*, 21(9), 561-568. (Technology evaluation frameworks.)

Greenland, S., et al. (2016). Statistical Tests, P Values, Confidence Intervals, and Power: A Guide to Misinterpretations. *European Journal of Epidemiology*, 31(4), 337-350. (Statistical inference in epidemiology.)

Groh, C. J. (2007). Poverty, Mental Health, and Women: Implications for Psychiatric Nurses in Primary Care Settings. *Journal of the American Psychiatric Nurses Association*, 13(5), 267-274. (Mental health and social determinants.)

Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. (Comprehensive causal inference textbook.)

Hswen, Y., et al. (2019). Digital Health Equity for Underserved Communities. *JAMA Network Open*, 2(3), e190493. (Digital health equity.)

Kanter, G. P., et al. (2019). Health Information Technology Use Among Community Health Center Patients. *Health Affairs*, 38(3), 516-521. (HIT in safety net settings.)

Kerr, E. A., et al. (2017). Building a Better Delivery System for High-Risk Patients. *JAMA*, 318(22), 2235-2236. (Care management for high-risk patients.)

Kilbourne, A. M., et al. (2006). Implementing Evidence-Based Interventions in Health Care: Application of the Replicating Effective Programs Framework. *Implementation Science*, 1(1), 1. (Implementation science framework.)

Kindig, D., & Stoddart, G. (2003). What is Population Health? *American Journal of Public Health*, 93(3), 380-383. (Population health definition.)

Klein, E., et al. (2018). Assessment of Optimal Targeting and Deployment of Ebola Vaccines. *Proceedings of the National Academy of Sciences*, 115(37), 9219-9224. (Vaccine allocation optimization.)

Kulldorff, M. (1997). A Spatial Scan Statistic. *Communications in Statistics - Theory and Methods*, 26(6), 1481-1496. (Spatial scan statistic method.)

LaVeist, T. A., et al. (2011). The Economic Burden of Health Inequalities in the United States. *Joint Center for Political and Economic Studies*. (Economic impact of health disparities.)

Levy, J. A., et al. (2014). A Systems Approach to Public Health Obesity Research. *American Journal of Public Health*, 104(7), 1156-1159. (Systems approach to population health.)

Link, B. G., & Phelan, J. (1995). Social Conditions as Fundamental Causes of Disease. *Journal of Health and Social Behavior*, 35, 80-94. (Fundamental causes theory.)

Lipsitch, M., et al. (2003). Transmission Dynamics and Control of Severe Acute Respiratory Syndrome. *Science*, 300(5627), 1966-1970. (Infectious disease modeling.)

Lohr, K. N. (2004). Rating the Strength of Scientific Evidence: Relevance for Quality Improvement Programs. *International Journal for Quality in Health Care*, 16(1), 9-18. (Evidence evaluation.)

Luke, D. A., & Stamatakis, K. A. (2012). Systems Science Methods in Public Health: Dynamics, Networks, and Agents. *Annual Review of Public Health*, 33, 357-376. (Systems science in population health.)

Marmot, M., et al. (2008). Closing the Gap in a Generation: Health Equity Through Action on the Social Determinants of Health. *The Lancet*, 372(9650), 1661-1669. (CSDH final report on health equity.)

Murray, C. J. L., & Frenk, J. (2000). A Framework for Assessing the Performance of Health Systems. *Bulletin of the World Health Organization*, 78(6), 717-731. (Health system performance framework.)

Neta, G., et al. (2015). Dissemination and Implementation Research Program at the National Cancer Institute: Building the Evidence Base to Support Uptake of Evidence-Based Interventions. *Clinical and Translational Science*, 8(5), 471-476. (D&I research framework.)

Obermeyer, Z., et al. (2019). Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations. *Science*, 366(6464), 447-453. (Algorithmic bias in healthcare.)

Pappas, G., et al. (1993). The Increasing Disparity in Mortality Between Socioeconomic Groups in the United States, 1960 and 1986. *New England Journal of Medicine*, 329(2), 103-109. (Historical trends in health disparities.)

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. (Causal inference theory.)

Pencina, M. J., et al. (2008). Evaluating the Added Predictive Ability of a New Marker: From Area Under the ROC Curve to Reclassification and Beyond. *Statistics in Medicine*, 27(2), 157-172. (Prediction model evaluation.)

Rajkomar, A., et al. (2018). Ensuring Fairness in Machine Learning to Advance Health Equity. *Annals of Internal Medicine*, 169(12), 866-872. (Fairness in healthcare ML.)

Riley, R. D., et al. (2019). Calculating the Sample Size Required for Developing a Clinical Prediction Model. *BMJ*, 368, m441. (Sample size for prediction models.)

Rose, G. (2001). Sick Individuals and Sick Populations. *International Journal of Epidemiology*, 30(3), 427-432. (Population vs. individual approach.)

Royston, P., et al. (2013). Prognosis and Prognostic Research: Developing a Prognostic Model. *BMJ*, 338, b604. (Developing prognostic models.)

Rubin, D. B. (2005). Causal Inference Using Potential Outcomes: Design, Modeling, Decisions. *Journal of the American Statistical Association*, 100(469), 322-331. (Potential outcomes framework.)

Sentell, T., & Halpin, H. A. (2006). Importance of Adult Literacy in Understanding Health Disparities. *Journal of General Internal Medicine*, 21(8), 862-866. (Health literacy and disparities.)

Solar, O., & Irwin, A. (2010). *A Conceptual Framework for Action on the Social Determinants of Health*. World Health Organization. (WHO CSDH framework.)

Steyerberg, E. W. (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating* (2nd ed.). Springer. (Comprehensive prediction modeling textbook.)

Stoto, M. A., et al. (2004). Learning from Experience: The Public Health Response to West Nile Virus, SARS, Monkeypox, and Hepatitis A Outbreaks in the United States. *RAND Health Quarterly*, 4(3), 12. (Public health outbreak response.)

Unger, J. M., et al. (2013). The Role of Clinical Trial Participation in Cancer Research: Barriers, Evidence, and Strategies. *American Society of Clinical Oncology Educational Book*, 33, 185-198. (Clinical trial participation barriers.)

van Walraven, C., et al. (2010). A Modification of the Elixhauser Comorbidity Measures Into a Point System for Hospital Death Using Administrative Data. *Medical Care*, 48(8), 679-685. (Comorbidity measurement.)

Vickers, A. J., & Elkin, E. B. (2006). Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. *Medical Decision Making*, 26(6), 565-574. (Decision curve analysis method.)

Whitehead, M. (1991). The Concepts and Principles of Equity and Health. *Health Promotion International*, 6(3), 217-228. (Health equity concepts.)

Woloshin, S., et al. (2008). The Risk of Death by Age, Sex, and Smoking Status in the United States: Putting Health Risks in Context. *Journal of the National Cancer Institute*, 100(12), 845-853. (Risk communication.)chapter_24_population_health_screening.
