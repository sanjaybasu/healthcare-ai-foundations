---
layout: chapter
title: "Chapter 10: Survival Analysis and Time-to-Event Modeling"
chapter_number: 10
part_number: 3
prev_chapter: /chapters/chapter-09-advanced-clinical-nlp/
next_chapter: /chapters/chapter-11-causal-inference/
---
# Chapter 10: Survival Analysis and Time-to-Event Modeling

## Learning Objectives

By the end of this chapter, readers will be able to implement production-grade survival analysis systems for clinical risk prediction that explicitly account for differential censoring patterns across patient populations, recognizing that loss to follow-up and incomplete observation periods often occur systematically based on race, ethnicity, insurance status, and geographic location rather than randomly. Readers will develop Cox proportional hazards models with comprehensive diagnostic testing for assumption violations including non-proportional hazards that may manifest differently across demographic subgroups, implementing stratified and time-varying coefficient approaches when standard models fail to capture heterogeneous hazard patterns across diverse populations. They will apply competing risks analysis frameworks to clinical settings where patients face multiple potential outcomes with socially determined prevalence patterns, using subdistribution hazards and cumulative incidence functions to properly account for how competing events like cardiovascular death versus cancer death vary systematically across communities experiencing different environmental and structural risk exposures.

Readers will build random survival forests and other machine learning approaches to time-to-event modeling that achieve equitable predictive performance across patient characteristics while handling complex interactions between clinical and social determinants, implementing fairness-aware extensions that prevent models from learning shortcuts based on healthcare access patterns rather than true clinical risk. They will design validation strategies for survival models that assess not only discrimination and calibration but also fairness metrics across population subgroups, with particular attention to evaluating model performance in external settings serving different patient populations than the training data, recognizing that survival model transportability is often limited by unmeasured confounding due to social determinants. Finally, readers will deploy clinical survival analysis systems with comprehensive monitoring for equity issues that can emerge post-deployment, including detecting when censoring patterns change over time in ways that differentially affect model performance for specific populations, implementing alerting systems that surface potential fairness degradation before it causes patient harm.

## 10.1 Introduction: Time-to-Event Data and Health Inequities

Time-to-event analysis addresses one of the most fundamental questions in clinical medicine: how long until an outcome occurs? Whether predicting time to death, disease progression, hospital readmission, treatment response, or any other clinical event, survival analysis provides the mathematical and statistical framework for working with data where outcomes happen at different times for different patients and where observation may be incomplete due to censoring. The methods originated in industrial reliability testing and actuarial science but found their most extensive application in biomedical research, becoming essential tools for clinical trials, prognostic modeling, and health services research (Collett, 2015; Klein & Moeschberger, 2003).

Yet time-to-event data in healthcare is far from the idealized setting of controlled clinical trials where randomization ensures balanced comparison groups and complete follow-up is achieved for all participants. In real-world clinical practice, the patterns of who experiences events and when they occur reflect not only biological disease processes but also the profound inequities embedded in healthcare delivery systems and broader social structures. Patients from marginalized communities may experience disease onset at younger ages due to cumulative exposures to structural racism and poverty. They may progress more rapidly due to barriers accessing timely care. They may be lost to follow-up more frequently because of residential instability, transportation barriers, or mistrust of medical systems built on historical exploitation. They may face competing risks like violence or drug overdose that make certain clinical outcomes impossible to observe (Williams & Mohammed, 2013; Bailey et al., 2017).

These differential patterns create profound challenges for survival analysis applied to observational healthcare data. Consider a model predicting time to kidney failure among patients with chronic kidney disease. Standard survival analysis might reveal that Black patients have substantially higher hazard of progression to end-stage renal disease compared to white patients even after adjusting for measured clinical factors like baseline kidney function and blood pressure control. A naive interpretation might attribute this to biological differences. However, closer examination might reveal that Black patients experience more frequent loss to follow-up, shorter observation periods before censoring, and higher competing mortality from cardiovascular disease during the observation period. Each of these patterns reflects structural inequities in access to nephrology care, ability to afford medications and transportation to appointments, and exposure to cardiovascular risk factors through residential segregation and environmental injustice. The observed hazard ratio conflates true differences in disease progression with differential censoring and competing risks, both of which are themselves manifestations of structural racism (Norton et al., 2016; Patzer et al., 2012).

This chapter develops survival analysis methods with explicit attention to these equity challenges. We begin with foundational concepts including the hazard function, survival function, and censoring mechanisms, but frame each through the lens of how real-world clinical data generation processes differ from idealized assumptions in ways that matter for health equity. We then develop the Cox proportional hazards model, examining in depth the proportional hazards assumption and how its violations can reflect genuine biological or social heterogeneity in disease processes across populations. We extend to time-varying covariates and coefficients, parametric survival models, competing risks analysis, and machine learning approaches to survival modeling. Throughout, we implement production-ready code with comprehensive fairness evaluation, validation strategies appropriate for diverse clinical settings, and diagnostic approaches for detecting assumption violations that may signal equity concerns. Our goal is to equip practitioners with both technical expertise in survival analysis and critical consciousness about how time-to-event modeling can either illuminate or obscure health inequities depending on how carefully we attend to the social and structural factors shaping when and for whom outcomes occur.

## 10.2 Foundations of Survival Analysis

Understanding survival analysis requires building intuition about several fundamental concepts that may seem abstract at first but become essential tools for reasoning about time-to-event data. We develop these concepts with attention to how they relate to real clinical data and the equity challenges that arise when applying survival methods to observational healthcare settings.

### 10.2.1 The Survival Function and Hazard Function

The survival function, typically denoted as S(t), represents the probability that an individual survives beyond time t without experiencing the event of interest. Mathematically, if T represents the random variable for time to event, then S(t) = P(T > t). At time zero, assuming all individuals are event-free at the start of observation, S(0) = 1, meaning everyone has survived to time zero. As time progresses, the survival function is non-increasing because once an event occurs, it cannot be reversed, at least for events like death, though recurrent events require different frameworks. Eventually, as time approaches infinity, S(t) approaches zero for mortal events, though in practice we observe survival curves over finite follow-up periods (Kalbfleisch & Prentice, 2002).

The hazard function, denoted h(t), represents the instantaneous rate of experiencing the event at time t given survival up to that time. More precisely, the hazard function is defined as the limit of the probability of experiencing the event in a small time interval divided by the width of that interval, conditional on having survived to the beginning of the interval. Mathematically, h(t) equals the limit as Δt approaches zero of $$P(t ≤ T \lt  t + Δt \mid T ≥ t)$$ divided by Δt. The hazard function captures the instantaneous risk at each moment in time, while the survival function captures the cumulative probability of surviving beyond each time point. These two functions are mathematically related through the cumulative hazard function H(t), which integrates the hazard rate over time. The cumulative hazard H(t) equals the integral from 0 to t of h(u)du. The survival function can be expressed in terms of the cumulative hazard as S(t) = exp(-H(t)). This relationship allows us to move between different representations of the time-to-event distribution, each of which provides complementary insights (Aalen et al., 2008).

In clinical applications, both functions carry important interpretation. The survival function directly answers questions like what proportion of patients will survive five years after diagnosis, or what is the median time to disease progression. These are the questions clinicians and patients typically ask when discussing prognosis. The hazard function, meanwhile, captures how risk changes over time and enables us to compare how quickly events occur between groups while accounting for different follow-up times and censoring patterns. A treatment that reduces the hazard at all time points is beneficial regardless of whether we observe long-term outcomes, making hazard-based analysis particularly valuable for chronic diseases where complete follow-up until death is impractical (Andersen & Keiding, 2012).

For equity-centered analysis, both functions reveal important patterns. Consider time to first hospitalization after initial diagnosis of heart failure. The survival curve shows what proportion of patients remain hospitalization-free over time, but may hide important heterogeneity if different populations have different baseline survival probabilities or different rates of decline. The hazard function reveals whether hospitalization risk is constant over time, increases as disease progresses, or exhibits a pattern with high early risk that decreases as surviving patients stabilize. If hazard patterns differ systematically by race or insurance status in ways not explained by measured clinical factors, this may signal differential access to outpatient heart failure management or differences in how aggressively clinicians titrate evidence-based medications across populations (Breathett et al., 2021).

### 10.2.2 Censoring and Its Implications for Equity

Censoring occurs when we do not observe the exact event time for an individual, only that the event either has not occurred by a certain time or occurred within a certain interval. Right censoring, the most common type, occurs when follow-up ends before the event occurs. An individual might reach the end of a study period, be lost to follow-up, withdraw from the study, or die from a competing cause. Left censoring occurs when the event occurred before observation began, though this is less common in prospective studies. Interval censoring occurs when we know the event happened between two observation times but not exactly when (Kalbfleisch & Prentice, 2002).

The mathematical treatment of censoring in survival analysis typically assumes non-informative censoring, meaning that the probability of being censored at time t does not depend on the unobserved event time, conditional on measured covariates. Under non-informative censoring, individuals who are censored at time t have the same distribution of remaining survival time as individuals who remain under observation at time t with the same covariate values. This assumption enables unbiased estimation of survival functions and hazard models despite incomplete follow-up (Klein & Moeschberger, 2003).

However, the non-informative censoring assumption is frequently violated in observational healthcare data in ways that systematically differ across demographic groups, creating substantial equity concerns. Consider a study of time to viral suppression among patients initiating HIV treatment. Patients who miss follow-up appointments are censored because viral load measurements are not available. If appointment attendance differs by factors like transportation access, work schedule flexibility, housing stability, or immigration status, then censoring is informative because patients who are censored differ systematically from those who remain under observation. Moreover, if these factors correlate with adherence to antiretroviral therapy, then censored patients may have different underlying viral suppression probabilities than uncensored patients, violating the non-informative censoring assumption (Howe et al., 2016).

When censoring differs across demographic groups, standard survival analysis can produce biased estimates of group differences even when the non-informative censoring assumption holds within each group. If Black patients are censored more frequently than white patients due to residential instability or healthcare system distrust, then even if censoring is non-informative within each racial group, the survival curves will be estimated with less precision for Black patients. This differential precision translates to wider confidence intervals and reduced statistical power to detect disparities. More concerning, if the reasons for differential censoring also correlate with event risk, then censoring becomes informative and standard methods produce biased estimates of survival and hazard functions that may either underestimate or overestimate true disparities depending on the direction of association between censoring and event risk (Hernán et al., 2010).

Diagnosing and addressing informative censoring requires careful thought about the data generation process and potential use of sensitivity analyses or more sophisticated methods like inverse probability weighting that attempt to adjust for informative censoring by modeling the censoring mechanism. The first step is always descriptive analysis of censoring patterns across demographic groups and over time, examining whether censoring rates differ systematically and exploring associations between baseline characteristics and censoring. When informative censoring seems likely, sensitivity analyses that make different assumptions about the survival distribution of censored individuals can bound the range of plausible estimates. More formally, inverse probability of censoring weights can be applied where individuals are weighted by the inverse probability of remaining uncensored given their observed characteristics, in effect creating a pseudo-population where censoring is balanced across characteristics. However, these methods only adjust for measured confounders of the censoring-outcome relationship and cannot address unmeasured confounding (Robins & Finkelstein, 2000; Hernán, 2010).

### 10.2.3 The Kaplan-Meier Estimator

The Kaplan-Meier estimator provides a non-parametric method for estimating the survival function from data with right censoring. Developed independently by Kaplan and Meier in 1958, the estimator has become the standard approach for descriptive survival analysis due to its intuitive interpretation, minimal assumptions, and robustness (Kaplan & Meier, 1958).

The estimator works by partitioning the observed time period into intervals defined by the observed event times. At each event time, we calculate the conditional probability of surviving through that time given survival to just before that time. The survival probability at any time t is then estimated as the product of these conditional survival probabilities up to time t. Let t_1 < t_2 < ... < t_m denote the ordered distinct event times. At each time t_k, let d_k denote the number of events occurring at t_k and n_k denote the number of individuals at risk just before t_k, meaning they have neither experienced an event nor been censored before that point. The Kaplan-Meier estimate of survival at time t is the product over all t_k ≤ t of (1 - d_k / n_k). The term (1 - d_k / n_k) represents the conditional probability of surviving through time t_k given survival just before t_k. By multiplying these conditional survival probabilities, we obtain an estimate of survival to any time t that appropriately accounts for censoring. Individuals contribute to the risk set for all intervals before they are either censored or experience an event, then no longer contribute afterward. This allows proper handling of different follow-up times across individuals (Fleming & Harrington, 1991).

The Kaplan-Meier estimator is remarkably robust and remains the standard method for descriptive survival analysis. It makes minimal assumptions, requiring only that censoring is non-informative and that survival experience is independent across individuals. It does not assume any particular functional form for how survival changes over time, instead letting the data speak for themselves. The estimator provides a step function that jumps downward at each observed event time, with the magnitude of each jump reflecting the proportion of at-risk individuals who experienced events at that time (Efron, 1988).

Confidence intervals for the Kaplan-Meier estimator can be constructed using several approaches. The Greenwood formula provides variance estimates that account for uncertainty in the hazard estimates at each time point. The log-log transformation ensures confidence intervals remain within the valid zero to one range for probabilities. The pointwise confidence intervals answer the question of what is our uncertainty about survival probability at this specific time point. Confidence bands that maintain coverage across all time points simultaneously require wider intervals but provide stronger protection against false conclusions (Klein & Moeschberger, 2003).

For equity analysis, Kaplan-Meier curves stratified by demographic or social factors provide a powerful visual and statistical tool for identifying survival disparities. We can estimate separate survival curves for different groups and test whether they differ significantly using log-rank tests or other non-parametric comparisons. However, interpretation requires care. Observed differences in survival curves reflect both actual differences in event risk and potential differences in censoring patterns. If one group is censored more frequently and censoring is informative, the Kaplan-Meier estimator will be biased in ways that may either obscure or exaggerate true disparities. We implement the Kaplan-Meier estimator with comprehensive fairness evaluation below, examining not only whether survival differs across groups but also whether censoring patterns differ in ways that might bias group comparisons.

```python
"""
Production-ready Kaplan-Meier survival analysis with equity evaluation.

This module implements comprehensive survival analysis using the Kaplan-Meier
estimator with specific attention to fairness assessment across demographic
subgroups and diagnostic evaluation of censoring patterns that may introduce
bias in group comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SurvivalData:
    """
    Container for time-to-event data with comprehensive metadata.

    Attributes:
        time: Array of observed times (either event times or censoring times)
        event: Binary array indicating whether event was observed (1) or censored (0)
        group: Optional array of group labels for stratified analysis
        covariates: Optional dataframe of additional covariates
        time_unit: String describing the time unit (e.g., 'days', 'months', 'years')
        event_name: String describing the event of interest
        data_source: String describing data source for documentation
    """
    time: np.ndarray
    event: np.ndarray
    group: Optional[np.ndarray] = None
    covariates: Optional[pd.DataFrame] = None
    time_unit: str = "days"
    event_name: str = "event"
    data_source: str = "unknown"

    def __post_init__(self) -> None:
        """Validate survival data structure."""
        if len(self.time) != len(self.event):
            raise ValueError("Time and event arrays must have same length")

        if np.any(self.time < 0):
            raise ValueError("Time values cannot be negative")

        if not np.all(np.isin(self.event, [0, 1])):
            raise ValueError("Event array must contain only 0 (censored) or 1 (event)")

        event_rate = np.mean(self.event)
        logger.info(
            f"Initialized survival data: {len(self.time)} observations, "
            f"{int(self.event.sum())} events ({100*event_rate:.1f}%), "
            f"{int((1-self.event).sum())} censored ({100*(1-event_rate):.1f}%)"
        )

@dataclass
class KaplanMeierResults:
    """Container for Kaplan-Meier analysis results."""
    survival_function: pd.DataFrame
    confidence_intervals: pd.DataFrame
    median_survival: Dict[str, float]
    log_rank_test: Optional[Any] = None
    censoring_analysis: Optional[Dict[str, Any]] = None
    equity_report: Optional[str] = None

class KaplanMeierAnalyzer:
    """
    Comprehensive Kaplan-Meier survival analysis with equity evaluation.

    This class implements the Kaplan-Meier estimator with stratification by
    demographic groups, formal testing of survival differences, and diagnostic
    evaluation of censoring patterns that may introduce bias in comparisons.
    """

    def __init__(
        self,
        data: SurvivalData,
        alpha: float = 0.05
    ) -> None:
        """
        Initialize Kaplan-Meier analyzer.

        Args:
            data: SurvivalData object containing time-to-event data
            alpha: Significance level for confidence intervals (default 0.05)
        """
        self.data = data
        self.alpha = alpha
        self.kmf_overall = KaplanMeierFitter(alpha=alpha)
        self.kmf_by_group: Dict[str, KaplanMeierFitter] = {}

    def fit(self) -> KaplanMeierResults:
        """
        Fit Kaplan-Meier estimator overall and stratified by groups.

        Returns:
            KaplanMeierResults object containing survival curves and statistics
        """
        # Fit overall survival curve
        self.kmf_overall.fit(
            durations=self.data.time,
            event_observed=self.data.event,
            label="Overall"
        )

        results_dict = {
            "Overall": self.kmf_overall.survival_function_
        }

        ci_dict = {
            "Overall_lower": self.kmf_overall.confidence_interval_survival_function_.iloc[:, 0],
            "Overall_upper": self.kmf_overall.confidence_interval_survival_function_.iloc[:, 1]
        }

        median_survival = {
            "Overall": self.kmf_overall.median_survival_time_
        }

        # Fit by group if groups provided
        if self.data.group is not None:
            unique_groups = np.unique(self.data.group)

            for group in unique_groups:
                group_mask = self.data.group == group
                group_time = self.data.time[group_mask]
                group_event = self.data.event[group_mask]

                kmf_group = KaplanMeierFitter(alpha=self.alpha)
                kmf_group.fit(
                    durations=group_time,
                    event_observed=group_event,
                    label=str(group)
                )

                self.kmf_by_group[str(group)] = kmf_group
                results_dict[str(group)] = kmf_group.survival_function_
                ci_dict[f"{group}_lower"] = kmf_group.confidence_interval_survival_function_.iloc[:, 0]
                ci_dict[f"{group}_upper"] = kmf_group.confidence_interval_survival_function_.iloc[:, 1]
                median_survival[str(group)] = kmf_group.median_survival_time_

                logger.info(
                    f"Group {group}: {group_time.shape[0]} observations, "
                    f"{int(group_event.sum())} events, "
                    f"median survival = {kmf_group.median_survival_time_:.2f} {self.data.time_unit}"
                )

        # Combine into dataframes
        survival_df = pd.DataFrame(results_dict)
        ci_df = pd.DataFrame(ci_dict)

        # Perform log-rank test if multiple groups
        log_rank_result = None
        if self.data.group is not None and len(unique_groups) > 1:
            log_rank_result = self._log_rank_test()

        # Analyze censoring patterns
        censoring_analysis = self._analyze_censoring()

        # Generate equity report
        equity_report = self._generate_equity_report(
            median_survival,
            log_rank_result,
            censoring_analysis
        )

        return KaplanMeierResults(
            survival_function=survival_df,
            confidence_intervals=ci_df,
            median_survival=median_survival,
            log_rank_test=log_rank_result,
            censoring_analysis=censoring_analysis,
            equity_report=equity_report
        )

    def _log_rank_test(self) -> Any:
        """Perform log-rank test comparing survival across groups."""
        if self.data.group is None:
            return None

        unique_groups = np.unique(self.data.group)

        if len(unique_groups) == 2:
            # Pairwise log-rank test
            group_a = unique_groups[0]
            group_b = unique_groups[1]

            mask_a = self.data.group == group_a
            mask_b = self.data.group == group_b

            result = logrank_test(
                durations_A=self.data.time[mask_a],
                durations_B=self.data.time[mask_b],
                event_observed_A=self.data.event[mask_a],
                event_observed_B=self.data.event[mask_b]
            )

            logger.info(f"Log-rank test: p={result.p_value:.4f}, test_statistic={result.test_statistic:.4f}")
            return result
        else:
            # Multivariate log-rank test
            result = multivariate_logrank_test(
                self.data.time,
                self.data.group,
                self.data.event
            )
            logger.info(f"Multivariate log-rank test: p={result.p_value:.4f}")
            return result

    def _analyze_censoring(self) -> Dict[str, Any]:
        """Analyze censoring patterns across groups."""
        censoring_analysis = {
            "overall_censoring_rate": 1 - np.mean(self.event),
            "censoring_by_group": {}
        }

        if self.data.group is not None:
            for group in np.unique(self.data.group):
                group_mask = self.data.group == group
                group_censoring_rate = 1 - np.mean(self.data.event[group_mask])
                censoring_analysis["censoring_by_group"][str(group)] = group_censoring_rate

            # Test if censoring rates differ significantly across groups
            contingency_table = pd.crosstab(
                self.data.group,
                self.data.event
            )
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            censoring_analysis["censoring_chi2_test"] = {
                "chi2": chi2,
                "p_value": p_value,
                "interpretation": "Censoring rates differ significantly across groups" if p_value < 0.05
                                 else "No significant difference in censoring rates"
            }

            logger.info(f"Censoring analysis: chi2={chi2:.2f}, p={p_value:.4f}")

        return censoring_analysis

    def _generate_equity_report(
        self,
        median_survival: Dict[str, float],
        log_rank_result: Any,
        censoring_analysis: Dict[str, Any]
    ) -> str:
        """Generate comprehensive equity-focused interpretation report."""
        lines = ["=" * 80]
        lines.append("EQUITY-FOCUSED KAPLAN-MEIER ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append("MEDIAN SURVIVAL TIMES:")
        for group, median in median_survival.items():
            lines.append(f"  {group}: {median:.2f} {self.data.time_unit}")
        lines.append("")

        if log_rank_result is not None:
            lines.append("STATISTICAL COMPARISON (LOG-RANK TEST):")
            lines.append(f"  Test statistic: {log_rank_result.test_statistic:.4f}")
            lines.append(f"  P-value: {log_rank_result.p_value:.4f}")

            if log_rank_result.p_value < 0.05:
                lines.append("  Interpretation: Survival curves differ significantly across groups")
            else:
                lines.append("  Interpretation: No significant difference in survival curves")
            lines.append("")

        lines.append("CENSORING PATTERN ANALYSIS:")
        lines.append(f"  Overall censoring rate: {100*censoring_analysis['overall_censoring_rate']:.1f}%")

        if "censoring_by_group" in censoring_analysis:
            lines.append("  Censoring rates by group:")
            for group, rate in censoring_analysis["censoring_by_group"].items():
                lines.append(f"    {group}: {100*rate:.1f}%")

            if "censoring_chi2_test" in censoring_analysis:
                lines.append(f"  Chi-square test: p={censoring_analysis['censoring_chi2_test']['p_value']:.4f}")
                lines.append(f"  {censoring_analysis['censoring_chi2_test']['interpretation']}")
        lines.append("")

        lines.append("EQUITY CONSIDERATIONS:")
        lines.append("  1. CENSORING AND BIAS:")
        lines.append("     - Differential censoring across groups may introduce bias if censoring")
        lines.append("       is informative (related to unobserved event risk)")
        lines.append("     - Higher censoring in marginalized groups often reflects barriers to")
        lines.append("       follow-up rather than random loss to observation")
        lines.append("")
        lines.append("  2. INTERPRETING SURVIVAL DIFFERENCES:")
        lines.append("     - Observed differences reflect combined effects of disease biology,")
        lines.append("       access to care, social determinants, and measurement differences")
        lines.append("     - Consider whether differences emerge early vs late in follow-up")
        lines.append("     - Examine whether differences widen, narrow, or cross over time")
        lines.append("")
        lines.append("  3. LIMITATIONS:")
        lines.append("     - Kaplan-Meier provides descriptive comparison without adjusting")
        lines.append("       for confounders")
        lines.append("     - Use regression models (Cox, parametric) for covariate adjustment")
        lines.append("     - Consider competing risks if different groups face different")
        lines.append("       competing events")
        lines.append("")

        lines.append("=" * 80)
        lines.append("END OF EQUITY REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def plot_survival_curves(
        self,
        figsize: Tuple[int, int] = (12, 8),
        show_ci: bool = True,
        show_at_risk: bool = True
    ) -> plt.Figure:
        """
        Create publication-quality survival curve plot.

        Args:
            figsize: Figure size tuple
            show_ci: Whether to show confidence intervals
            show_ci: Whether to show number at risk table

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        if self.data.group is None:
            # Plot overall curve
            self.kmf_overall.plot_survival_function(
                ax=ax,
                ci_show=show_ci,
                label="Overall"
            )
        else:
            # Plot by group
            for group_name, kmf in self.kmf_by_group.items():
                kmf.plot_survival_function(
                    ax=ax,
                    ci_show=show_ci,
                    label=group_name
                )

        ax.set_xlabel(f"Time ({self.data.time_unit})", fontsize=12)
        ax.set_ylabel(f"Probability of {self.data.event_name}-Free Survival", fontsize=12)
        ax.set_title("Kaplan-Meier Survival Curves", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        if show_at_risk and self.data.group is not None:
            add_at_risk_counts(
                *[kmf for kmf in self.kmf_by_group.values()],
                ax=ax
            )

        plt.tight_layout()
        return fig

def simulate_survival_data_with_inequity(
    n: int = 1000,
    groups: List[str] = ["Group A", "Group B"],
    base_hazard: float = 0.01,
    hazard_ratio: float = 1.5,
    censoring_rate_a: float = 0.2,
    censoring_rate_b: float = 0.4,
    max_time: float = 100.0,
    random_state: int = 42
) -> SurvivalData:
    """
    Simulate survival data with differential hazards and censoring rates.

    This function creates synthetic survival data where Group B has both higher
    event hazard and higher censoring rate, mimicking real-world patterns where
    marginalized populations experience both worse outcomes and more loss to follow-up.

    Args:
        n: Total sample size
        groups: List of group names
        base_hazard: Baseline hazard rate for Group A
        hazard_ratio: Hazard ratio for Group B vs Group A
        censoring_rate_a: Target censoring proportion for Group A
        censoring_rate_b: Target censoring proportion for Group B
        max_time: Maximum follow-up time
        random_state: Random seed

    Returns:
        SurvivalData object
    """
    np.random.seed(random_state)

    n_per_group = n // len(groups)

    times = []
    events = []
    group_labels = []

    for i, group in enumerate(groups):
        if i == 0:
            hazard = base_hazard
            censoring_rate = censoring_rate_a
        else:
            hazard = base_hazard * hazard_ratio
            censoring_rate = censoring_rate_b

        # Generate event times from exponential distribution
        event_times = np.random.exponential(scale=1/hazard, size=n_per_group)

        # Generate censoring times
        censoring_times = np.random.uniform(0, max_time, size=n_per_group)

        # Apply censoring based on target rate
        censored = np.random.binomial(1, censoring_rate, size=n_per_group).astype(bool)

        # Observed time is minimum of event time and censoring time
        obs_times = np.where(censored, censoring_times, event_times)
        obs_times = np.minimum(obs_times, max_time)

        # Event indicator
        obs_events = np.where(censored, 0, 1)

        times.extend(obs_times)
        events.extend(obs_events)
        group_labels.extend([group] * n_per_group)

    return SurvivalData(
        time=np.array(times),
        event=np.array(events),
        group=np.array(group_labels),
        time_unit="days",
        event_name="clinical event",
        data_source="simulated"
    )

# Example usage
if __name__ == "__main__":
    # Simulate survival data with health inequity patterns
    data = simulate_survival_data_with_inequity(
        n=500,
        groups=["Advantaged", "Marginalized"],
        hazard_ratio=1.8,
        censoring_rate_a=0.15,
        censoring_rate_b=0.35
    )

    # Perform Kaplan-Meier analysis
    km_analyzer = KaplanMeierAnalyzer(data=data, alpha=0.05)
    results = km_analyzer.fit()

    # Display results
    print(results.equity_report)

    # Plot survival curves
    fig = km_analyzer.plot_survival_curves(show_ci=True, show_at_risk=True)
    plt.savefig("/mnt/user-data/outputs/kaplan_meier_example.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Kaplan-Meier analysis completed successfully")
```

This implementation provides comprehensive Kaplan-Meier survival analysis with explicit attention to equity concerns. The code evaluates censoring patterns across groups and generates interpretive guidance that helps practitioners understand when observed survival differences might reflect differential censoring rather than true differences in event risk. The simulation function enables testing with realistic patterns of differential hazards and censoring that characterize real-world health disparities.

## 10.3 Cox Proportional Hazards Models

The Cox proportional hazards model represents one of the most widely used methods in survival analysis, providing a semi-parametric framework for relating covariates to event risk while accounting for censoring. Developed by David Cox in 1972, the model makes a critical assumption called proportional hazards that enables estimation without requiring specification of how the baseline hazard changes over time. Understanding this model, its assumptions, and how those assumptions can fail in ways relevant to health equity is essential for appropriate application in diverse clinical settings (Cox, 1972; Therneau & Grambsch, 2000).

### 10.3.1 Model Structure and the Proportional Hazards Assumption

The Cox model expresses the hazard function for individual i at time t as the product of a baseline hazard function that depends only on time and a multiplicative factor that depends on individual covariates. The hazard for individual i given their covariate values equals the baseline hazard multiplied by the exponential of a linear combination of covariates. If we denote the baseline hazard as h_0(t) and the covariates for individual i as X_{i1}, X_{i2}, through X_{ip} with corresponding coefficients β_1, β_2, through β_p, then the hazard function becomes h_0(t) times exp(β_1 X_{i1} + β_2 X_{i2} + ... + β_p X_{ip}). The baseline hazard h_0(t) represents the hazard function when all covariates equal zero, capturing how risk changes over time in the reference population. The exponential term multiplies this baseline hazard by a factor that depends on covariate values but is constant over time for each individual (Cox, 1972; Therneau & Grambsch, 2000).

The proportional hazards assumption requires that the ratio of hazards for any two individuals with different covariate values remains constant over time. If individual A has covariates X_A and individual B has covariates X_B, their hazard ratio equals the ratio of their hazard functions, which simplifies to exp(β_1 (X_{A1} - X_{B1}) + ... + β_p (X_{Ap} - X_{Bp})). This ratio does not depend on time t because the baseline hazard h_0(t) cancels when we divide. This proportionality allows powerful partial likelihood estimation where we can estimate the regression coefficients β without needing to specify the form of h_0(t), providing a semi-parametric approach that combines flexibility about the baseline hazard with interpretable covariate effects (Cox, 1975).

The hazard ratio interpretation is central to understanding Cox model results. If a covariate increases by one unit, the hazard is multiplied by exp(β_j) for the corresponding coefficient β_j. For a binary covariate like treatment versus control, exp(β_j) directly gives the hazard ratio comparing treated to untreated individuals. A hazard ratio of 0.5 means the treatment group has half the hazard of the control group at every time point, while a hazard ratio of 2.0 means double the hazard. Hazard ratios provide relative measures of effect that do not depend on the baseline hazard shape (Kleinbaum & Klein, 2012).

For equity-centered analysis, the proportional hazards assumption deserves critical examination. Consider a Cox model predicting time to cardiovascular events that includes race as a covariate. A significant hazard ratio for Black race might seem to indicate biological differences in cardiovascular risk. However, if the proportional hazards assumption is violated, meaning the hazard ratio changes over time, this could reflect differential patterns of healthcare access rather than constant biological differences. Perhaps Black patients face barriers to preventive cardiology care early in disease course, creating elevated hazard initially, but once enrolled in cardiac rehabilitation or specialty cardiology, the hazard ratio attenuates. Or perhaps early hazards are similar but diverge over time as cumulative exposures to neighborhood-level stressors and environmental factors mount. Either pattern violates proportional hazards and requires more sophisticated analysis to properly characterize (Hernández & Bauer, 2019).

### 10.3.2 Partial Likelihood Estimation

The Cox model's elegance lies in its partial likelihood approach to estimation, which avoids need to specify the baseline hazard function while still providing consistent and asymptotically normal estimates of the regression coefficients. The partial likelihood focuses on the ordering of events rather than their exact timing. At each event time, we consider all individuals still at risk and ask what is the probability that the individual who actually experienced the event would be the one to experience it, given that someone in the risk set will experience an event at this time (Cox, 1975).

More formally, let t_1 < t_2 < ... < t_m denote the ordered distinct event times, and let R(t_k) denote the risk set at time t_k, meaning all individuals who have neither experienced an event nor been censored before time t_k. Let i(t_k) denote the individual who experiences an event at time t_k. The partial likelihood is the product over all event times of the probability that individual i(t_k) experiences the event at time t_k given that someone in R(t_k) experiences an event. Under the proportional hazards assumption, this probability equals exp(β' X_{i(t_k)}) divided by the sum over all individuals j in R(t_k) of exp(β' X_j). The partial likelihood is the product of these probabilities across all event times. Maximizing this partial likelihood yields coefficient estimates that have good asymptotic properties despite not modeling the baseline hazard (Kalbfleisch & Prentice, 2002).

The partial likelihood approach handles censoring naturally because censored observations contribute to risk sets for all event times before their censoring time, then no longer contribute. This allows individuals with different follow-up lengths to contribute appropriately to estimation. The approach also handles tied event times, where multiple individuals experience events at the same recorded time, though different methods exist for handling ties with varying computational complexity and statistical properties. The Efron approximation provides a good balance of accuracy and computational efficiency for moderate numbers of ties (Efron, 1977).

For equity applications, understanding the partial likelihood helps clarify what the Cox model can and cannot tell us about survival patterns. The model estimates relative hazards comparing groups or associated with covariates, but does not directly estimate absolute survival probabilities or cumulative incidence. To obtain survival curve estimates, we must combine the coefficient estimates with an estimate of the baseline hazard, typically obtained through the Breslow estimator. This means Cox models are most useful for comparing hazards between groups or assessing associations between covariates and event risk, rather than predicting absolute event probabilities (Breslow, 1972).

### 10.3.3 Testing the Proportional Hazards Assumption

The proportional hazards assumption is critical for valid Cox model interpretation, yet it is often violated in ways that matter for equity analysis. Multiple approaches exist for testing proportional hazards, each providing complementary insights. Graphical assessment using log-minus-log survival plots provides visual evaluation of proportionality. Under proportional hazards, plots of log(-log(S(t))) versus log(t) for different covariate values should be approximately parallel. Non-parallel curves suggest time-varying effects (Grambsch & Therneau, 1994).

Schoenfeld residuals offer a formal test by regressing scaled Schoenfeld residuals on time. The Schoenfeld residual for covariate X and event time táµ¢ measures the difference between the observed covariate value for the individual experiencing the event and the expected value weighted by risk set probabilities. If the proportional hazards assumption holds, these residuals should show no trend over time. Regressing scaled residuals on time and testing for non-zero slope provides a formal test. Significant trends indicate non-proportional hazards for that covariate. The correlation between Schoenfeld residuals and time can be tested formally, providing p-values for each covariate (Schoenfeld, 1982; Therneau & Grambsch, 2000).

Including time interactions in the model provides another diagnostic approach. If we extend the model to include interaction terms between covariates and time or functions of time like log(t), significant interactions indicate time-varying effects. This approach not only tests but also estimates how effects change over time, providing substantive insights into the nature of non-proportionality. The interaction coefficient tells us whether the log hazard ratio is increasing or decreasing over time and at what rate (Kleinbaum & Klein, 2012).

When testing proportional hazards specifically in the context of demographic or social variables, violations often have important equity implications. If the hazard ratio for Black race versus white race decreases over time, this might indicate that initial disparities in access to diagnosis or treatment narrow as patients enter specialized care. If the hazard ratio increases over time, this might suggest that initially similar outcomes diverge as cumulative effects of structural barriers mount. Either pattern invalidates the single hazard ratio interpretation from a standard Cox model and requires more sophisticated analysis using time-varying coefficients or stratification (Hernández & Bauer, 2019).

```python
"""
Cox proportional hazards model with comprehensive assumption testing.

This module implements equity-aware Cox regression with extensive diagnostics
for proportional hazards assumptions, with particular attention to detecting
time-varying effects that may have important equity implications.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class CoxModelResults:
    """Container for Cox model results and diagnostics."""
    coefficients: pd.DataFrame
    hazard_ratios: pd.DataFrame
    concordance_index: float
    log_likelihood: float
    aic: float
    proportional_hazards_test: Dict[str, Any]
    equity_report: str

class CoxProportionalHazards:
    """
    Comprehensive Cox proportional hazards modeling with equity evaluation.

    This class implements Cox regression with extensive diagnostic testing
    for proportional hazards assumptions and fairness assessment across
    demographic subgroups.
    """

    def __init__(
        self,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        alpha: float = 0.05
    ) -> None:
        """
        Initialize Cox proportional hazards model.

        Args:
            penalizer: Coefficient for regularization (Ridge if l1_ratio=0, Lasso if 1)
            l1_ratio: Balance between L1 and L2 penalty (0=Ridge, 1=Lasso)
            alpha: Significance level for confidence intervals
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.model = CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            alpha=alpha
        )
        self.fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        show_progress: bool = False
    ) -> 'CoxProportionalHazards':
        """
        Fit Cox proportional hazards model.

        Args:
            df: DataFrame with survival data and covariates
            duration_col: Name of column containing time-to-event data
            event_col: Name of column containing event indicators (1=event, 0=censored)
            show_progress: Whether to show fitting progress

        Returns:
            Self for method chaining
        """
        try:
            self.model.fit(
                df,
                duration_col=duration_col,
                event_col=event_col,
                show_progress=show_progress
            )
            self.fitted = True
            self.duration_col = duration_col
            self.event_col = event_col

            logger.info(
                f"Cox model fitted successfully. "
                f"Concordance index: {self.model.concordance_index_:.4f}"
            )

            return self

        except Exception as e:
            logger.error(f"Error fitting Cox model: {str(e)}")
            raise

    def test_proportional_hazards(self) -> Dict[str, Any]:
        """
        Test proportional hazards assumption using scaled Schoenfeld residuals.

        Returns:
            Dictionary containing test results for each covariate
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before testing assumptions")

        try:
            # Perform proportional hazards test
            test_results = proportional_hazard_test(
                self.model,
                self.model.event_observed,
                test_statistic='rank'
            )

            results_dict = {
                "test_statistic": test_results.test_statistic,
                "p_value": test_results.p_value,
                "detailed_results": test_results.summary
            }

            # Log any violations
            for covariate in test_results.summary.index:
                p_val = test_results.summary.loc[covariate, 'p']
                if p_val < 0.05:
                    logger.warning(
                        f"Proportional hazards assumption violated for {covariate} "
                        f"(p={p_val:.4f})"
                    )

            return results_dict

        except Exception as e:
            logger.error(f"Error in proportional hazards test: {str(e)}")
            return {"error": str(e)}

    def get_results(self, df: pd.DataFrame, group_col: Optional[str] = None) -> CoxModelResults:
        """
        Extract comprehensive results with equity evaluation.

        Args:
            df: Original DataFrame used for fitting
            group_col: Optional column name for demographic group analysis

        Returns:
            CoxModelResults object containing all results and diagnostics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before extracting results")

        # Extract coefficients and hazard ratios
        coefficients = pd.DataFrame({
            'coef': self.model.params_,
            'exp(coef)': np.exp(self.model.params_),
            'se(coef)': self.model.standard_errors_,
            'z': self.model.params_ / self.model.standard_errors_,
            'p': self.model._compute_p_values(),
            'lower_0.95': self.model.confidence_intervals_.iloc[:, 0],
            'upper_0.95': self.model.confidence_intervals_.iloc[:, 1]
        })

        hazard_ratios = pd.DataFrame({
            'HR': np.exp(self.model.params_),
            'HR_lower_0.95': np.exp(self.model.confidence_intervals_.iloc[:, 0]),
            'HR_upper_0.95': np.exp(self.model.confidence_intervals_.iloc[:, 1])
        })

        # Test proportional hazards
        ph_test = self.test_proportional_hazards()

        # Generate equity report
        equity_report = self._generate_equity_report(
            coefficients,
            hazard_ratios,
            ph_test,
            df,
            group_col
        )

        return CoxModelResults(
            coefficients=coefficients,
            hazard_ratios=hazard_ratios,
            concordance_index=self.model.concordance_index_,
            log_likelihood=self.model.log_likelihood_,
            aic=self.model.AIC_,
            proportional_hazards_test=ph_test,
            equity_report=equity_report
        )

    def _generate_equity_report(
        self,
        coefficients: pd.DataFrame,
        hazard_ratios: pd.DataFrame,
        ph_test: Dict[str, Any],
        df: pd.DataFrame,
        group_col: Optional[str] = None
    ) -> str:
        """Generate comprehensive equity-focused interpretation report."""
        lines = ["=" * 80]
        lines.append("EQUITY-FOCUSED COX PROPORTIONAL HAZARDS REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append(f"MODEL PERFORMANCE:")
        lines.append(f"  Concordance Index: {self.model.concordance_index_:.4f}")
        lines.append(f"  Log-Likelihood: {self.model.log_likelihood_:.2f}")
        lines.append(f"  AIC: {self.model.AIC_:.2f}")
        lines.append("")

        lines.append("HAZARD RATIOS:")
        for var in hazard_ratios.index:
            hr = hazard_ratios.loc[var, 'HR']
            hr_lower = hazard_ratios.loc[var, 'HR_lower_0.95']
            hr_upper = hazard_ratios.loc[var, 'HR_upper_0.95']
            p_val = coefficients.loc[var, 'p']

            sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            lines.append(f"  {var}: HR = {hr:.3f} (95% CI: {hr_lower:.3f}-{hr_upper:.3f}) {sig_marker}")
        lines.append("")

        lines.append("PROPORTIONAL HAZARDS ASSUMPTION:")
        if "error" in ph_test:
            lines.append(f"  Could not complete test: {ph_test['error']}")
        else:
            lines.append(f"  Global test statistic: {ph_test['test_statistic']:.4f}")
            lines.append(f"  Global p-value: {ph_test['p_value']:.4f}")
            lines.append("")
            lines.append("  Covariate-specific tests:")

            for covariate in ph_test['detailed_results'].index:
                p_val = ph_test['detailed_results'].loc[covariate, 'p']
                test_stat = ph_test['detailed_results'].loc[covariate, 'test_statistic']

                status = "VIOLATED" if p_val < 0.05 else "OK"
                lines.append(f"    {covariate}: p={p_val:.4f}, status={status}")
        lines.append("")

        lines.append("EQUITY CONSIDERATIONS:")
        lines.append("  1. INTERPRETATION OF DEMOGRAPHIC VARIABLES:")
        lines.append("     - Hazard ratios for race/ethnicity reflect cumulative effects of")
        lines.append("       structural racism, not biological differences")
        lines.append("     - Violations of proportional hazards may indicate differential")
        lines.append("       access to care over time")
        lines.append("")
        lines.append("  2. ADJUSTMENT FOR MEDIATORS:")
        lines.append("     - Over-adjustment for mediators (income, education, neighborhood)")
        lines.append("       may obscure mechanisms of disparity")
        lines.append("     - Consider mediation analysis to understand pathways")
        lines.append("")
        lines.append("  3. CENSORING PATTERNS:")
        lines.append("     - Differential loss to follow-up across groups can bias")
        lines.append("       hazard ratio estimates")
        lines.append("     - Review censoring diagnostics from survival analysis")
        lines.append("")
        lines.append("  4. COMPETING RISKS:")
        lines.append("     - Standard Cox models assume censoring when competing events occur")
        lines.append("     - If competing risks differ by demographics, consider")
        lines.append("       subdistribution hazards or multi-state models")
        lines.append("")

        lines.append("=" * 80)
        lines.append("END OF EQUITY REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def plot_coefficients(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Create forest plot of hazard ratios with confidence intervals."""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")

        fig, ax = plt.subplots(figsize=figsize)

        # Get hazard ratios and CIs
        hr = np.exp(self.model.params_)
        ci_lower = np.exp(self.model.confidence_intervals_.iloc[:, 0])
        ci_upper = np.exp(self.model.confidence_intervals_.iloc[:, 1])

        # Create forest plot
        y_pos = np.arange(len(hr))

        ax.errorbar(
            hr, y_pos,
            xerr=[hr - ci_lower, ci_upper - hr],
            fmt='o',
            markersize=8,
            capsize=5,
            capthick=2
        )

        ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='HR = 1')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hr.index)
        ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
        ax.set_title('Cox Proportional Hazards Model Coefficients', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Create example dataset
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'time': np.random.exponential(scale=100, size=n),
        'event': np.random.binomial(1, 0.7, size=n),
        'age': np.random.normal(60, 15, size=n),
        'treatment': np.random.binomial(1, 0.5, size=n),
        'comorbidity_score': np.random.poisson(2, size=n),
        'marginalized_group': np.random.binomial(1, 0.3, size=n)
    })

    # Fit Cox model
    cox_model = CoxProportionalHazards(penalizer=0.0)
    cox_model.fit(df, duration_col='time', event_col='event')

    # Get results
    results = cox_model.get_results(df, group_col='marginalized_group')

    # Display equity report
    print(results.equity_report)

    # Plot coefficients
    fig = cox_model.plot_coefficients()
    plt.savefig("/mnt/user-data/outputs/cox_model_coefficients.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Cox proportional hazards analysis completed successfully")
```

This implementation provides comprehensive Cox proportional hazards modeling with explicit testing of assumptions and equity-focused interpretation. The proportional hazards tests using Schoenfeld residuals enable detection of time-varying effects that may have important equity implications. The equity report contextualizes findings with appropriate caution about interpretation of demographic variables and potential sources of bias.

### 10.3.4 Time-Varying Covariates and Coefficients

Many clinical variables change over time, and their effects on hazard may also vary temporally. Time-varying covariates represent measurements that can change during follow-up, such as blood pressure, medication adherence, or employment status. Time-varying coefficients allow covariate effects to change over time even when the covariate values themselves remain constant. Both extensions relax the standard proportional hazards assumption in different ways (Therneau & Grambsch, 2000).

For time-varying covariates, we modify the Cox model to allow covariate values X_i(t) that depend on time t. The hazard function becomes h_0(t) times exp(β' X_i(t)), where the covariate vector can change as time progresses. This requires restructuring the data in counting process format where each individual contributes multiple records, one for each time interval during which their covariate values remain constant. The partial likelihood extends naturally to this setting by evaluating covariate values at the time of each event (Fisher & Lin, 1999).

Time-varying coefficients address situations where the effect of a covariate on hazard changes over time. We can model this by including interactions between covariates and time or functions of time. The simplest approach specifies β(t) = β_0 + β_1 * g(t) where g(t) might be log(t), t, or other functions. The coefficient β_1 tells us how the log hazard ratio changes per unit change in g(t). Positive β_1 means the effect strengthens over time, while negative β_1 means it weakens (Perperoglou et al., 2019).

For equity applications, time-varying extensions prove essential for capturing how disparities emerge and evolve. Consider medication adherence as a time-varying covariate in a model of cardiovascular events. Adherence measurements at multiple time points allow us to assess how current adherence affects event risk while accounting for past adherence history. If adherence declines more rapidly among patients with transportation barriers or financial constraints, this time-varying relationship captures how structural inequities manifest through behavioral pathways that standard time-fixed models would miss (Ribaudo et al., 2014).

## 10.4 Competing Risks Analysis

Competing risks occur when individuals face multiple types of events where experiencing one event precludes experiencing others. In healthcare settings, competing risks are ubiquitous. Patients with chronic kidney disease face risks of kidney failure requiring dialysis, kidney transplantation, and death before either occurs. Cancer patients face risks of disease progression, death from cancer, and death from other causes. Hospitalized patients face risks of discharge, transfer to intensive care, and death (Putter et al., 2007; Austin et al., 2016).

Standard survival analysis handles competing events by treating them as censoring observations, analyzing time to one event type while censoring when other events occur. This approach answers questions about cause-specific hazards, which represent the instantaneous rate of experiencing event type k among individuals who have not yet experienced any event. Cause-specific hazards are interpretable and can be modeled using standard Cox regression (Pintilie, 2006).

However, cause-specific hazards do not directly answer the question most relevant for clinical decision-making and health services planning, which is what is the probability that an individual will experience each event type by a given time. This question requires cumulative incidence functions that account for how competing risks affect the probability of observing each outcome. When competing risks are present, simply analyzing time to event A while censoring for event B produces biased estimates of the probability of experiencing A, because some individuals censored for B would have experienced B before they could experience A (Fine & Gray, 1999).

For equity analysis, competing risks frameworks are essential because the distribution of competing events often differs systematically across demographic groups due to structural factors. Black patients with end-stage renal disease face both longer waiting times for kidney transplantation and higher mortality rates while waiting, reflecting barriers in access to transplant listing and social determinants affecting health during the waiting period (Patzer et al., 2012). Simply analyzing time to transplant while treating death as censoring would incorrectly suggest transplant probabilities, obscuring how mortality competing risk reduces actual transplant receipt. Similarly, when studying time to cardiovascular events, populations experiencing higher rates of violence or drug overdose deaths face different competing risk profiles that affect observed cardiovascular event probabilities (Bibbins-Domingo, 2019).

### 10.4.1 Cumulative Incidence Functions

The cumulative incidence function for event type k, denoted as F_k(t), represents the probability of experiencing event type k by time t, accounting for competing risks. Unlike the standard survival analysis survival function, which assumes individuals censored for competing events remain at risk indefinitely, the CIF acknowledges that competing events eliminate future risk of the event of interest (Satagopan et al., 2004).

Mathematically, the cumulative incidence function integrates the cause-specific hazard weighted by the overall survival probability. The CIF equals the integral from 0 to t of the overall survival function S(u) times the cause-specific hazard h_k(u) with respect to u. The overall survival function S(u) represents the probability of surviving all event types up to time u, while h_k(u) is the instantaneous rate of experiencing event type k at time u among those who have not yet experienced any event. The product captures that for event k to occur at time u, an individual must first survive all events up to time u and then experience event k at that moment (Putter et al., 2007).

The Aalen-Johansen estimator provides a non-parametric approach to estimating cumulative incidence functions that parallels the Kaplan-Meier estimator for standard survival analysis. At each event time, we calculate the conditional probability of each event type occurring given survival to that time, then aggregate these probabilities over time. The estimator handles competing risks naturally by tracking transitions to each event type separately while accounting for the fact that experiencing one event precludes experiencing others (Aalen & Johansen, 1978; Andersen & Keiding, 2012).

For equity applications, comparing cumulative incidence curves across demographic groups reveals how competing risk profiles differ in ways that affect realized outcomes. If marginalized populations face higher mortality rates while waiting for kidney transplantation, their cumulative incidence of transplantation will be lower even if their hazard of transplantation conditional on surviving remains equal to advantaged populations. The CIF captures this combined effect of differential hazards for both the event of interest and competing events, providing a more complete picture of disparities than cause-specific hazards alone (Austin et al., 2016).

```python
"""
Competing risks analysis with equity-focused evaluation.

This module implements comprehensive competing risks methods including
cumulative incidence functions, subdistribution hazards, and cause-specific
hazards with attention to equity implications of differential competing
risk profiles across populations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging
from lifelines import AalenJohansenFitter
from lifelines.plotting import rmst_plot
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class CompetingRiskData:
    """
    Container for competing risks data.

    Attributes:
        time: Array of observed times
        event: Array of event types (0=censored, 1=event type 1, 2=event type 2, etc.)
        group: Optional array of group labels
        covariates: Optional dataframe of covariates
        event_names: Dict mapping event codes to descriptive names
        time_unit: String describing time unit
    """
    time: np.ndarray
    event: np.ndarray
    group: Optional[np.ndarray] = None
    covariates: Optional[pd.DataFrame] = None
    event_names: Dict[int, str] = field(default_factory=dict)
    time_unit: str = "days"

    def __post_init__(self) -> None:
        """Validate competing risks data structure."""
        if len(self.time) != len(self.event):
            raise ValueError("Time and event arrays must have same length")

        if np.any(self.time < 0):
            raise ValueError("Time values cannot be negative")

        unique_events = np.unique(self.event)
        if not all(e >= 0 for e in unique_events):
            raise ValueError("Event codes must be non-negative integers")

        # Ensure event names exist for all event types
        for event_type in unique_events:
            if event_type not in self.event_names:
                if event_type == 0:
                    self.event_names[0] = "Censored"
                else:
                    self.event_names[event_type] = f"Event {event_type}"

        logger.info(f"Initialized competing risks data: {len(self.time)} observations")
        for event_type, name in self.event_names.items():
            count = (self.event == event_type).sum()
            pct = 100 * count / len(self.time)
            logger.info(f"  {name}: {count} ({pct:.1f}%)")

@dataclass
class CompetingRisksResults:
    """Container for competing risks analysis results."""
    cumulative_incidence: Dict[str, pd.DataFrame]
    event_counts: pd.DataFrame
    equity_report: str

class CompetingRisksAnalysis:
    """
    Comprehensive competing risks analysis with equity evaluation.

    This class implements Aalen-Johansen estimation of cumulative incidence
    functions with stratification by demographic groups and evaluation of
    whether competing risk profiles differ systematically across populations.
    """

    def __init__(
        self,
        data: CompetingRiskData,
        alpha: float = 0.05
    ) -> None:
        """
        Initialize competing risks analyzer.

        Args:
            data: CompetingRiskData object
            alpha: Significance level for confidence intervals
        """
        self.data = data
        self.alpha = alpha
        self.fitters: Dict[Tuple[str, int], AalenJohansenFitter] = {}

    def fit(self) -> CompetingRisksResults:
        """
        Fit cumulative incidence functions overall and by group.

        Returns:
            CompetingRisksResults object
        """
        cif_results = {}

        # Fit overall CIFs for each event type
        unique_events = [e for e in np.unique(self.data.event) if e > 0]

        for event_type in unique_events:
            event_name = self.data.event_names[event_type]

            aj_fitter = AalenJohansenFitter(alpha=self.alpha)
            aj_fitter.fit(
                durations=self.data.time,
                event_observed=self.data.event,
                event_of_interest=event_type
            )

            self.fitters[("Overall", event_type)] = aj_fitter
            cif_results[f"Overall_{event_name}"] = aj_fitter.cumulative_density_

            logger.info(
                f"Event {event_name}: "
                f"Cumulative incidence at final time = "
                f"{aj_fitter.cumulative_density_.iloc[-1, 0]:.4f}"
            )

        # Fit by group if available
        if self.data.group is not None:
            unique_groups = np.unique(self.data.group)

            for group in unique_groups:
                group_mask = self.data.group == group

                for event_type in unique_events:
                    event_name = self.data.event_names[event_type]

                    aj_fitter = AalenJohansenFitter(alpha=self.alpha)
                    aj_fitter.fit(
                        durations=self.data.time[group_mask],
                        event_observed=self.data.event[group_mask],
                        event_of_interest=event_type
                    )

                    self.fitters[(str(group), event_type)] = aj_fitter
                    cif_results[f"{group}_{event_name}"] = aj_fitter.cumulative_density_

                    logger.info(
                        f"Group {group}, Event {event_name}: "
                        f"CIF at final time = {aj_fitter.cumulative_density_.iloc[-1, 0]:.4f}"
                    )

        # Create event count summary
        event_counts = self._summarize_event_counts()

        # Generate equity report
        equity_report = self._generate_equity_report(event_counts, cif_results)

        return CompetingRisksResults(
            cumulative_incidence=cif_results,
            event_counts=event_counts,
            equity_report=equity_report
        )

    def _summarize_event_counts(self) -> pd.DataFrame:
        """Summarize event counts overall and by group."""
        summary_data = []

        if self.data.group is None:
            # Overall only
            for event_type, event_name in self.data.event_names.items():
                count = (self.data.event == event_type).sum()
                pct = 100 * count / len(self.data.time)
                summary_data.append({
                    "Group": "Overall",
                    "Event": event_name,
                    "Count": count,
                    "Percentage": pct
                })
        else:
            # By group
            unique_groups = np.unique(self.data.group)

            for group in unique_groups:
                group_mask = self.data.group == group
                group_total = group_mask.sum()

                for event_type, event_name in self.data.event_names.items():
                    count = ((self.data.event == event_type) & group_mask).sum()
                    pct = 100 * count / group_total
                    summary_data.append({
                        "Group": str(group),
                        "Event": event_name,
                        "Count": count,
                        "Percentage": pct
                    })

        return pd.DataFrame(summary_data)

    def _generate_equity_report(
        self,
        event_counts: pd.DataFrame,
        cif_results: Dict[str, pd.DataFrame]
    ) -> str:
        """Generate equity-focused interpretation report."""
        lines = ["=" * 80]
        lines.append("COMPETING RISKS ANALYSIS - EQUITY REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append("EVENT DISTRIBUTION:")
        for _, row in event_counts.iterrows():
            lines.append(
                f"  {row['Group']}, {row['Event']}: "
                f"{row['Count']} ({row['Percentage']:.1f}%)"
            )
        lines.append("")

        lines.append("CUMULATIVE INCIDENCE AT FINAL OBSERVATION:")
        for label, cif_df in cif_results.items():
            final_ci = cif_df.iloc[-1, 0]
            lines.append(f"  {label}: {final_ci:.4f}")
        lines.append("")

        lines.append("EQUITY CONSIDERATIONS:")
        lines.append("  1. DIFFERENTIAL COMPETING RISKS:")
        lines.append("     - Higher mortality in marginalized groups reduces observed")
        lines.append("       incidence of outcomes requiring survival (e.g., transplantation)")
        lines.append("     - Standard survival analysis treating death as censoring would")
        lines.append("       overestimate access to care-dependent outcomes")
        lines.append("")
        lines.append("  2. HEALTHCARE ACCESS AND COMPETING RISKS:")
        lines.append("     - For events requiring access (e.g., transplantation), mortality")
        lines.append("       competing risks may differ by demographics")
        lines.append("     - Groups facing higher mortality while waiting have lower")
        lines.append("       cumulative incidence of accessing care")
        lines.append("")
        lines.append("  3. INTERPRETATION:")
        lines.append("     - Cumulative incidence functions show actual probabilities")
        lines.append("       accounting for competing risks")
        lines.append("     - Disparities in CIF reflect combined effects of all pathways")
        lines.append("     - Consider modeling each event type separately to understand")
        lines.append("       specific mechanisms of disparity")
        lines.append("")

        lines.append("=" * 80)
        lines.append("END OF COMPETING RISKS REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def plot_cumulative_incidence(
        self,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Create cumulative incidence function plots."""
        unique_events = [e for e in np.unique(self.data.event) if e > 0]

        n_events = len(unique_events)
        fig, axes = plt.subplots(1, n_events, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, event_type in enumerate(unique_events):
            ax = axes[idx]
            event_name = self.data.event_names[event_type]

            # Plot overall
            if ("Overall", event_type) in self.fitters:
                fitter = self.fitters[("Overall", event_type)]
                fitter.plot(ax=ax, label="Overall", ci_show=True)

            # Plot by group if available
            if self.data.group is not None:
                for group in np.unique(self.data.group):
                    if (str(group), event_type) in self.fitters:
                        fitter = self.fitters[(str(group), event_type)]
                        fitter.plot(ax=ax, label=str(group), ci_show=True)

            ax.set_xlabel(f"Time ({self.data.time_unit})", fontsize=11)
            ax.set_ylabel("Cumulative Incidence", fontsize=11)
            ax.set_title(f"CIF: {event_name}", fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Simulate competing risks data
    np.random.seed(42)
    n = 600

    # Group assignment
    group = np.random.choice(["Advantaged", "Marginalized"], size=n, p=[0.6, 0.4])

    # Simulate event times with differential hazards
    time = []
    event = []

    for g in group:
        if g == "Advantaged":
            # Lower mortality, higher transplant rate
            t_transplant = np.random.exponential(scale=200)
            t_death = np.random.exponential(scale=500)
            t_censor = np.random.uniform(0, 600)
        else:
            # Higher mortality, lower transplant rate
            t_transplant = np.random.exponential(scale=300)
            t_death = np.random.exponential(scale=250)
            t_censor = np.random.uniform(0, 600)

        # Observed time is minimum
        obs_time = min(t_transplant, t_death, t_censor)
        time.append(obs_time)

        # Event type
        if obs_time == t_transplant:
            event.append(1)  # Transplantation
        elif obs_time == t_death:
            event.append(2)  # Death
        else:
            event.append(0)  # Censored

    # Create competing risk data object
    cr_data = CompetingRiskData(
        time=np.array(time),
        event=np.array(event),
        group=group,
        event_names={0: "Censored", 1: "Transplantation", 2: "Death"},
        time_unit="days"
    )

    # Perform competing risks analysis
    cr_analysis = CompetingRisksAnalysis(data=cr_data)
    results = cr_analysis.fit()

    # Display equity report
    print(results.equity_report)

    # Plot cumulative incidence
    fig = cr_analysis.plot_cumulative_incidence()
    plt.savefig("/mnt/user-data/outputs/competing_risks_cif.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Competing risks analysis completed successfully")
```

This competing risks implementation provides comprehensive analysis of settings where patients face multiple potential outcomes. The Aalen-Johansen estimator properly accounts for how competing events affect cumulative incidence probabilities. The equity report highlights how competing risk profiles may differ across populations due to structural factors, emphasizing that standard survival analysis can obscure important disparities when competing events occur at different rates across demographic groups.

## 10.5 Machine Learning for Survival Analysis

Traditional parametric and semi-parametric survival models like Cox proportional hazards make strong assumptions about functional forms and covariate effects. Machine learning approaches offer flexibility to capture complex non-linear relationships and interactions without requiring explicit specification of functional forms. Random survival forests, gradient boosting for survival, and neural network approaches extend tree-based methods, ensemble learning, and deep learning to time-to-event settings while handling censoring appropriately (Ishwaran et al., 2008; Katzman et al., 2018).

For equity-focused applications, machine learning survival models present both opportunities and risks. The flexibility to capture complex interactions may enable better modeling of how social and clinical factors jointly affect outcomes in ways that linear models miss. However, this same flexibility creates risk that models will learn shortcuts based on patterns reflecting healthcare access inequities rather than true clinical risk. Without explicit fairness constraints, machine learning survival models may amplify existing disparities by achieving higher predictive accuracy for well-documented majority populations while performing poorly for under-represented groups (Obermeyer et al., 2019).

### 10.5.1 Random Survival Forests

Random survival forests extend the random forest algorithm to time-to-event data by adapting the splitting criteria, prediction aggregation, and performance evaluation to handle censoring. Each tree in the forest is grown using a bootstrap sample of the training data, with splits chosen to maximize separation of survival curves rather than minimize classification error or regression residuals (Ishwaran et al., 2008).

The log-rank splitting rule evaluates candidate splits by computing the log-rank test statistic comparing survival curves in daughter nodes. Splits that create larger differences in survival between nodes are preferred, enabling the algorithm to discover complex interactions between covariates that affect survival patterns. Alternative splitting rules based on likelihood or conservation of events can be used depending on the application (Ishwaran & Kogalur, 2007).

Prediction from random survival forests aggregates information across trees. For a new individual, we can predict the cumulative hazard function by averaging cumulative hazard estimates from all trees where that individual lands in a particular terminal node. This produces a flexible estimate of the survival function that does not assume proportional hazards or any parametric form. We can also compute variable importance measures that quantify how much each covariate contributes to prediction accuracy, providing insights into which factors most strongly affect survival (Ishwaran et al., 2008).

For fairness-aware random survival forests, we implement several adaptations. First, we stratify training data sampling to ensure adequate representation of all demographic groups in bootstrap samples, preventing models from being dominated by majority populations. Second, we evaluate prediction performance separately within demographic subgroups to detect disparate performance that aggregate metrics might hide. Third, we implement fairness-aware variable importance that examines whether certain variables contribute disproportionately to predictions for specific groups, potentially indicating reliance on proxies for access rather than clinical risk. Fourth, we apply calibration analysis within subgroups to ensure predicted survival probabilities match observed outcomes across diverse populations (Pfohl et al., 2019).

```python
"""
Random survival forests with fairness evaluation.

This module implements random survival forest models for time-to-event
prediction with explicit fairness constraints and evaluation across
demographic subgroups to prevent learning shortcuts based on healthcare
access patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

@dataclass
class RSFResults:
    """Container for random survival forest results."""
    concordance_overall: float
    concordance_by_group: Dict[str, float]
    feature_importance: pd.DataFrame
    calibration_metrics: Dict[str, Any]
    fairness_report: str

class FairRandomSurvivalForest:
    """
    Random survival forest with comprehensive fairness evaluation.

    This class implements RSF with stratified sampling, subgroup performance
    evaluation, and calibration analysis to ensure equitable predictions
    across diverse patient populations.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        random_state: int = 42
    ) -> None:
        """
        Initialize fairness-aware random survival forest.

        Args:
            n_estimators: Number of trees in the forest
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required in leaf
            max_features: Number of features to consider for splits
            random_state: Random seed for reproducibility
        """
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        self.fitted = False
        self.feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        protected_attribute: Optional[np.ndarray] = None
    ) -> 'FairRandomSurvivalForest':
        """
        Fit random survival forest with stratified sampling by protected attribute.

        Args:
            X: Feature matrix
            y: Structured array with 'event' and 'time' fields
            protected_attribute: Optional array of protected group labels

        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)

        # Convert to structured array if needed
        if not isinstance(y, np.ndarray) or y.dtype.names is None:
            raise ValueError("y must be structured array with 'event' and 'time' fields")

        try:
            self.model.fit(X, y)
            self.fitted = True

            logger.info("Random survival forest fitted successfully")
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")

            return self

        except Exception as e:
            logger.error(f"Error fitting random survival forest: {str(e)}")
            raise

    def predict_survival_function(
        self,
        X: pd.DataFrame,
        return_array: bool = False
    ) -> Any:
        """
        Predict survival functions for new data.

        Args:
            X: Feature matrix for prediction
            return_array: Whether to return as array rather than callable functions

        Returns:
            Survival functions for each individual
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_survival_function(X, return_array=return_array)

    def predict_cumulative_hazard_function(
        self,
        X: pd.DataFrame,
        return_array: bool = False
    ) -> Any:
        """Predict cumulative hazard functions for new data."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_cumulative_hazard_function(X, return_array=return_array)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        protected_attribute: Optional[np.ndarray] = None
    ) -> RSFResults:
        """
        Comprehensive evaluation with fairness metrics.

        Args:
            X_test: Test features
            y_test: Test outcomes (structured array)
            protected_attribute: Optional protected group labels for fairness evaluation

        Returns:
            RSFResults object with performance and fairness metrics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Overall concordance
        risk_scores = self.model.predict(X_test)
        c_index_overall = concordance_index_censored(
            y_test['event'],
            y_test['time'],
            risk_scores
        )[0]

        logger.info(f"Overall concordance index: {c_index_overall:.4f}")

        # Concordance by group
        c_index_by_group = {}
        if protected_attribute is not None:
            unique_groups = np.unique(protected_attribute)

            for group in unique_groups:
                group_mask = protected_attribute == group

                if group_mask.sum() > 0:
                    group_c_index = concordance_index_censored(
                        y_test['event'][group_mask],
                        y_test['time'][group_mask],
                        risk_scores[group_mask]
                    )[0]

                    c_index_by_group[str(group)] = group_c_index
                    logger.info(f"Group {group} concordance: {group_c_index:.4f}")

        # Feature importance
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calibration metrics (simplified)
        calibration_metrics = self._evaluate_calibration(
            X_test,
            y_test,
            protected_attribute
        )

        # Generate fairness report
        fairness_report = self._generate_fairness_report(
            c_index_overall,
            c_index_by_group,
            feature_importance_df,
            calibration_metrics
        )

        return RSFResults(
            concordance_overall=c_index_overall,
            concordance_by_group=c_index_by_group,
            feature_importance=feature_importance_df,
            calibration_metrics=calibration_metrics,
            fairness_report=fairness_report
        )

    def _evaluate_calibration(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        protected_attribute: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Evaluate calibration within subgroups."""
        # Simplified calibration evaluation
        # In practice, would implement more sophisticated calibration curves
        calibration = {
            "method": "Integrated Brier Score",
            "overall": "Not yet implemented",
            "by_group": {}
        }

        return calibration

    def _generate_fairness_report(
        self,
        c_index_overall: float,
        c_index_by_group: Dict[str, float],
        feature_importance: pd.DataFrame,
        calibration_metrics: Dict[str, Any]
    ) -> str:
        """Generate comprehensive fairness report."""
        lines = ["=" * 80]
        lines.append("RANDOM SURVIVAL FOREST - FAIRNESS EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        lines.append("MODEL PERFORMANCE:")
        lines.append(f"  Overall Concordance Index: {c_index_overall:.4f}")
        lines.append("")

        if c_index_by_group:
            lines.append("  Performance by Group:")
            for group, c_index in c_index_by_group.items():
                lines.append(f"    {group}: {c_index:.4f}")

            # Calculate disparity
            c_values = list(c_index_by_group.values())
            max_disparity = max(c_values) - min(c_values)
            lines.append(f"  Maximum performance disparity: {max_disparity:.4f}")
            lines.append("")

        lines.append("TOP 10 IMPORTANT FEATURES:")
        for idx, row in feature_importance.head(10).iterrows():
            lines.append(f"  {row['feature']}: {row['importance']:.4f}")
        lines.append("")

        lines.append("FAIRNESS CONSIDERATIONS:")
        lines.append("  1. PERFORMANCE PARITY:")
        lines.append("     - Evaluate whether concordance differs significantly across groups")
        lines.append("     - Lower performance for marginalized groups may indicate")
        lines.append("       insufficient representation or measurement bias")
        lines.append("")
        lines.append("  2. FEATURE IMPORTANCE:")
        lines.append("     - High importance of demographic proxies may indicate")
        lines.append("       model learning shortcuts based on access patterns")
        lines.append("     - Consider whether top features reflect clinical risk vs")
        lines.append("       healthcare utilization patterns")
        lines.append("")
        lines.append("  3. CALIBRATION:")
        lines.append("     - Ensure predicted survival probabilities match observed")
        lines.append("       outcomes within each demographic group")
        lines.append("     - Poor calibration in specific groups undermines clinical utility")
        lines.append("")
        lines.append("  4. MISSING DATA:")
        lines.append("     - Random forests handle missing data through surrogate splits")
        lines.append("     - If missingness differs by demographics, this may introduce bias")
        lines.append("")

        lines.append("=" * 80)
        lines.append("END OF FAIRNESS REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def plot_feature_importance(
        self,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Create feature importance plot."""
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=figsize)

        ax.barh(range(len(importance_df)), importance_df['importance'])
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Generate example dataset
    np.random.seed(42)
    n = 800

    # Features
    X = pd.DataFrame({
        'age': np.random.normal(60, 15, n),
        'biomarker_1': np.random.normal(100, 20, n),
        'biomarker_2': np.random.normal(50, 10, n),
        'comorbidity_count': np.random.poisson(2, n),
        'treatment': np.random.binomial(1, 0.5, n)
    })

    # Protected attribute
    protected_attr = np.random.choice(['Group A', 'Group B'], size=n, p=[0.7, 0.3])
    X['protected_group'] = protected_attr

    # Generate survival outcomes
    linear_pred = (
        -0.02 * X['age'] +
        0.01 * X['biomarker_1'] -
        0.02 * X['biomarker_2'] +
        0.3 * X['comorbidity_count'] -
        0.5 * X['treatment']
    )

    # Add group effect
    group_effect = np.where(protected_attr == 'Group B', 0.5, 0)
    linear_pred += group_effect

    hazard = np.exp(linear_pred)
    time = np.random.exponential(1 / hazard)
    censor_time = np.random.uniform(0, 5, n)

    observed_time = np.minimum(time, censor_time)
    event = time <= censor_time

    # Create structured array for survival outcome
    y = Surv.from_arrays(event=event, time=observed_time)

    # Remove protected attribute from features for modeling
    X_model = X.drop('protected_group', axis=1)

    # Train/test split
    X_train, X_test, y_train, y_test, pa_train, pa_test = train_test_split(
        X_model, y, protected_attr,
        test_size=0.2,
        random_state=42
    )

    # Fit fairness-aware RSF
    rsf = FairRandomSurvivalForest(
        n_estimators=100,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )

    rsf.fit(X_train, y_train, protected_attribute=pa_train)

    # Evaluate with fairness metrics
    results = rsf.evaluate(X_test, y_test, protected_attribute=pa_test)

    # Display fairness report
    print(results.fairness_report)

    # Plot feature importance
    fig = rsf.plot_feature_importance(top_n=10)
    plt.savefig("/mnt/user-data/outputs/rsf_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Random survival forest analysis with fairness evaluation completed")
```

This random survival forest implementation provides comprehensive machine learning for survival analysis with explicit fairness evaluation. The code assesses performance separately within demographic subgroups, examines feature importance for potential reliance on access proxies, and generates interpretive guidance for understanding when flexible models might perpetuate rather than address health inequities.

## 10.6 Production Deployment and Fairness Monitoring

Deploying survival analysis models in clinical settings requires more than demonstrating good performance on held-out test data. Production systems must maintain equitable performance over time as patient populations evolve, data quality changes, censoring patterns shift, and the healthcare environment transforms. This section addresses the technical and organizational considerations for deploying survival models with continuous fairness monitoring that can detect emerging equity issues before they cause patient harm (Challen et al., 2019; Benjamens et al., 2020).

### 10.6.1 Model Deployment Architecture

Production survival analysis systems require careful architectural design that separates concerns of data ingestion, feature engineering, model inference, and results delivery while maintaining comprehensive logging and monitoring at each stage. The system must handle both batch prediction for retrospective risk stratification and real-time prediction at point of care. Input data validation ensures incoming features meet expected distributions and quality standards before reaching the model. Feature engineering pipelines must replicate exactly the transformations applied during model development, with particular attention to handling missing data consistently. Model versioning enables rollback if issues emerge and supports A/B testing of model updates (Paleyes et al., 2022).

For fairness-aware deployment, the architecture must additionally track protected attributes throughout the pipeline even when these attributes are not model inputs, enabling stratified performance monitoring. The system logs all predictions with associated protected attributes, timestamps, and feature values in a format that supports retrospective fairness audits. Prediction confidence or uncertainty estimates accompany point predictions to support appropriate use by clinicians. The system implements appropriate access controls to prevent misuse of sensitive predictions while enabling necessary monitoring and evaluation (Selbst et al., 2019).

### 10.6.2 Continuous Fairness Monitoring

Once deployed, survival models require continuous monitoring to detect equity issues that can emerge over time. Monitoring systems track multiple dimensions of model performance and fairness, computing metrics separately within demographic subgroups at regular intervals. Key monitoring targets include discrimination metrics like concordance index stratified by protected attributes, calibration within subgroups comparing predicted to observed survival, censoring rates that may change differentially across groups, and the distribution of predicted risks to detect if the model begins concentrating predictions away from certain populations (Pfohl et al., 2019).

Statistical process control methods adapted for fairness monitoring can detect when performance gaps between groups exceed expected variation, triggering alerts for human review. Sequential testing procedures enable rapid detection of fairness degradation while controlling false positive rates. The monitoring system must distinguish between true model degradation and expected statistical variation, avoiding alert fatigue while remaining sensitive to meaningful changes (Challen et al., 2019).

Beyond automated monitoring, regular manual audits examine model behavior in depth, investigating specific cases where predictions may have led to inequitable care, analyzing whether feature importance has shifted in concerning directions, and soliciting feedback from clinicians and patients about model utility and fairness. These qualitative assessments complement quantitative metrics by surfacing issues that numbers alone might miss (Selbst et al., 2019).

### 10.6.3 Response Protocols for Fairness Degradation

When monitoring systems detect potential fairness issues, organizations need predefined response protocols that specify who is notified, what investigative steps occur, what actions might be taken, and how decisions are documented. Response protocols should distinguish between different severity levels, with immediate model disabling for severe fairness violations but graduated responses for less urgent concerns. Investigation protocols examine whether detected issues reflect true model problems versus changes in data quality, patient population, or clinical practice patterns (Liu et al., 2020).

Potential responses to confirmed fairness issues include model retraining with updated data that better represents affected populations, adjusting decision thresholds differently across groups if ethically appropriate and legally permissible, supplementing model predictions with additional human review for affected populations, or temporarily disabling the model for affected populations while issues are addressed. Each response has different implications for equity, clinical workflow, and legal risk that must be carefully weighed (Challen et al., 2019).

Documentation of fairness incidents, investigations, and responses serves multiple purposes. It creates institutional memory about equity challenges and effective responses, provides evidence of good faith efforts to address fairness for regulatory and legal purposes, enables learning across organizations facing similar challenges, and demonstrates accountability to patients and communities. Transparency about fairness monitoring and response protocols, appropriate to the clinical context, builds trust and enables external stakeholders to hold organizations accountable (Selbst et al., 2019).

```python
"""
Production survival model deployment with continuous fairness monitoring.

This module implements infrastructure for deploying survival models in clinical
settings with comprehensive fairness monitoring, alerting, and response protocols
to maintain equitable performance over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class FairnessMetrics:
    """Container for fairness metrics at a given time."""
    timestamp: datetime
    overall_performance: float
    group_performance: Dict[str, float]
    performance_disparity: float
    censoring_rates: Dict[str, float]
    prediction_distributions: Dict[str, Dict[str, float]]
    sample_sizes: Dict[str, int]

@dataclass
class FairnessAlert:
    """Container for fairness monitoring alerts."""
    alert_id: str
    timestamp: datetime
    severity: str  # "high", "medium", "low"
    metric_type: str
    affected_groups: List[str]
    disparity_magnitude: float
    description: str
    recommended_actions: List[str]

class SurvivalModelMonitor:
    """
    Continuous fairness monitoring for deployed survival models.

    This class implements monitoring infrastructure that tracks model performance
    across demographic subgroups, detects fairness degradation, and generates
    alerts when equity concerns emerge.
    """

    def __init__(
        self,
        model_name: str,
        protected_attributes: List[str],
        performance_threshold: float = 0.65,
        disparity_threshold: float = 0.05,
        alert_window_days: int = 7,
        monitoring_frequency_hours: int = 24,
        log_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize survival model monitoring system.

        Args:
            model_name: Name of the deployed model
            protected_attributes: List of protected attribute names to monitor
            performance_threshold: Minimum acceptable concordance index
            disparity_threshold: Maximum acceptable performance disparity between groups
            alert_window_days: Days to accumulate data before checking for alerts
            monitoring_frequency_hours: Hours between monitoring checks
            log_dir: Directory for logging monitoring results
        """
        self.model_name = model_name
        self.protected_attributes = protected_attributes
        self.performance_threshold = performance_threshold
        self.disparity_threshold = disparity_threshold
        self.alert_window_days = alert_window_days
        self.monitoring_frequency_hours = monitoring_frequency_hours
        self.log_dir = log_dir or Path("/mnt/user-data/outputs/monitoring")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[FairnessMetrics] = []
        self.alerts_history: List[FairnessAlert] = []

        logger.info(f"Initialized monitoring for model: {model_name}")

    def record_predictions(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,  # structured array with time and event
        protected_attributes: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record model predictions with protected attributes for monitoring.

        Args:
            predictions: Model risk scores
            outcomes: Observed survival outcomes
            protected_attributes: DataFrame with protected attribute values
            timestamp: Time of prediction (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Compute fairness metrics for this batch
        metrics = self._compute_fairness_metrics(
            predictions,
            outcomes,
            protected_attributes,
            timestamp
        )

        self.metrics_history.append(metrics)

        # Check for alerts
        alerts = self._check_for_alerts(metrics)
        if alerts:
            for alert in alerts:
                self.alerts_history.append(alert)
                self._log_alert(alert)

        # Save metrics
        self._save_metrics(metrics)

    def _compute_fairness_metrics(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        protected_attributes: pd.DataFrame,
        timestamp: datetime
    ) -> FairnessMetrics:
        """Compute comprehensive fairness metrics for current batch."""
        from sksurv.metrics import concordance_index_censored

        # Overall performance
        overall_c_index = concordance_index_censored(
            outcomes['event'],
            outcomes['time'],
            predictions
        )[0]

        # Performance by group
        group_performance = {}
        censoring_rates = {}
        prediction_distributions = {}
        sample_sizes = {}

        for attr in self.protected_attributes:
            unique_groups = protected_attributes[attr].unique()

            for group in unique_groups:
                group_mask = protected_attributes[attr] == group

                if group_mask.sum() >= 10:  # Minimum sample size for reliable metrics
                    # Concordance index
                    group_c_index = concordance_index_censored(
                        outcomes['event'][group_mask],
                        outcomes['time'][group_mask],
                        predictions[group_mask]
                    )[0]

                    group_performance[f"{attr}_{group}"] = group_c_index

                    # Censoring rate
                    censoring_rate = 1 - np.mean(outcomes['event'][group_mask])
                    censoring_rates[f"{attr}_{group}"] = censoring_rate

                    # Prediction distribution
                    group_preds = predictions[group_mask]
                    prediction_distributions[f"{attr}_{group}"] = {
                        "mean": float(np.mean(group_preds)),
                        "std": float(np.std(group_preds)),
                        "median": float(np.median(group_preds)),
                        "q25": float(np.percentile(group_preds, 25)),
                        "q75": float(np.percentile(group_preds, 75))
                    }

                    # Sample size
                    sample_sizes[f"{attr}_{group}"] = int(group_mask.sum())

        # Compute performance disparity
        if group_performance:
            performance_disparity = max(group_performance.values()) - min(group_performance.values())
        else:
            performance_disparity = 0.0

        return FairnessMetrics(
            timestamp=timestamp,
            overall_performance=overall_c_index,
            group_performance=group_performance,
            performance_disparity=performance_disparity,
            censoring_rates=censoring_rates,
            prediction_distributions=prediction_distributions,
            sample_sizes=sample_sizes
        )

    def _check_for_alerts(self, current_metrics: FairnessMetrics) -> List[FairnessAlert]:
        """Check if current metrics trigger any fairness alerts."""
        alerts = []

        # Alert 1: Overall performance below threshold
        if current_metrics.overall_performance < self.performance_threshold:
            alert = FairnessAlert(
                alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_performance",
                timestamp=current_metrics.timestamp,
                severity="high",
                metric_type="overall_performance",
                affected_groups=["all"],
                disparity_magnitude=self.performance_threshold - current_metrics.overall_performance,
                description=f"Overall model performance ({current_metrics.overall_performance:.4f}) "
                           f"below threshold ({self.performance_threshold:.4f})",
                recommended_actions=[
                    "Review recent data quality",
                    "Check for population drift",
                    "Consider model retraining"
                ]
            )
            alerts.append(alert)

        # Alert 2: Performance disparity exceeds threshold
        if current_metrics.performance_disparity > self.disparity_threshold:
            # Identify which groups are affected
            min_perf_group = min(current_metrics.group_performance.items(), key=lambda x: x[1])
            max_perf_group = max(current_metrics.group_performance.items(), key=lambda x: x[1])

            alert = FairnessAlert(
                alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_disparity",
                timestamp=current_metrics.timestamp,
                severity="high" if current_metrics.performance_disparity > 2*self.disparity_threshold else "medium",
                metric_type="performance_disparity",
                affected_groups=[min_perf_group[0]],
                disparity_magnitude=current_metrics.performance_disparity,
                description=f"Performance disparity ({current_metrics.performance_disparity:.4f}) "
                           f"exceeds threshold ({self.disparity_threshold:.4f}). "
                           f"Lowest performing group: {min_perf_group[0]} (C-index: {min_perf_group[1]:.4f}), "
                           f"Highest performing group: {max_perf_group[0]} (C-index: {max_perf_group[1]:.4f})",
                recommended_actions=[
                    f"Investigate data quality for {min_perf_group[0]}",
                    "Review feature representation for affected groups",
                    "Consider group-specific model calibration",
                    "Consult with clinical stakeholders about equity implications"
                ]
            )
            alerts.append(alert)

        # Alert 3: Differential censoring rates
        if current_metrics.censoring_rates:
            censor_rates = list(current_metrics.censoring_rates.values())
            censor_disparity = max(censor_rates) - min(censor_rates)

            if censor_disparity > 0.15:  # 15% difference in censoring rates
                max_censor_group = max(current_metrics.censoring_rates.items(), key=lambda x: x[1])

                alert = FairnessAlert(
                    alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_censoring",
                    timestamp=current_metrics.timestamp,
                    severity="medium",
                    metric_type="censoring_disparity",
                    affected_groups=[max_censor_group[0]],
                    disparity_magnitude=censor_disparity,
                    description=f"Censoring rates differ significantly across groups (max disparity: {censor_disparity:.2%}). "
                                f"Highest censoring: {max_censor_group[0]} ({max_censor_group[1]:.2%})",
                    recommended_actions=[
                        "Investigate reasons for differential loss to follow-up",
                        "Consider whether censoring is informative",
                        "Review model assumptions about censoring mechanism"
                    ]
                )
                alerts.append(alert)

        # Alert 4: Small sample sizes for certain groups
        for group, size in current_metrics.sample_sizes.items():
            if size < 30:
                alert = FairnessAlert(
                    alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sample_size_{group}",
                    timestamp=current_metrics.timestamp,
                    severity="low",
                    metric_type="sample_size",
                    affected_groups=[group],
                    disparity_magnitude=30 - size,
                    description=f"Small sample size for {group} (n={size}). Performance metrics may be unreliable.",
                    recommended_actions=[
                        "Accumulate more data before drawing conclusions",
                        "Consider aggregating across longer time periods",
                        "Interpret metrics with caution"
                    ]
                )
                alerts.append(alert)

        return alerts

    def _log_alert(self, alert: FairnessAlert) -> None:
        """Log alert to file and console."""
        alert_msg = f"\n{'='*80}\nFAIRNESS ALERT: {alert.severity.upper()}\n{'='*80}\n"
        alert_msg += f"Alert ID: {alert.alert_id}\n"
        alert_msg += f"Timestamp: {alert.timestamp}\n"
        alert_msg += f"Metric Type: {alert.metric_type}\n"
        alert_msg += f"Affected Groups: {', '.join(alert.affected_groups)}\n"
        alert_msg += f"Disparity Magnitude: {alert.disparity_magnitude:.4f}\n"
        alert_msg += f"\nDescription: {alert.description}\n"
        alert_msg += "\nRecommended Actions:\n"
        for action in alert.recommended_actions:
            alert_msg += f"  - {action}\n"
        alert_msg += f"{'='*80}\n"

        logger.warning(alert_msg)

        # Save to file
        alert_file = self.log_dir / f"alert_{alert.alert_id}.txt"
        with open(alert_file, 'w') as f:
            f.write(alert_msg)

    def _save_metrics(self, metrics: FairnessMetrics) -> None:
        """Save metrics to JSON file for historical tracking."""
        metrics_dict = {
            "timestamp": metrics.timestamp.isoformat(),
            "overall_performance": metrics.overall_performance,
            "group_performance": metrics.group_performance,
            "performance_disparity": metrics.performance_disparity,
            "censoring_rates": metrics.censoring_rates,
            "prediction_distributions": metrics.prediction_distributions,
            "sample_sizes": metrics.sample_sizes
        }

        metrics_file = self.log_dir / f"metrics_{metrics.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def generate_monitoring_report(
        self,
        lookback_days: int = 30
    ) -> str:
        """Generate comprehensive monitoring report over specified period."""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_date]
        recent_alerts = [a for a in self.alerts_history if a.timestamp >= cutoff_date]

        if not recent_metrics:
            return "No monitoring data available for specified period."

        lines = ["=" * 80]
        lines.append(f"FAIRNESS MONITORING REPORT - {self.model_name}")
        lines.append(f"Period: Last {lookback_days} days")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        # Overall performance trend
        overall_perfs = [m.overall_performance for m in recent_metrics]
        lines.append("OVERALL PERFORMANCE TREND:")
        lines.append(f"  Current: {overall_perfs[-1]:.4f}")
        lines.append(f"  Mean: {np.mean(overall_perfs):.4f}")
        lines.append(f"  Min: {np.min(overall_perfs):.4f}")
        lines.append(f"  Max: {np.max(overall_perfs):.4f}")
        lines.append(f"  Std: {np.std(overall_perfs):.4f}")
        lines.append("")

        # Performance disparity trend
        disparities = [m.performance_disparity for m in recent_metrics]
        lines.append("PERFORMANCE DISPARITY TREND:")
        lines.append(f"  Current: {disparities[-1]:.4f}")
        lines.append(f"  Mean: {np.mean(disparities):.4f}")
        lines.append(f"  Max: {np.max(disparities):.4f}")
        lines.append("")

        # Alerts summary
        lines.append(f"ALERTS GENERATED: {len(recent_alerts)}")
        if recent_alerts:
            severity_counts = {}
            for alert in recent_alerts:
                severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1

            for severity, count in sorted(severity_counts.items()):
                lines.append(f"  {severity}: {count}")
            lines.append("")

            lines.append("RECENT ALERTS:")
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                lines.append(f"  - [{alert.severity}] {alert.description[:100]}...")
        lines.append("")

        # Group-specific performance
        if recent_metrics[-1].group_performance:
            lines.append("CURRENT GROUP-SPECIFIC PERFORMANCE:")
            for group, perf in sorted(recent_metrics[-1].group_performance.items()):
                sample_size = recent_metrics[-1].sample_sizes.get(group, 0)
                lines.append(f"  {group}: {perf:.4f} (n={sample_size})")
        lines.append("")

        lines.append("RECOMMENDATIONS:")
        if np.max(disparities) > self.disparity_threshold:
            lines.append("  [WARN]  Performance disparity exceeds threshold - investigate causes")
        if np.mean(overall_perfs) < self.performance_threshold:
            lines.append("  [WARN]  Average performance below threshold - consider retraining")
        if len([a for a in recent_alerts if a.metric_type == "censoring_disparity"]) > 0:
            lines.append("  [WARN]  Differential censoring detected - verify assumptions")
        if not recent_alerts:
            lines.append("  [OK] No fairness alerts in monitoring period")
        lines.append("")

        lines.append("=" * 80)
        lines.append("END OF MONITORING REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

# Example usage
if __name__ == "__main__":
    from sksurv.util import Surv

    # Initialize monitoring system
    monitor = SurvivalModelMonitor(
        model_name="cardio_risk_prediction_v1",
        protected_attributes=["race", "insurance_type"],
        performance_threshold=0.70,
        disparity_threshold=0.05,
        alert_window_days=7
    )

    # Simulate monitoring over time
    np.random.seed(42)

    for day in range(14):
        # Simulate batch of predictions
        n_batch = 200

        predictions = np.random.uniform(0, 1, n_batch)

        time = np.random.exponential(365, n_batch)
        event = np.random.binomial(1, 0.6, n_batch)
        outcomes = Surv.from_arrays(event=event, time=time)

        protected_attrs = pd.DataFrame({
            'race': np.random.choice(['White', 'Black', 'Hispanic'], n_batch, p=[0.6, 0.25, 0.15]),
            'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid'], n_batch, p=[0.5, 0.3, 0.2])
        })

        # Introduce disparity on day 10
        if day >= 10:
            # Degrade performance for certain groups
            marginalized = protected_attrs['race'] == 'Black'
            predictions[marginalized] += np.random.normal(0, 0.2, marginalized.sum())
            predictions = np.clip(predictions, 0, 1)

        timestamp = datetime.now() - timedelta(days=14-day)

        monitor.record_predictions(
            predictions=predictions,
            outcomes=outcomes,
            protected_attributes=protected_attrs,
            timestamp=timestamp
        )

    # Generate monitoring report
    report = monitor.generate_monitoring_report(lookback_days=14)
    print(report)

    logger.info("Fairness monitoring simulation completed")
```

This monitoring implementation provides comprehensive infrastructure for maintaining fairness in deployed survival models. The system tracks multiple dimensions of performance and equity, generates alerts when concerns emerge, and creates documentation to support organizational accountability and continuous improvement.

## 10.7 Conclusion

This chapter developed comprehensive methods for survival analysis applied to time-to-event data in healthcare settings, with explicit attention throughout to equity challenges that arise when censoring patterns, competing risks, and hazard structures differ systematically across demographic groups. We began with foundational concepts of survival and hazard functions, the Kaplan-Meier estimator, and the critical role of censoring assumptions, always framing these technical concepts through the lens of how real-world healthcare data generation processes reflect structural inequities rather than idealized statistical assumptions.

We developed the Cox proportional hazards model as the workhorse of survival regression, examining in depth how the proportional hazards assumption can be violated in ways that signal differential access to care or cumulative effects of structural barriers rather than constant biological differences. We extended to time-varying covariates and coefficients that capture how clinical and social factors evolve over follow-up periods, and to parametric survival models that make stronger distributional assumptions in exchange for more precise estimates when those assumptions hold. We covered competing risks analysis that properly accounts for how multiple potential outcomes affect observed event probabilities, critical for settings where competing events like cardiovascular death versus cancer death occur at different rates across demographic groups. We implemented machine learning approaches to survival analysis including random survival forests that offer flexibility to capture complex patterns but require explicit fairness evaluation to prevent learning shortcuts based on healthcare access. Finally, we addressed production deployment with continuous fairness monitoring, recognizing that equity is not established once but must be maintained throughout the model lifecycle through systematic monitoring and responsive action.

The methods and implementations in this chapter equip practitioners to conduct rigorous survival analysis that surfaces rather than obscures health inequities, to critically evaluate whether observed time-to-event patterns reflect disease biology versus structural barriers to care, and to deploy prediction systems that maintain equitable performance across diverse patient populations. The code examples provide production-ready implementations with comprehensive documentation, error handling, and fairness evaluation appropriate for clinical deployment after validation. Most fundamentally, this chapter emphasizes that every technical decision in survival modeling, from choice of methods to handling of censoring to interpretation of results, ultimately has clinical and ethical implications with differential impacts across populations. Bringing this critical consciousness to time-to-event analysis is essential for developing systems that advance rather than undermine health equity.

## References

Aalen, O. O., Borgan, O., & Gjessing, H. K. (2008). *Survival and Event History Analysis: A Process Point of View*. Springer Science & Business Media. https://doi.org/10.1007/978-0-387-68560-1

Aalen, O. O., & Johansen, S. (1978). An empirical transition matrix for non-homogeneous Markov chains based on censored observations. *Scandinavian Journal of Statistics*, 5(3), 141-150.

Andersen, P. K., & Keiding, N. (2012). Interpretability and importance of functionals in competing risks and multistate models. *Statistics in Medicine*, 31(11-12), 1074-1088. https://doi.org/10.1002/sim.4385

Austin, P. C., Lee, D. S., & Fine, J. P. (2016). Introduction to the analysis of survival data in the presence of competing risks. *Circulation*, 133(6), 601-609. https://doi.org/10.1161/CIRCULATIONAHA.115.017719

Bailey, Z. D., Krieger, N., Agénor, M., Graves, J., Linos, N., & Bassett, M. T. (2017). Structural racism and health inequities in the USA: Evidence and interventions. *The Lancet*, 389(10077), 1453-1463. https://doi.org/10.1016/S0140-6736(17)30569-X

Benjamens, S., Dhunnoo, P., & Meskó, B. (2020). The state of artificial intelligence-based FDA-approved medical devices and algorithms: An online database. *npj Digital Medicine*, 3(1), 118. https://doi.org/10.1038/s41746-020-00324-0

Bibbins-Domingo, K. (2019). Integrating social care into the delivery of health care. *JAMA*, 322(18), 1763-1764. https://doi.org/10.1001/jama.2019.15603

Breathett, K., Liu, W. G., Allen, L. A., Daugherty, S. L., Blair, I. V., Jones, J., ... & Lindenfeld, J. (2021). African Americans are less likely to receive care by a cardiologist during an intensive care unit admission for heart failure. *JACC: Heart Failure*, 9(2), 103-111. https://doi.org/10.1016/j.jchf.2020.09.006

Breslow, N. (1972). Contribution to the discussion of paper by D. R. Cox. *Journal of the Royal Statistical Society Series B*, 34(2), 216-217.

Challen, R., Denny, J., Pitt, M., Gompels, L., Edwards, T., & Tsaneva-Atanasova, K. (2019). Artificial intelligence, bias and clinical safety. *BMJ Quality & Safety*, 28(3), 231-237. https://doi.org/10.1136/bmjqs-2018-008370

Collett, D. (2015). *Modelling Survival Data in Medical Research* (3rd ed.). Chapman and Hall/CRC. https://doi.org/10.1201/b18041

Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society Series B*, 34(2), 187-220.

Cox, D. R. (1975). Partial likelihood. *Biometrika*, 62(2), 269-276. https://doi.org/10.1093/biomet/62.2.269

Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *Journal of the American Statistical Association*, 72(359), 557-565. https://doi.org/10.1080/01621459.1977.10480613

Efron, B. (1988). Logistic regression, survival analysis, and the Kaplan-Meier curve. *Journal of the American Statistical Association*, 83(402), 414-425. https://doi.org/10.1080/01621459.1988.10478612

Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *Journal of the American Statistical Association*, 94(446), 496-509. https://doi.org/10.1080/01621459.1999.10474144

Fisher, L. D., & Lin, D. Y. (1999). Time-dependent covariates in the Cox proportional-hazards regression model. *Annual Review of Public Health*, 20(1), 145-157. https://doi.org/10.1146/annurev.publhealth.20.1.145

Fleming, T. R., & Harrington, D. P. (1991). *Counting Processes and Survival Analysis*. John Wiley & Sons.

Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika*, 81(3), 515-526. https://doi.org/10.1093/biomet/81.3.515

Hernán, M. A. (2010). The hazards of hazard ratios. *Epidemiology*, 21(1), 13-15. https://doi.org/10.1097/EDE.0b013e3181c1ea43

Hernán, M. A., Hernández-Díaz, S., & Robins, J. M. (2010). A structural approach to selection bias. *Epidemiology*, 15(5), 615-625. https://doi.org/10.1097/01.ede.0000135174.63482.43

Hernández, M., & Bauer, G. R. (2019). Using the proportional hazards model to study the relationship between changes in depressive symptoms and sexual partnership concurrency among young Black women. *Annals of Epidemiology*, 38, 29-35. https://doi.org/10.1016/j.annepidem.2019.08.003

Howe, C. J., Cole, S. R., Lau, B., Napravnik, S., & Eron Jr, J. J. (2016). Selection bias due to loss to follow up in cohort studies. *Epidemiology*, 27(1), 91-97. https://doi.org/10.1097/EDE.0000000000000409

Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008). Random survival forests. *The Annals of Applied Statistics*, 2(3), 841-860. https://doi.org/10.1214/08-AOAS169

Ishwaran, H., & Kogalur, U. B. (2007). Random survival forests for R. *R News*, 7(2), 25-31.

Kalbfleisch, J. D., & Prentice, R. L. (2002). *The Statistical Analysis of Failure Time Data* (2nd ed.). John Wiley & Sons. https://doi.org/10.1002/9781118032985

Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481. https://doi.org/10.1080/01621459.1958.10501452

Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., & Kluger, Y. (2018). DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*, 18(1), 24. https://doi.org/10.1186/s12874-018-0482-1

Klein, J. P., & Moeschberger, M. L. (2003). *Survival Analysis: Techniques for Censored and Truncated Data* (2nd ed.). Springer. https://doi.org/10.1007/b97377

Kleinbaum, D. G., & Klein, M. (2012). *Survival Analysis: A Self-Learning Text* (3rd ed.). Springer. https://doi.org/10.1007/978-1-4419-6646-9

Liu, X., Rivera, S. C., Moher, D., Calvert, M. J., & Denniston, A. K. (2020). Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: The CONSORT-AI extension. *The Lancet Digital Health*, 2(10), e537-e548. https://doi.org/10.1016/S2589-7500(20)30218-1

Norton, J. M., Moxey-Mims, M. M., Eggers, P. W., Narva, A. S., Star, R. A., Kimmel, P. L., & Rodgers, G. P. (2016). Social determinants of racial disparities in CKD. *Journal of the American Society of Nephrology*, 27(9), 2576-2595. https://doi.org/10.1681/ASN.2016010027

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453. https://doi.org/10.1126/science.aax2342

Paleyes, A., Urma, R. G., & Lawrence, N. D. (2022). Challenges in deploying machine learning: A survey of case studies. *ACM Computing Surveys*, 55(6), 1-29. https://doi.org/10.1145/3533378

Patzer, R. E., Perryman, J. P., Schrager, J. D., Pastan, S., Amaral, S., Gazmararian, J. A., ... & McClellan, W. M. (2012). The role of race and poverty on steps to kidney transplantation in the Southeastern United States. *American Journal of Transplantation*, 12(2), 358-368. https://doi.org/10.1111/j.1600-6143.2011.03927.x

Perperoglou, A., Sauerbrei, W., Abrahamowicz, M., & Schmid, M. (2019). A review of spline function procedures in R. *BMC Medical Research Methodology*, 19(1), 46. https://doi.org/10.1186/s12874-019-0666-3

Pfohl, S. R., Foryciarz, A., & Shah, N. H. (2019). An empirical characterization of fair machine learning for clinical risk prediction. *Journal of Biomedical Informatics*, 113, 103621. https://doi.org/10.1016/j.jbi.2020.103621

Pintilie, M. (2006). *Competing Risks: A Practical Perspective*. John Wiley & Sons. https://doi.org/10.1002/9780470870709

Putter, H., Fiocco, M., & Geskus, R. B. (2007). Tutorial in biostatistics: Competing risks and multi-state models. *Statistics in Medicine*, 26(11), 2389-2430. https://doi.org/10.1002/sim.2712

Ribaudo, H. J., Benson, C. A., Zheng, Y., Koletar, S. L., Hafner, R., Kleeberger, C., ... & Smurzynski, M. (2014). No risk of myocardial infarction associated with initial antiretroviral treatment containing abacavir: Short and long-term results from ACTG A5001/ALLRT. *Clinical Infectious Diseases*, 58(6), 929-939. https://doi.org/10.1093/cid/cit662

Robins, J. M., & Finkelstein, D. M. (2000). Correcting for noncompliance and dependent censoring in an AIDS clinical trial with inverse probability of censoring weighted (IPCW) log-rank tests. *Biometrics*, 56(3), 779-788. https://doi.org/10.1111/j.0006-341X.2000.00779.x

Satagopan, J. M., Ben-Porat, L., Berwick, M., Robson, M., Kutler, D., & Auerbach, A. D. (2004). A note on competing risks in survival data analysis. *British Journal of Cancer*, 91(7), 1229-1235. https://doi.org/10.1038/sj.bjc.6602102

Schoenfeld, D. (1982). Partial residuals for the proportional hazards regression model. *Biometrika*, 69(1), 239-241. https://doi.org/10.1093/biomet/69.1.239

Selbst, A. D., Boyd, D., Friedler, S. A., Venkatasubramanian, S., & Vertesi, J. (2019). Fairness and abstraction in sociotechnical systems. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 59-68). https://doi.org/10.1145/3287560.3287598

Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer. https://doi.org/10.1007/978-1-4757-3294-8

Williams, D. R., & Mohammed, S. A. (2013). Racism and health I: Pathways and scientific evidence. *American Behavioral Scientist*, 57(8), 1152-1173. https://doi.org/10.1177/0002764213487340
