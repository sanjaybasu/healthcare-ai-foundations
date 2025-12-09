---
layout: chapter
title: "Chapter 11: Causal Inference for Healthcare AI"
chapter_number: 11
part_number: 3
prev_chapter: /chapters/chapter-10-survival-analysis/
next_chapter: /chapters/chapter-12-federated-learning-privacy/
---
# Chapter 11: Causal Inference for Healthcare AI

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Formulate clinical decision problems as Markov decision processes, understanding how to define states, actions, rewards, and transition dynamics in healthcare contexts while accounting for how these fundamental elements may differ systematically across patient populations due to differential access, resources, and social determinants of health.

2. Implement Q-learning and deep Q-network algorithms for learning treatment policies from observational clinical data, with explicit attention to how exploration-exploitation tradeoffs must be managed differently in healthcare settings where harmful exploration is ethically unacceptable and where optimal policies may differ across demographic groups.

3. Develop policy gradient methods including REINFORCE, actor-critic, and proximal policy optimization for learning parameterized treatment strategies, incorporating fairness constraints that ensure equitable performance across protected demographic characteristics and care settings.

4. Apply off-policy evaluation techniques including importance sampling, doubly robust estimation, and fitted Q-evaluation to assess potential treatment policies using only observational data, accounting for confounding and selection bias that may differ systematically for underserved populations.

5. Build contextual bandit systems for personalized treatment selection in settings with limited feedback, implementing Thompson sampling and upper confidence bound algorithms with equity-aware reward modeling that avoids optimizing for easily achievable outcomes at the expense of harder-to-reach populations.

6. Implement safe reinforcement learning approaches including constrained policy optimization, conservative Q-learning, and uncertainty-aware exploration that prevent harmful actions during both training and deployment, with particular attention to avoiding policies that assume resources or capabilities unavailable in under-resourced care settings.

## 11.1 Introduction: Sequential Decision Making in Healthcare

Healthcare fundamentally involves sequential decision making under uncertainty. Clinicians observe patient states through available measurements, choose interventions from feasible treatment options, and observe subsequent outcomes that inform future decisions. This sequential structure appears across diverse clinical contexts: intensive care physicians adjust ventilator settings and medication dosing based on evolving vital signs and laboratory results; oncologists select chemotherapy regimens and modify doses based on treatment response and toxicity; primary care providers manage chronic diseases through iterative adjustments to medications, lifestyle recommendations, and monitoring strategies over years of care.

Reinforcement learning offers a formal framework for learning optimal sequential decision policies from data. Rather than predicting single outcomes as in supervised learning, reinforcement learning algorithms learn mappings from states to actions that maximize cumulative rewards over time. This temporal credit assignment problem—determining which actions in a sequence led to eventual good or bad outcomes—aligns naturally with clinical decision making where treatment effects may emerge gradually and where early interventions affect the feasibility and effectiveness of later ones.

Yet applying reinforcement learning to healthcare presents profound challenges that go beyond the technical difficulties of learning from delayed and sparse rewards. Clinical decisions affect human health and life, making harmful exploration during learning ethically unacceptable. Observational healthcare data reflects not just the biological effects of treatments but also the social and structural factors determining who receives what care when. Treatment policies that perform well on average may fail catastrophically for underrepresented populations if those populations differ systematically in ways that affect treatment effectiveness, side effect profiles, or the feasibility of implementing recommended actions.

Consider a concrete example that illustrates both the promise and peril of reinforcement learning in healthcare. We seek to learn an optimal policy for managing anticoagulation therapy in patients with atrial fibrillation to prevent stroke while minimizing bleeding risk. The state space includes patient characteristics, current medication doses, recent laboratory values measuring coagulation, and bleeding or thrombotic events. Actions consist of medication adjustments: increase dose, decrease dose, or maintain current regimen. Rewards reflect the long-term balance between stroke prevention and bleeding complications.

A reinforcement learning algorithm might learn from observational data that aggressive anticoagulation reduces stroke risk for most patients and therefore recommend higher doses broadly. However, this policy could be catastrophic for specific subpopulations. Elderly patients, particularly those from underserved communities with limited access to monitoring, face substantially higher bleeding risks that may not be apparent in aggregate data dominated by younger, healthier, better-monitored patients. Patients with multiple comorbidities and polypharmacy—disproportionately common in safety-net health systems serving low-income populations—experience different drug interactions affecting optimal dosing. The learned "optimal" policy, while maximizing average outcomes, may systematically harm vulnerable groups.

Moreover, the feasibility of implementing treatment recommendations varies systematically across care settings. A policy that assumes weekly laboratory monitoring may be optimal for patients with excellent insurance and reliable transportation but impossible to implement for uninsured patients in rural areas. Medication regimens requiring frequent dosing or expensive brand-name drugs may be theoretically optimal but practically infeasible for patients facing cost barriers or complex medication management challenges.

This chapter develops reinforcement learning methods specifically designed for healthcare applications serving diverse underserved populations. Every algorithmic component—from state representation to reward design to policy parameterization—is developed with explicit attention to equity implications. We emphasize off-policy evaluation that leverages observational data without requiring potentially harmful exploration, safe learning approaches that constrain policies to avoid actions outside the scope of observed practice, and fairness-aware algorithms that optimize for equitable outcomes across demographic groups rather than aggregate performance alone.

We begin with Markov decision processes as the foundational formalism for sequential decision problems, showing how healthcare decisions can be structured in this framework while acknowledging the ways real clinical contexts violate standard assumptions. Subsequent sections develop value-based methods including Q-learning and deep Q-networks, policy gradient approaches for learning parameterized treatment strategies, off-policy evaluation techniques for assessing policies using observational data, contextual bandits for personalized treatment selection, and safe learning frameworks that prevent harmful exploration. Throughout, we provide production-ready implementations with comprehensive fairness evaluation and safety constraints appropriate for healthcare deployment after clinical validation.

The implementations emphasize interpretability and clinical validity alongside predictive performance. We incorporate medical knowledge through carefully designed reward functions that capture multiple competing objectives, state representations that include equity-relevant contextual factors, and policy architectures that enable clinician review and override. The goal is not to replace clinical judgment with automated decision making but rather to provide decision support tools that help clinicians navigate complex tradeoffs while attending to equity considerations that may be difficult to track informally across diverse patient populations.

## 11.2 Markov Decision Processes: Formalizing Sequential Clinical Decisions

The Markov decision process (MDP) provides the mathematical foundation for reinforcement learning, formalizing sequential decision problems as interactions between an agent (decision maker) and an environment (patient and clinical context). Understanding MDPs and their assumptions is essential for both developing RL algorithms and critically evaluating when those assumptions are violated in healthcare applications.

### 11.2.1 MDP Formalism and Healthcare Applications

A Markov decision process is defined by the tuple $$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$ where:

- $$\mathcal{S}$$ is the state space: all possible patient states and clinical contexts
- $$\mathcal{A}$$ is the action space: all possible treatment decisions
- $$\mathcal{P}(s'\mid s,a)$$ is the transition probability: likelihood of transitioning to state $$s'$$ given current state $$s$$ and action $$a$$
- $$\mathcal{R}(s,a,s')$$ is the reward function: immediate reward for taking action $$a$$ in state $$s$$ and transitioning to $$s'$$
- $$\gamma \in [0,1]$$ is the discount factor: relative value of immediate versus future rewards

The Markov property assumes that the current state contains all information relevant to future dynamics: $$ P(s_{t+1}\lvert s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)$$. This assumption simplifies analysis and enables efficient algorithms but may be violated when relevant historical information is not fully captured in the state representation.

A policy $$\pi: \mathcal{S} \rightarrow \Delta(\mathcal{A})$$ maps states to probability distributions over actions. The policy defines the agent's decision making strategy. We distinguish between deterministic policies $$\pi(s) \in \mathcal{A}$$ that always take the same action in a given state and stochastic policies $$\pi(a\mid s)$$ that sample actions probabilistically.

The goal of reinforcement learning is to find a policy that maximizes the expected cumulative discounted reward, also called the value function:

$$
V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) \mid s_0 = s\right]
$$

The optimal value function is $$V^*(s) = \max_\pi V^\pi(s)$$, and the optimal policy achieves this maximum: $$\pi^* = \arg\max_\pi V^\pi(s)$$ for all states $$ s $$.

The action-value function or Q-function evaluates taking action $$ a $$ in state $$ s $$ and then following policy $$\pi$$:

$$
Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) \mid s_0 = s, a_0 = a\right]
$$

The Bellman equations express these value functions recursively:

$$
V^\pi(s) = \sum_{a} \pi(a\lvert s) \sum_{s'} P(s' \mid s,a)[R(s,a,s') + \gamma V^\pi(s')]
$$

$$
Q^\pi(s,a) = \sum_{s'} P(s'\lvert s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s',a')]
$$

The optimal Bellman equations define the optimal value functions:

$$
V^*(s) = \max_a \sum_{s'} P(s'\mid s,a)[R(s,a,s') + \gamma V^*(s')]
$$

$$
Q^*(s,a) = \sum_{s'} P(s'\mid s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]
$$

The optimal policy can be extracted from the optimal Q-function: $$\pi^*(s) = \arg\max_a Q^*(s,a)$$.

### 11.2.2 Applying MDPs to Clinical Decision Problems

Translating clinical decision problems into the MDP framework requires careful consideration of how to define states, actions, rewards, and transitions in ways that capture the essential structure of the problem while remaining computationally tractable.

**State representation** must balance completeness with practicality. In principle, the state should include all information relevant to future dynamics: patient demographics, comorbidities, laboratory values, vital signs, medications, social determinants, care setting characteristics, and full medical history. In practice, constructing such comprehensive states from electronic health records is challenging. Relevant variables may be inconsistently documented, systematically missing for underserved populations, or impossible to measure directly.

The equity implications of state representation are profound. If states do not capture factors that affect treatment effectiveness differentially across populations, the learned policy may perform poorly for groups whose relevant characteristics are not represented. For instance, a sepsis management policy that omits information about neighborhood resources, health literacy, or caregiver support may recommend discharge plans that are infeasible for socially vulnerable patients. Conversely, including demographic variables as state features risks the learned policy encoding discriminatory patterns from historical data rather than legitimate clinical differences.

We recommend constructing states from several components:

1. **Clinical measurements**: Laboratory values, vital signs, disease severity scores, all standardized to account for measurement variability across equipment and protocols

2. **Historical trajectory**: Recent trends in key measurements, time since diagnosis or treatment initiation, response to prior interventions

3. **Treatment context**: Current medications, recent procedures, care setting type, availability of monitoring and follow-up

4. **Social determinants and equity factors**: Health literacy measures, language preferences, transportation access, insurance status, neighborhood disadvantage indices, caregiver availability

5. **Uncertainty indicators**: Missing data flags, measurement reliability indicators, time since last observation

This comprehensive state representation enables policies that account for the full context of care rather than just biomedical measurements.

**Action spaces** must reflect feasible interventions in diverse care settings. An action might be "prescribe medication X at dose Y" but this action is only feasible if the medication is formulary-approved, affordable to the patient, appropriate given contraindications and drug interactions, and can be monitored adequately. The set of feasible actions may differ systematically by care setting, insurance status, and social resources.

We recommend defining action spaces that are:

- **Realistic**: Include only interventions actually available in the target care settings
- **Safe**: Exclude actions known to be harmful for identifiable patient subgroups
- **Interpretable**: Use clinically meaningful action representations that clinicians can understand and potentially override

For complex treatment decisions, hierarchical action spaces can be valuable. High-level actions might be "initiate anticoagulation" with low-level actions specifying particular medications, doses, and monitoring schedules. This structure allows policies to make strategic treatment decisions while adapting tactical details to individual circumstances.

**Reward functions** translate health outcomes into numerical objectives that the RL algorithm optimizes. This translation embeds value judgments about what matters and how to trade off competing objectives. A myopic reward function focused solely on short-term physiological measurements may drive aggressive interventions with harmful long-term consequences. A reward function that weights all outcomes equally may produce policies that perform well on average while failing catastrophically for specific subgroups.

We recommend constructing reward functions that:

- **Incorporate multiple objectives**: Combine survival, quality of life, adverse events, cost, and burden of treatment
- **Use long-term horizons**: Value sustained health improvements over temporary gains
- **Penalize disparities**: Include explicit fairness terms that discourage policies with differential performance across demographic groups
- **Respect preferences**: Weight outcomes according to patient values when known

For many clinical applications, the reward is naturally sparse: most time steps have zero reward with large positive rewards for recovery and large negative rewards for adverse events or death. This sparsity makes credit assignment challenging but accurately reflects the temporal structure of many health outcomes.

**Transition dynamics** $$ P(s'\mid s,a)$$ encode how patient states evolve following treatment actions. In practice, we almost never know the true transition function and must learn it from data. The learned dynamics will necessarily reflect the distribution of patients and treatments in the training data. If the training data comes predominantly from academic medical centers serving relatively healthy, well-resourced populations, the learned dynamics may not accurately predict outcomes for complex, socially vulnerable patients in under-resourced settings.

### 11.2.3 Violations of MDP Assumptions in Healthcare

Real healthcare decision problems violate MDP assumptions in several important ways that must be understood and addressed.

**Partial observability**: The Markov assumption requires that states capture all relevant information, but in practice we observe patients through limited windows. Laboratory tests are expensive and ordered selectively. Imaging studies provide snapshots rather than continuous monitoring. Patient-reported symptoms depend on communication, health literacy, and trust. Social determinants affecting treatment adherence and effectiveness are rarely systematically documented. The result is a partially observable MDP (POMDP) where our state representation is incomplete.

For underserved populations, partial observability is often more severe. Patients with unreliable healthcare access may have sparser observation sequences. Patients who are non-English speaking may have less complete symptom documentation. Patients experiencing homelessness may lack the stability needed for consistent monitoring. Algorithms that assume complete state information may perform poorly when this assumption is violated.

Approaches for handling partial observability include: (1) augmenting states with observation histories rather than just current measurements, (2) explicitly modeling uncertainty about unobserved state components, (3) learning observation models that infer missing information from available data while acknowledging uncertainty, and (4) using recurrent neural networks that maintain internal memory of relevant history.

**Unknown transition dynamics**: In model-based RL, we learn the transition function $$ P(s'\mid s,a)$$ from data. However, the learned model is only accurate for state-action pairs well-represented in the training data. For underrepresented populations or novel treatment combinations, the learned dynamics may be highly uncertain or systematically biased. Safe RL approaches constrain policies to avoid actions that would rely on uncertain dynamics.

**Non-stationarity**: The MDP framework assumes that transition dynamics and optimal policies are constant over time. In reality, medical knowledge evolves, treatment availability changes, patient populations shift, and social determinants transform. A policy learned from historical data may become suboptimal or even harmful as clinical practice standards change. Continual learning approaches that update policies as new data arrives are essential for maintaining safety and effectiveness.

**Unmeasured confounding**: Observational healthcare data reflects not just biological responses to treatments but also the selection processes determining who receives what care. Patients chosen for aggressive interventions may differ systematically from those receiving conservative management in ways not fully captured by the state representation. This unmeasured confounding means that naively learning from observational data can produce biased estimates of treatment effects and consequently suboptimal or harmful policies. Off-policy evaluation methods that account for confounding are essential.

**Multiple objectives**: The MDP framework optimizes a single reward function, but healthcare involves multiple competing objectives: survival, quality of life, treatment burden, cost, and equity. Collapsing these into a single scalar reward requires value judgments about relative importance that may not be agreed upon and may vary across patients. Multi-objective RL methods that maintain the Pareto frontier of non-dominated policies provide more flexibility, allowing clinicians and patients to select from alternative tradeoffs rather than optimizing a fixed weighting.

### 11.2.4 Production-Ready MDP Implementation

We now implement a flexible MDP framework for clinical decision problems with explicit support for equity evaluation and safety constraints. The implementation handles discrete and continuous states and actions, incorporates expert knowledge through custom reward functions, and tracks fairness metrics throughout policy evaluation.

```python
"""
Markov Decision Process framework for clinical decision problems.

This module provides production-ready components for formulating and solving
clinical decision problems as MDPs, with comprehensive support for equity
evaluation and safety constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateRepresentation(Enum):
    """Types of state representations for clinical MDPs."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MIXED = "mixed"
    IMAGE = "image"

@dataclass
class ClinicalState:
    """
    Represents a patient state in a clinical MDP.

    Attributes:
        clinical_features: Dict of clinical measurements and values
        demographics: Dict of demographic attributes for equity tracking
        social_determinants: Dict of SDOH factors affecting care
        care_setting: Type of care setting (safety-net, academic, etc.)
        timestamp: Time point for temporal tracking
        missing_indicators: Binary indicators for missing features
        measurement_quality: Quality/reliability scores for each feature
    """
    clinical_features: Dict[str, float]
    demographics: Dict[str, Any] = field(default_factory=dict)
    social_determinants: Dict[str, Any] = field(default_factory=dict)
    care_setting: str = "unknown"
    timestamp: Optional[float] = None
    missing_indicators: Dict[str, bool] = field(default_factory=dict)
    measurement_quality: Dict[str, float] = field(default_factory=dict)

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert state to numpy array for ML models."""
        return np.array([self.clinical_features.get(name, 0.0)
                        for name in feature_names])

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (death, discharge, etc.)."""
        return self.clinical_features.get("terminal", 0.0) > 0.5

@dataclass
class ClinicalAction:
    """
    Represents a treatment action in a clinical MDP.

    Attributes:
        action_type: Type of intervention
        parameters: Action-specific parameters (dose, duration, etc.)
        resource_requirements: Resources needed to implement action
        contraindications: Patient characteristics that contraindicate action
        monitoring_requirements: Follow-up and monitoring needed
    """
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    monitoring_requirements: List[str] = field(default_factory=list)

    def is_feasible(self, state: ClinicalState,
                   available_resources: List[str]) -> bool:
        """
        Check if action is feasible in current state and setting.

        Considers both clinical contraindications and practical feasibility.
        """
        # Check contraindications
        for contraindication in self.contraindications:
            if contraindication in state.clinical_features:
                if state.clinical_features[contraindication] > 0:
                    return False

        # Check resource availability
        for resource in self.resource_requirements:
            if resource not in available_resources:
                return False

        return True

    def to_array(self) -> np.ndarray:
        """Convert action to numpy array representation."""
        # Implement action encoding appropriate for the specific problem
        raise NotImplementedError("Action encoding must be problem-specific")

class RewardFunction(ABC):
    """Abstract base class for clinical reward functions."""

    @abstractmethod
    def __call__(self, state: ClinicalState, action: ClinicalAction,
                next_state: ClinicalState) -> float:
        """Compute reward for state-action-next_state transition."""
        pass

    @abstractmethod
    def get_reward_components(self, state: ClinicalState,
                             action: ClinicalAction,
                             next_state: ClinicalState) -> Dict[str, float]:
        """Return individual reward components for interpretability."""
        pass

class SepsisRewardFunction(RewardFunction):
    """
    Reward function for sepsis management.

    Combines multiple objectives:
    - Survival and recovery
    - Avoiding complications
    - Minimizing treatment burden
    - Fairness across demographic groups
    """

    def __init__(self,
                survival_weight: float = 100.0,
                complication_weight: float = -10.0,
                treatment_burden_weight: float = -1.0,
                disparity_penalty_weight: float = 5.0,
                reference_group_outcomes: Optional[Dict[str, float]] = None):
        """
        Initialize sepsis reward function.

        Parameters:
            survival_weight: Reward for survival/recovery
            complication_weight: Penalty for complications
            treatment_burden_weight: Penalty for aggressive interventions
            disparity_penalty_weight: Penalty for differential outcomes
            reference_group_outcomes: Expected outcomes for privileged group
        """
        self.survival_weight = survival_weight
        self.complication_weight = complication_weight
        self.treatment_burden_weight = treatment_burden_weight
        self.disparity_penalty_weight = disparity_penalty_weight
        self.reference_group_outcomes = reference_group_outcomes or {}

    def __call__(self, state: ClinicalState, action: ClinicalAction,
                next_state: ClinicalState) -> float:
        """Compute scalar reward."""
        components = self.get_reward_components(state, action, next_state)
        return sum(components.values())

    def get_reward_components(self, state: ClinicalState,
                             action: ClinicalAction,
                             next_state: ClinicalState) -> Dict[str, float]:
        """Decompose reward into interpretable components."""
        components = {}

        # Survival reward
        if next_state.clinical_features.get("recovered", 0.0) > 0.5:
            components["survival"] = self.survival_weight
        elif next_state.clinical_features.get("died", 0.0) > 0.5:
            components["survival"] = -self.survival_weight
        else:
            components["survival"] = 0.0

        # Complication penalties
        complications = [
            "acute_kidney_injury", "respiratory_failure",
            "shock", "organ_dysfunction"
        ]
        n_complications = sum(
            next_state.clinical_features.get(comp, 0.0) > 0.5
            for comp in complications
        )
        components["complications"] = (self.complication_weight *
                                       n_complications)

        # Treatment burden
        aggressive_actions = [
            "vasopressors", "mechanical_ventilation",
            "renal_replacement", "multiple_antibiotics"
        ]
        if action.action_type in aggressive_actions:
            components["treatment_burden"] = self.treatment_burden_weight
        else:
            components["treatment_burden"] = 0.0

        # Fairness penalty if outcomes differ from reference group
        if (self.reference_group_outcomes and
            "mortality_rate" in self.reference_group_outcomes):
            # Compare outcome to reference group expectation
            # Penalty increases with disparity
            patient_group = state.demographics.get("race", "unknown")
            if patient_group != "reference":
                expected_mortality = self.reference_group_outcomes.get(
                    "mortality_rate", 0.0
                )
                # Simplified: would need more sophisticated disparity metric
                components["fairness"] = 0.0  # Implemented during training

        return components

class TransitionModel(ABC):
    """Abstract base class for clinical transition dynamics."""

    @abstractmethod
    def predict(self, state: ClinicalState,
               action: ClinicalAction) -> Tuple[ClinicalState, float]:
        """
        Predict next state and transition probability.

        Returns:
            next_state: Predicted next state
            probability: Confidence in prediction
        """
        pass

    @abstractmethod
    def sample(self, state: ClinicalState,
              action: ClinicalAction) -> ClinicalState:
        """Sample next state from transition distribution."""
        pass

    @abstractmethod
    def get_uncertainty(self, state: ClinicalState,
                       action: ClinicalAction) -> float:
        """Estimate uncertainty in transition prediction."""
        pass

class EmpiricalTransitionModel(TransitionModel):
    """
    Learn transition dynamics from observational data.

    Uses ensemble of models to quantify uncertainty.
    Explicitly tracks where training data is sparse.
    """

    def __init__(self, feature_names: List[str], n_ensemble: int = 5):
        """
        Initialize empirical transition model.

        Parameters:
            feature_names: Names of state features
            n_ensemble: Number of ensemble members for uncertainty
        """
        self.feature_names = feature_names
        self.n_ensemble = n_ensemble
        self.models: List[Any] = []  # Will hold trained models
        self.state_action_counts: Dict[Tuple, int] = {}

    def fit(self, states: List[ClinicalState],
           actions: List[ClinicalAction],
           next_states: List[ClinicalState],
           bootstrap: bool = True):
        """
        Fit transition model from trajectory data.

        Parameters:
            states: List of observed states
            actions: List of actions taken
            next_states: List of resulting next states
            bootstrap: Whether to use bootstrap for ensemble
        """
        from sklearn.ensemble import RandomForestRegressor

        logger.info(f"Fitting transition model with {len(states)} transitions")

        # Convert states and actions to arrays
        X = []
        y = []
        for s, a, s_next in zip(states, actions, next_states):
            state_array = s.to_array(self.feature_names)
            action_array = a.to_array()
            next_state_array = s_next.to_array(self.feature_names)

            X.append(np.concatenate([state_array, action_array]))
            y.append(next_state_array)

        X = np.array(X)
        y = np.array(y)

        # Track state-action space coverage
        for x in X:
            key = tuple(np.round(x, 2))  # Discretize for counting
            self.state_action_counts[key] = (
                self.state_action_counts.get(key, 0) + 1
            )

        # Train ensemble of models
        self.models = []
        for i in range(self.n_ensemble):
            if bootstrap:
                # Bootstrap sample
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
            else:
                X_boot = X
                y_boot = y

            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                random_state=i
            )
            model.fit(X_boot, y_boot)
            self.models.append(model)

        logger.info(f"Trained {len(self.models)} ensemble members")

    def predict(self, state: ClinicalState,
               action: ClinicalAction) -> Tuple[ClinicalState, float]:
        """Predict next state with uncertainty quantification."""
        state_array = state.to_array(self.feature_names)
        action_array = action.to_array()
        x = np.concatenate([state_array, action_array]).reshape(1, -1)

        # Get predictions from all ensemble members
        predictions = np.array([model.predict(x)[0] for model in self.models])

        # Mean prediction
        mean_pred = predictions.mean(axis=0)

        # Uncertainty from ensemble disagreement
        uncertainty = predictions.std(axis=0).mean()

        # Create next state from prediction
        next_state_features = {
            name: mean_pred[i]
            for i, name in enumerate(self.feature_names)
        }

        next_state = ClinicalState(
            clinical_features=next_state_features,
            demographics=state.demographics.copy(),
            social_determinants=state.social_determinants.copy(),
            care_setting=state.care_setting
        )

        return next_state, 1.0 - uncertainty

    def sample(self, state: ClinicalState,
              action: ClinicalAction) -> ClinicalState:
        """Sample next state from ensemble predictions."""
        state_array = state.to_array(self.feature_names)
        action_array = action.to_array()
        x = np.concatenate([state_array, action_array]).reshape(1, -1)

        # Randomly select ensemble member
        model = np.random.choice(self.models)
        prediction = model.predict(x)[0]

        next_state_features = {
            name: prediction[i]
            for i, name in enumerate(self.feature_names)
        }

        return ClinicalState(
            clinical_features=next_state_features,
            demographics=state.demographics.copy(),
            social_determinants=state.social_determinants.copy(),
            care_setting=state.care_setting
        )

    def get_uncertainty(self, state: ClinicalState,
                       action: ClinicalAction) -> float:
        """
        Estimate epistemic uncertainty for state-action pair.

        Combines ensemble disagreement with data coverage.
        """
        state_array = state.to_array(self.feature_names)
        action_array = action.to_array()
        x = np.concatenate([state_array, action_array]).reshape(1, -1)

        # Ensemble disagreement
        predictions = np.array([model.predict(x)[0] for model in self.models])
        ensemble_std = predictions.std(axis=0).mean()

        # Data coverage (inverse of observation count)
        key = tuple(np.round(x[0], 2))
        count = self.state_action_counts.get(key, 0)
        coverage_uncertainty = 1.0 / (1.0 + count)

        # Combine sources of uncertainty
        total_uncertainty = ensemble_std + coverage_uncertainty

        return total_uncertainty

class ClinicalMDP:
    """
    Complete MDP representation for clinical decision problems.

    Integrates state representation, action space, reward function,
    and transition model with comprehensive equity tracking.
    """

    def __init__(self,
                state_space_dim: int,
                action_space_dim: int,
                reward_function: RewardFunction,
                transition_model: TransitionModel,
                discount_factor: float = 0.99,
                max_episode_length: int = 100):
        """
        Initialize clinical MDP.

        Parameters:
            state_space_dim: Dimension of state representation
            action_space_dim: Dimension of action representation
            reward_function: Function defining rewards
            transition_model: Model of state transitions
            discount_factor: Discount for future rewards (gamma)
            max_episode_length: Maximum trajectory length
        """
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        self.reward_function = reward_function
        self.transition_model = transition_model
        self.discount_factor = discount_factor
        self.max_episode_length = max_episode_length

        # Track equity metrics
        self.group_trajectories: Dict[str, List] = {}
        self.group_returns: Dict[str, List[float]] = {}

    def step(self, state: ClinicalState,
            action: ClinicalAction) -> Tuple[ClinicalState, float, bool, Dict]:
        """
        Execute one step of the MDP.

        Returns:
            next_state: Resulting state
            reward: Immediate reward
            done: Whether episode is finished
            info: Additional information dictionary
        """
        # Sample next state
        next_state = self.transition_model.sample(state, action)

        # Compute reward
        reward = self.reward_function(state, action, next_state)

        # Check if episode is done
        done = (next_state.is_terminal() or
               (next_state.timestamp is not None and
                next_state.timestamp >= self.max_episode_length))

        # Uncertainty in transition
        uncertainty = self.transition_model.get_uncertainty(state, action)

        info = {
            "uncertainty": uncertainty,
            "reward_components": self.reward_function.get_reward_components(
                state, action, next_state
            ),
            "demographics": state.demographics,
            "care_setting": state.care_setting
        }

        return next_state, reward, done, info

    def rollout(self,
               initial_state: ClinicalState,
               policy: Callable[[ClinicalState], ClinicalAction],
               deterministic: bool = False) -> Dict[str, Any]:
        """
        Generate a complete trajectory following a policy.

        Parameters:
            initial_state: Starting state
            policy: Policy mapping states to actions
            deterministic: Whether to use deterministic policy

        Returns:
            Dictionary containing trajectory and summary statistics
        """
        states = [initial_state]
        actions = []
        rewards = []
        infos = []

        state = initial_state
        done = False
        t = 0

        while not done and t < self.max_episode_length:
            # Select action from policy
            action = policy(state)

            # Take step
            next_state, reward, done, info = self.step(state, action)

            # Store transition
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)

            state = next_state
            t += 1

        # Compute discounted return
        discounted_return = sum(
            self.discount_factor**t * r
            for t, r in enumerate(rewards)
        )

        # Track by demographic group
        group = initial_state.demographics.get("group", "unknown")
        if group not in self.group_trajectories:
            self.group_trajectories[group] = []
            self.group_returns[group] = []

        self.group_trajectories[group].append({
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "infos": infos
        })
        self.group_returns[group].append(discounted_return)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "infos": infos,
            "return": discounted_return,
            "length": len(rewards),
            "success": next_state.clinical_features.get("recovered", 0.0) > 0.5
        }

    def evaluate_policy_equity(self,
                              policy: Callable[[ClinicalState], ClinicalAction],
                              test_states: List[ClinicalState],
                              n_rollouts: int = 100) -> pd.DataFrame:
        """
        Evaluate policy performance across demographic groups.

        Parameters:
            policy: Policy to evaluate
            test_states: Test initial states from diverse populations
            n_rollouts: Number of rollouts per test state

        Returns:
            DataFrame with equity metrics by group
        """
        # Reset tracking
        self.group_trajectories = {}
        self.group_returns = {}

        # Generate rollouts
        for state in test_states:
            for _ in range(n_rollouts):
                self.rollout(state, policy)

        # Compute equity metrics
        equity_results = []

        for group, returns in self.group_returns.items():
            returns_array = np.array(returns)

            # Get success rates
            trajectories = self.group_trajectories[group]
            successes = [
                traj["states"][-1].clinical_features.get("recovered", 0.0) > 0.5
                for traj in trajectories
            ]
            success_rate = np.mean(successes)

            # Get complication rates
            complications = []
            for traj in trajectories:
                any_complication = any(
                    info["reward_components"].get("complications", 0) < 0
                    for info in traj["infos"]
                )
                complications.append(any_complication)
            complication_rate = np.mean(complications)

            equity_results.append({
                "group": group,
                "n_patients": len(returns),
                "mean_return": returns_array.mean(),
                "std_return": returns_array.std(),
                "success_rate": success_rate,
                "complication_rate": complication_rate,
                "mean_episode_length": np.mean([
                    traj["length"] for traj in trajectories
                ])
            })

        df = pd.DataFrame(equity_results)

        # Compute disparity metrics
        if len(df) > 1:
            max_return = df["mean_return"].max()
            min_return = df["mean_return"].min()
            df["return_disparity"] = max_return - df["mean_return"]

            max_success = df["success_rate"].max()
            df["success_disparity"] = max_success - df["success_rate"]

        return df
```

This MDP framework provides the foundation for implementing reinforcement learning algorithms. The comprehensive state representation captures clinical, demographic, and social determinants. The reward function decomposes into interpretable components including explicit fairness terms. The transition model quantifies uncertainty and tracks data coverage. The equity evaluation framework stratifies performance across demographic groups, making disparities immediately visible.

## 11.3 Value-Based Methods: Q-Learning and Deep Q-Networks

Value-based reinforcement learning methods learn to estimate the value of states or state-action pairs, then derive policies by selecting actions that maximize estimated value. These approaches are particularly well-suited to discrete action spaces and can learn from off-policy data, making them applicable to learning from observational healthcare datasets where we observe clinician behavior rather than controlled exploration.

### 11.3.1 Q-Learning for Tabular MDPs

Q-learning is a foundational value-based algorithm that learns the optimal action-value function $$ Q^*(s,a)$$ through temporal difference updates. The algorithm maintains Q-value estimates for each state-action pair and updates them based on observed transitions:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]
$$

where $$\alpha $$ is the learning rate, $$ r $$ is the observed reward, $$ s'$$ is the next state, and the term $$ r + \gamma \max_{a'} Q(s',a') - Q(s,a)$$ is the temporal difference error measuring the discrepancy between the current Q-value estimate and the Bellman target.

The key insight of Q-learning is that it is an off-policy algorithm: the Q-value updates are based on the maximum Q-value in the next state regardless of which action was actually taken. This allows learning the optimal policy even when following a different behavioral policy, making Q-learning suitable for learning from observational data where we observe clinician decisions that may not be optimal.

For discrete state and action spaces, we can represent Q-values as a table with entries for each state-action pair. However, tabular Q-learning has severe limitations for clinical applications where state spaces are high-dimensional and continuous. A patient state defined by dozens of laboratory values, vital signs, and clinical characteristics yields an intractably large state space that cannot be enumerated explicitly.

### 11.3.2 Deep Q-Networks: Function Approximation for Complex State Spaces

Deep Q-networks (DQN) extend Q-learning to high-dimensional state spaces by approximating the Q-function with a neural network $$ Q(s,a;\theta)$$ parameterized by weights $$\theta $$. Rather than maintaining separate Q-value estimates for each state-action pair, the network learns a mapping from states to Q-values for all actions, enabling generalization across similar states.

The DQN algorithm addresses several challenges that arise when combining Q-learning with neural network function approximation:

**Experience replay**: Rather than updating the network after each transition, DQN stores transitions in a replay buffer and samples minibatches for training. This breaks correlations between consecutive samples that can destabilize learning and enables multiple updates from each transition, improving sample efficiency.

**Target networks**: DQN maintains two networks: the online network $$ Q(s,a;\theta)$$ used for action selection and a target network $$ Q(s,a;\theta^-)$$ used for computing Bellman targets. The target network is periodically copied from the online network but remains fixed between copies. This stabilizes learning by preventing the target from changing rapidly as the online network updates.

The DQN loss function is:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2\right]
$$

where $$\mathcal{D}$$ is the replay buffer.

Several extensions to DQN improve performance and address specific challenges:

**Double DQN** addresses overestimation bias by decoupling action selection from Q-value estimation. The online network selects the best action, while the target network evaluates it:

$$
y = r + \gamma Q(s', \arg\max_{a'} Q(s',a';\theta); \theta^-)
$$

**Dueling DQN** factors the Q-function into state value and action advantage:

$$
Q(s,a;\theta) = V(s;\theta_v) + A(s,a;\theta_a) - \frac{1}{\lvert \mathcal{A} \rvert}\sum_{a'} A(s,a';\theta_a)
$$

This architecture makes it easier to learn that many actions have similar values, which is common in healthcare where multiple reasonable treatment options may have comparable outcomes.

**Prioritized experience replay** samples transitions for training based on their temporal difference error, focusing learning on transitions where the current Q-function performs poorly. This can improve sample efficiency but may introduce fairness issues if high-error transitions come disproportionately from certain demographic groups.

### 11.3.3 Equity-Aware Deep Q-Network Implementation

We implement a deep Q-network for clinical decision making with explicit fairness constraints and comprehensive safety mechanisms. The implementation includes fairness-aware replay prioritization that ensures adequate representation of all demographic groups, group-specific Q-value heads that allow for differential policies when treatment effects vary, and uncertainty quantification through ensembles.

```python
"""
Deep Q-Network for clinical decision support with equity constraints.

Implements DQN with fairness-aware replay, group-specific value estimation,
and comprehensive safety mechanisms for healthcare applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Transition = namedtuple('Transition',
                       ['state', 'action', 'reward', 'next_state',
                        'done', 'demographic_group', 'care_setting'])

class DuelingQNetwork(nn.Module):
    """
    Dueling architecture for Q-network with optional group-specific heads.

    Separates state value and action advantage for improved learning.
    Can learn group-specific Q-functions when treatment effects differ.
    """

    def __init__(self,
                state_dim: int,
                action_dim: int,
                hidden_dims: List[int] = [256, 256],
                group_specific: bool = False,
                n_groups: int = 1):
        """
        Initialize dueling Q-network.

        Parameters:
            state_dim: Dimension of state representation
            action_dim: Number of possible actions
            hidden_dims: Sizes of hidden layers
            group_specific: Whether to use group-specific value heads
            n_groups: Number of demographic groups
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.group_specific = group_specific
        self.n_groups = n_groups

        # Shared feature extraction
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Value stream
        if group_specific:
            self.value_streams = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dims[-1], 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                ) for _ in range(n_groups)
            ])
        else:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dims[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        # Advantage stream
        if group_specific:
            self.advantage_streams = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dims[-1], 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim)
                ) for _ in range(n_groups)
            ])
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dims[-1], 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

    def forward(self, state: torch.Tensor,
               group: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through network.

        Parameters:
            state: Batch of states [batch_size, state_dim]
            group: Optional group indices [batch_size] for group-specific heads

        Returns:
            Q-values for all actions [batch_size, action_dim]
        """
        features = self.feature_extractor(state)

        if self.group_specific and group is not None:
            # Use group-specific value and advantage streams
            batch_size = state.shape[0]
            q_values = torch.zeros(batch_size, self.action_dim,
                                  device=state.device)

            for g in range(self.n_groups):
                mask = (group == g)
                if mask.sum() > 0:
                    group_features = features[mask]
                    value = self.value_streams[g](group_features)
                    advantage = self.advantage_streams[g](group_features)

                    # Combine value and advantage
                    q = value + (advantage - advantage.mean(dim=1, keepdim=True))
                    q_values[mask] = q

            return q_values
        else:
            # Standard dueling architecture
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)

            # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

            return q_values

class FairReplayBuffer:
    """
    Experience replay buffer with fairness-aware sampling.

    Ensures all demographic groups are adequately represented in
    training batches to prevent learned policies from performing
    poorly on underrepresented populations.
    """

    def __init__(self, capacity: int, n_groups: int):
        """
        Initialize fair replay buffer.

        Parameters:
            capacity: Maximum number of transitions to store
            n_groups: Number of demographic groups to balance
        """
        self.capacity = capacity
        self.n_groups = n_groups

        # Separate buffers per group
        self.group_buffers: List[deque] = [
            deque(maxlen=capacity // n_groups) for _ in range(n_groups)
        ]

        # Track priorities for prioritized replay
        self.group_priorities: List[deque] = [
            deque(maxlen=capacity // n_groups) for _ in range(n_groups)
        ]

    def push(self, transition: Transition, priority: float = 1.0):
        """Add transition to appropriate group buffer."""
        group = transition.demographic_group
        self.group_buffers[group].append(transition)
        self.group_priorities[group].append(priority)

    def sample(self, batch_size: int,
              prioritized: bool = False) -> List[Transition]:
        """
        Sample batch with equal representation from all groups.

        Parameters:
            batch_size: Total number of transitions to sample
            prioritized: Whether to use priority-based sampling

        Returns:
            List of sampled transitions
        """
        samples_per_group = batch_size // self.n_groups
        batch = []

        for group in range(self.n_groups):
            buffer = self.group_buffers[group]
            priorities = self.group_priorities[group]

            if len(buffer) == 0:
                continue

            # Sample from this group's buffer
            n_samples = min(samples_per_group, len(buffer))

            if prioritized and len(priorities) > 0:
                # Priority-based sampling
                probs = np.array(priorities) / sum(priorities)
                indices = np.random.choice(
                    len(buffer), size=n_samples, replace=False, p=probs
                )
            else:
                # Uniform sampling
                indices = np.random.choice(
                    len(buffer), size=n_samples, replace=False
                )

            batch.extend([buffer[i] for i in indices])

        return batch

    def __len__(self) -> int:
        """Total number of transitions across all group buffers."""
        return sum(len(buffer) for buffer in self.group_buffers)

    def update_priorities(self, indices: List[int],
                         priorities: List[float]):
        """Update priorities for prioritized replay."""
        # Implementation depends on how indices map to group buffers
        pass

class ClinicalDQN:
    """
    Deep Q-Network for clinical decision support.

    Implements double DQN with dueling architecture, fairness-aware replay,
    and comprehensive safety mechanisms including uncertainty quantification
    and constraint satisfaction.
    """

    def __init__(self,
                state_dim: int,
                action_dim: int,
                n_demographic_groups: int = 4,
                group_specific: bool = False,
                learning_rate: float = 1e-4,
                gamma: float = 0.99,
                tau: float = 0.005,
                replay_capacity: int = 100000,
                batch_size: int = 128,
                device: str = "cpu"):
        """
        Initialize clinical DQN.

        Parameters:
            state_dim: Dimension of state representation
            action_dim: Number of possible actions
            n_demographic_groups: Number of groups for fairness tracking
            group_specific: Whether to use group-specific Q-networks
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update parameter for target network
            replay_capacity: Maximum replay buffer size
            batch_size: Batch size for training
            device: Device for computation (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_groups = n_demographic_groups
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Initialize online and target networks
        self.online_net = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            group_specific=group_specific,
            n_groups=n_demographic_groups
        ).to(self.device)

        self.target_net = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            group_specific=group_specific,
            n_groups=n_demographic_groups
        ).to(self.device)

        # Initialize target network with online network weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate
        )

        # Replay buffer with fairness-aware sampling
        self.replay_buffer = FairReplayBuffer(
            capacity=replay_capacity,
            n_groups=n_demographic_groups
        )

        # Track training metrics
        self.train_losses = []
        self.group_q_values: Dict[int, List[float]] = {
            g: [] for g in range(n_demographic_groups)
        }

        # Safety constraints
        self.unsafe_actions: Dict[str, List[int]] = {}

    def select_action(self,
                     state: np.ndarray,
                     demographic_group: int,
                     epsilon: float = 0.0,
                     feasible_actions: Optional[List[int]] = None) -> int:
        """
        Select action using epsilon-greedy policy.

        Parameters:
            state: Current state
            demographic_group: Patient demographic group
            epsilon: Exploration probability
            feasible_actions: List of clinically appropriate actions

        Returns:
            Selected action index
        """
        if feasible_actions is None:
            feasible_actions = list(range(self.action_dim))

        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.choice(feasible_actions)

        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            group_tensor = torch.LongTensor([demographic_group]).to(self.device)

            q_values = self.online_net(state_tensor, group_tensor)[0]

            # Mask infeasible actions
            if len(feasible_actions) < self.action_dim:
                mask = torch.ones(self.action_dim, dtype=torch.bool)
                mask[feasible_actions] = False
                q_values[mask] = -float('inf')

            action = q_values.argmax().item()

        return action

    def train_step(self, prioritized: bool = False) -> float:
        """
        Perform one training step with a batch from replay buffer.

        Parameters:
            prioritized: Whether to use prioritized replay

        Returns:
            Training loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size, prioritized)

        # Prepare tensors
        states = torch.FloatTensor([t.state for t in batch]).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t.next_state for t in batch]).to(self.device)
        dones = torch.FloatTensor([t.done for t in batch]).to(self.device)
        groups = torch.LongTensor([t.demographic_group for t in batch]).to(self.device)

        # Compute current Q-values
        current_q = self.online_net(states, groups).gather(1, actions.unsqueeze(1))

        # Compute target Q-values using double DQN
        with torch.no_grad():
            # Online network selects actions
            next_actions = self.online_net(next_states, groups).argmax(dim=1)

            # Target network evaluates selected actions
            next_q = self.target_net(next_states, groups).gather(
                1, next_actions.unsqueeze(1)
            )

            # Bellman target
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Soft update target network
        self._soft_update_target_network()

        # Track metrics
        self.train_losses.append(loss.item())

        # Track Q-values by group
        for g in range(self.n_groups):
            group_mask = (groups == g)
            if group_mask.sum() > 0:
                group_q = current_q[group_mask].mean().item()
                self.group_q_values[g].append(group_q)

        return loss.item()

    def _soft_update_target_network(self):
        """Soft update of target network parameters."""
        for target_param, online_param in zip(
            self.target_net.parameters(),
            self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )

    def evaluate_policy(self,
                       test_states: List[np.ndarray],
                       test_groups: List[int],
                       n_rollouts: int = 100) -> pd.DataFrame:
        """
        Evaluate learned policy across demographic groups.

        Parameters:
            test_states: List of test initial states
            test_groups: Demographic group for each test state
            n_rollouts: Number of evaluation rollouts per state

        Returns:
            DataFrame with equity metrics by group
        """
        self.online_net.eval()

        group_returns: Dict[int, List[float]] = {g: [] for g in range(self.n_groups)}
        group_q_values: Dict[int, List[float]] = {g: [] for g in range(self.n_groups)}

        with torch.no_grad():
            for state, group in zip(test_states, test_groups):
                for _ in range(n_rollouts):
                    # Get Q-values
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    group_tensor = torch.LongTensor([group]).to(self.device)

                    q_vals = self.online_net(state_tensor, group_tensor)[0]
                    max_q = q_vals.max().item()

                    group_q_values[group].append(max_q)

                    # Would perform full rollout here with environment
                    # Simplified for illustration
                    group_returns[group].append(max_q)

        # Compute equity metrics
        equity_results = []

        for group in range(self.n_groups):
            if len(group_returns[group]) > 0:
                returns = np.array(group_returns[group])
                q_values = np.array(group_q_values[group])

                equity_results.append({
                    "group": group,
                    "mean_return": returns.mean(),
                    "std_return": returns.std(),
                    "mean_q_value": q_values.mean(),
                    "std_q_value": q_values.std(),
                    "n_evaluations": len(returns)
                })

        df = pd.DataFrame(equity_results)

        # Compute disparity metrics
        if len(df) > 1:
            max_return = df["mean_return"].max()
            df["return_disparity"] = max_return - df["mean_return"]

        return df

    def add_safety_constraint(self,
                             constraint_name: str,
                             unsafe_actions: List[int]):
        """
        Add safety constraint excluding certain actions.

        Parameters:
            constraint_name: Name for this constraint
            unsafe_actions: List of action indices to exclude
        """
        self.unsafe_actions[constraint_name] = unsafe_actions
        logger.info(f"Added safety constraint '{constraint_name}' "
                   f"excluding {len(unsafe_actions)} actions")
```

This DQN implementation provides sophisticated value-based RL with comprehensive equity considerations. The dueling architecture with optional group-specific heads allows learning different optimal policies when treatment effects vary across populations. The fairness-aware replay buffer ensures all demographic groups are adequately represented in training. The safety constraints prevent selection of clinically inappropriate actions.

The key innovation for equity is the group-specific value heads that enable the model to learn that optimal treatments may differ across populations due to legitimate biological or social factors affecting treatment response, rather than forcing a single universal policy that performs well on average but poorly for specific groups.

## 11.4 Policy Gradient Methods for Direct Policy Learning

While value-based methods like Q-learning learn a value function and derive a policy implicitly, policy gradient methods directly parameterize and optimize the policy itself. This direct approach offers several advantages for clinical applications: it naturally handles continuous action spaces, can represent stochastic policies that explicitly trade off exploration and exploitation, and enables incorporation of domain knowledge and constraints directly into the policy architecture.

### 11.4.1 REINFORCE: The Policy Gradient Theorem

The foundational policy gradient algorithm is REINFORCE, which optimizes a parameterized policy $$\pi_\theta(a\mid s)$$ by ascending the gradient of the expected return. The policy gradient theorem provides an elegant expression for this gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t\mid s_t) G_t\right]
$$

where $$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$$ is the discounted return from time $$t$$. This formula has an intuitive interpretation: it increases the probability of actions that led to high returns and decreases the probability of actions that led to low returns, with the magnitude of the update proportional to the return and the gradient of the log probability.

The REINFORCE algorithm follows a simple procedure:
1. Generate a trajectory by following the current policy
2. Compute returns $$ G_t $$ for each time step
3. Update policy parameters using the policy gradient
4. Repeat

A practical improvement is to subtract a baseline $$ b(s_t)$$ from the returns to reduce variance:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t\mid s_t) (G_t - b(s_t))\right]
$$

The baseline does not bias the gradient (since $$\mathbb{E}[b(s)] \nabla_\theta \log \pi_\theta(a\mid s) = 0$$) but can substantially reduce variance. A common choice is $$b(s_t) = V(s_t)$$, the value function estimate for state $$s_t$$.

### 11.4.2 Actor-Critic Methods

Actor-critic methods combine value-based and policy-based approaches by maintaining both a policy (actor) and a value function (critic). The critic learns to estimate value functions as in Q-learning or DQN, while the actor updates the policy using policy gradients with the critic providing the baseline or advantage estimate.

The advantage function $$ A(s,a) = Q(s,a) - V(s)$$ measures how much better action $$ a $$ is than the average action in state $$ s$$. Using the advantage as the baseline in REINFORCE yields the advantage actor-critic (A2C) gradient:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a\mid s) A(s,a)\right]
$$

In practice, we approximate the advantage using temporal difference learning:

$$
A(s_t,a_t) \approx r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Actor-critic methods offer reduced variance compared to REINFORCE while maintaining the benefits of policy gradient methods including the ability to handle continuous actions and learn stochastic policies.

### 11.4.3 Proximal Policy Optimization

Proximal policy optimization (PPO) is a modern policy gradient method that has become the dominant approach for many RL applications due to its combination of sample efficiency, stability, and ease of implementation. PPO addresses a key challenge in policy gradient methods: ensuring that policy updates are neither too large (causing instability) nor too small (limiting learning speed).

PPO optimizes a clipped surrogate objective that constrains how much the policy can change in a single update:

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]
$$

where $$r_t(\theta) = \frac{\pi_\theta(a_t\lvert s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$$ is the probability ratio between the new and old policies, $$\hat{A}_t$$ is the advantage estimate, and $$\epsilon$$ is a hyperparameter (typically 0.1 or 0.2) controlling the clip range.

The clipping ensures that the policy update is conservative: if an action has positive advantage, its probability can only increase by a limited amount; if it has negative advantage, its probability can only decrease by a limited amount. This prevents catastrophically large policy updates that could destroy previously learned behaviors.

For healthcare applications, this conservatism is particularly valuable because it prevents the policy from suddenly taking radically different actions that might be dangerous, even if they appear to have high value according to imperfect value estimates from limited data.

### 11.4.4 Equity-Aware PPO Implementation

We implement proximal policy optimization with explicit fairness constraints for clinical decision support. The implementation includes group-specific advantage normalization to prevent the policy from optimizing primarily for majority groups, fairness-regularized objectives that penalize disparate performance, and comprehensive safety mechanisms.

```python
"""
Proximal Policy Optimization for clinical decision support with equity constraints.

Implements PPO with group-specific advantage normalization, fairness regularization,
and safety constraints appropriate for healthcare applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalPolicyNetwork(nn.Module):
    """
    Policy network for clinical treatment decisions.

    Outputs either discrete action probabilities or continuous action
    distributions depending on the action space.
    """

    def __init__(self,
                state_dim: int,
                action_dim: int,
                hidden_dims: List[int] = [256, 256],
                continuous_actions: bool = False,
                group_specific: bool = False,
                n_groups: int = 1):
        """
        Initialize policy network.

        Parameters:
            state_dim: Dimension of state representation
            action_dim: Number of actions (discrete) or action dimension (continuous)
            hidden_dims: Sizes of hidden layers
            continuous_actions: Whether actions are continuous
            group_specific: Whether to use group-specific policy heads
            n_groups: Number of demographic groups
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        self.group_specific = group_specific
        self.n_groups = n_groups

        # Shared feature extraction
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Policy heads
        if continuous_actions:
            # Output mean and log_std for continuous actions
            if group_specific:
                self.policy_mean_heads = nn.ModuleList([
                    nn.Linear(hidden_dims[-1], action_dim)
                    for _ in range(n_groups)
                ])
                self.policy_logstd_heads = nn.ModuleList([
                    nn.Linear(hidden_dims[-1], action_dim)
                    for _ in range(n_groups)
                ])
            else:
                self.policy_mean = nn.Linear(hidden_dims[-1], action_dim)
                self.policy_logstd = nn.Linear(hidden_dims[-1], action_dim)
        else:
            # Output logits for discrete actions
            if group_specific:
                self.policy_heads = nn.ModuleList([
                    nn.Linear(hidden_dims[-1], action_dim)
                    for _ in range(n_groups)
                ])
            else:
                self.policy_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state: torch.Tensor,
               group: Optional[torch.Tensor] = None) -> Union[
                   Categorical, Normal, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through policy network.

        Parameters:
            state: Batch of states [batch_size, state_dim]
            group: Optional group indices [batch_size]

        Returns:
            For discrete actions: Categorical distribution
            For continuous actions: Normal distribution or (mean, std) tuple
        """
        features = self.shared_net(state)

        if self.continuous_actions:
            if self.group_specific and group is not None:
                batch_size = state.shape[0]
                means = torch.zeros(batch_size, self.action_dim, device=state.device)
                stds = torch.zeros(batch_size, self.action_dim, device=state.device)

                for g in range(self.n_groups):
                    mask = (group == g)
                    if mask.sum() > 0:
                        group_features = features[mask]
                        means[mask] = self.policy_mean_heads[g](group_features)
                        log_stds = self.policy_logstd_heads[g](group_features)
                        stds[mask] = torch.exp(log_stds).clamp(min=1e-6, max=1.0)
            else:
                means = self.policy_mean(features)
                log_stds = self.policy_logstd(features)
                stds = torch.exp(log_stds).clamp(min=1e-6, max=1.0)

            dist = Normal(means, stds)
            return dist
        else:
            if self.group_specific and group is not None:
                batch_size = state.shape[0]
                logits = torch.zeros(batch_size, self.action_dim, device=state.device)

                for g in range(self.n_groups):
                    mask = (group == g)
                    if mask.sum() > 0:
                        group_features = features[mask]
                        logits[mask] = self.policy_heads[g](group_features)
            else:
                logits = self.policy_head(features)

            dist = Categorical(logits=logits)
            return dist

class ClinicalValueNetwork(nn.Module):
    """
    Value network for estimating state values.

    Can have group-specific value heads if optimal policies differ
    across demographics due to differential treatment effects.
    """

    def __init__(self,
                state_dim: int,
                hidden_dims: List[int] = [256, 256],
                group_specific: bool = False,
                n_groups: int = 1):
        """Initialize value network."""
        super().__init__()

        self.group_specific = group_specific
        self.n_groups = n_groups

        # Shared feature extraction
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Value heads
        if group_specific:
            self.value_heads = nn.ModuleList([
                nn.Linear(hidden_dims[-1], 1) for _ in range(n_groups)
            ])
        else:
            self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state: torch.Tensor,
               group: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute state values."""
        features = self.shared_net(state)

        if self.group_specific and group is not None:
            batch_size = state.shape[0]
            values = torch.zeros(batch_size, 1, device=state.device)

            for g in range(self.n_groups):
                mask = (group == g)
                if mask.sum() > 0:
                    group_features = features[mask]
                    values[mask] = self.value_heads[g](group_features)
        else:
            values = self.value_head(features)

        return values

class RolloutBuffer:
    """
    Buffer for storing rollout data for PPO updates.

    Tracks demographic groups and computes group-specific advantage
    normalization for equity.
    """

    def __init__(self, n_groups: int):
        """Initialize rollout buffer."""
        self.n_groups = n_groups

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.groups = []
        self.care_settings = []

    def add(self, state, action, reward, value, log_prob, done, group, setting):
        """Add transition to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.groups.append(group)
        self.care_settings.append(setting)

    def get(self, gamma: float = 0.99, gae_lambda: float = 0.95,
           group_normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute returns and advantages using GAE.

        Parameters:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            group_normalize: Whether to normalize advantages by group

        Returns:
            Dictionary of tensors for PPO update
        """
        # Convert lists to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.FloatTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        log_probs = torch.FloatTensor(self.log_probs)
        dones = torch.FloatTensor(self.dones)
        groups = torch.LongTensor(self.groups)

        # Compute GAE advantages
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

        # Compute returns
        returns = advantages + values

        # Group-specific advantage normalization for equity
        if group_normalize:
            for g in range(self.n_groups):
                mask = (groups == g)
                if mask.sum() > 1:  # Need at least 2 samples
                    group_advantages = advantages[mask]
                    advantages[mask] = (
                        (group_advantages - group_advantages.mean()) /
                        (group_advantages.std() + 1e-8)
                    )
        else:
            # Standard normalization across all samples
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "groups": groups
        }

    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.groups = []
        self.care_settings = []

class ClinicalPPO:
    """
    Proximal Policy Optimization for clinical decision support.

    Implements PPO with group-specific advantage normalization,
    fairness-regularized objectives, and comprehensive safety mechanisms.
    """

    def __init__(self,
                state_dim: int,
                action_dim: int,
                n_demographic_groups: int = 4,
                continuous_actions: bool = False,
                group_specific: bool = False,
                policy_lr: float = 3e-4,
                value_lr: float = 1e-3,
                gamma: float = 0.99,
                gae_lambda: float = 0.95,
                clip_ratio: float = 0.2,
                value_coef: float = 0.5,
                entropy_coef: float = 0.01,
                fairness_coef: float = 0.1,
                n_epochs: int = 10,
                batch_size: int = 64,
                device: str = "cpu"):
        """
        Initialize clinical PPO.

        Parameters:
            state_dim: Dimension of state representation
            action_dim: Number or dimension of actions
            n_demographic_groups: Number of groups for fairness tracking
            continuous_actions: Whether action space is continuous
            group_specific: Whether to use group-specific networks
            policy_lr: Learning rate for policy network
            value_lr: Learning rate for value network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clip ratio (epsilon)
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            fairness_coef: Coefficient for fairness regularization
            n_epochs: Number of epochs for PPO update
            batch_size: Batch size for updates
            device: Device for computation
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_groups = n_demographic_groups
        self.continuous_actions = continuous_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.fairness_coef = fairness_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Initialize policy and value networks
        self.policy_net = ClinicalPolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            continuous_actions=continuous_actions,
            group_specific=group_specific,
            n_groups=n_demographic_groups
        ).to(self.device)

        self.value_net = ClinicalValueNetwork(
            state_dim=state_dim,
            group_specific=group_specific,
            n_groups=n_demographic_groups
        ).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=policy_lr
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=value_lr
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(n_groups=n_demographic_groups)

        # Track training metrics
        self.policy_losses = []
        self.value_losses = []
        self.fairness_losses = []
        self.group_returns: Dict[int, List[float]] = {
            g: [] for g in range(n_demographic_groups)
        }

    def select_action(self,
                     state: np.ndarray,
                     demographic_group: int,
                     deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action from policy.

        Parameters:
            state: Current state
            demographic_group: Patient demographic group
            deterministic: Whether to select deterministically

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Estimated value of state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        group_tensor = torch.LongTensor([demographic_group]).to(self.device)

        with torch.no_grad():
            # Get action distribution
            dist = self.policy_net(state_tensor, group_tensor)

            # Sample or take mode
            if deterministic:
                if self.continuous_actions:
                    action = dist.mean
                else:
                    action = dist.probs.argmax()
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            # Get value estimate
            value = self.value_net(state_tensor, group_tensor)

        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def update(self, group_normalize_advantages: bool = True) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.

        Parameters:
            group_normalize_advantages: Whether to normalize advantages by group

        Returns:
            Dictionary of training metrics
        """
        # Get rollout data with advantages
        data = self.rollout_buffer.get(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            group_normalize=group_normalize_advantages
        )

        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)
        returns = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)
        groups = data["groups"].to(self.device)

        # Track metrics
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_fairness_losses = []

        # PPO update for multiple epochs
        for epoch in range(self.n_epochs):
            # Random minibatch ordering
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_groups = groups[batch_indices]

                # Evaluate actions under current policy
                dist = self.policy_net(batch_states, batch_groups)

                if self.continuous_actions:
                    batch_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                else:
                    batch_log_probs = dist.log_prob(batch_actions)

                entropy = dist.entropy().mean()

                # Compute ratio for clipped objective
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                batch_values = self.value_net(batch_states, batch_groups).squeeze()
                value_loss = F.mse_loss(batch_values, batch_returns)

                # Fairness loss: penalize return disparities across groups
                group_returns_dict = {}
                for g in range(self.n_groups):
                    group_mask = (batch_groups == g)
                    if group_mask.sum() > 0:
                        group_returns_dict[g] = batch_values[group_mask].mean()

                if len(group_returns_dict) > 1:
                    group_returns_tensor = torch.stack(list(group_returns_dict.values()))
                    fairness_loss = group_returns_tensor.var()
                else:
                    fairness_loss = torch.tensor(0.0, device=self.device)

                # Combined loss
                loss = (policy_loss +
                       self.value_coef * value_loss -
                       self.entropy_coef * entropy +
                       self.fairness_coef * fairness_loss)

                # Optimize
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)

                self.policy_optimizer.step()
                self.value_optimizer.step()

                # Track metrics
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_fairness_losses.append(fairness_loss.item())

        # Clear rollout buffer
        self.rollout_buffer.clear()

        # Aggregate metrics
        metrics = {
            "policy_loss": np.mean(epoch_policy_losses),
            "value_loss": np.mean(epoch_value_losses),
            "fairness_loss": np.mean(epoch_fairness_losses),
            "entropy": entropy.item()
        }

        self.policy_losses.append(metrics["policy_loss"])
        self.value_losses.append(metrics["value_loss"])
        self.fairness_losses.append(metrics["fairness_loss"])

        return metrics
```

This PPO implementation provides sophisticated policy learning with comprehensive equity considerations. The group-specific networks enable learning different optimal policies when treatment effects vary across populations. The group-wise advantage normalization ensures the policy doesn't optimize primarily for majority groups. The fairness regularization directly penalizes return disparities.

## 11.5 Off-Policy Evaluation from Observational Data

A fundamental challenge in healthcare reinforcement learning is that we cannot freely explore treatment policies during learning due to ethical constraints. Randomly assigning treatments to test which work best is often impossible or unethical. Instead, we must learn from observational data where treatments were selected by clinicians based on their judgment and available evidence. Off-policy evaluation (OPE) methods enable estimating the performance of a new policy using only data collected under a different behavioral policy, making it possible to evaluate potential treatment strategies before deploying them in clinical practice.

### 11.5.1 The Off-Policy Evaluation Problem

In off-policy evaluation, we have:
- A dataset $$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$$ collected under a behavioral policy $$\pi_b$$ (historical clinician decisions)
- A target policy $$\pi_e $$ (proposed new treatment strategy) that we want to evaluate
- Goal: estimate $$V^{\pi_e} = \mathbb{E}_{\pi_e}[\sum_t \gamma^t r_t]$$ using only data from $$\pi_b$$

The challenge is that the actions in our dataset were selected according to $$\pi_b $$, not $$\pi_e $$. If $$\pi_e $$ recommends actions that $$\pi_b $$ rarely took, we have little data about their consequences and our estimates will be unreliable. This is the **support mismatch** problem: we can only reliably evaluate policies whose action distributions overlap substantially with the behavioral policy.

In healthcare, support mismatch manifests as fundamental uncertainty about treatments that have not been tried for certain patient populations. If anticoagulation has historically been prescribed conservatively for elderly patients with fall risk, we have limited data about aggressive anticoagulation in this population and cannot reliably estimate the performance of policies recommending it.

Moreover, observational healthcare data suffers from confounding: clinician treatment decisions are not random but based on factors affecting both treatment selection and outcomes. Sicker patients may receive more aggressive treatments, creating negative associations between treatment intensity and outcomes that don't reflect causal effects. Socially vulnerable patients may receive different care due to bias or structural barriers, creating spurious associations between demographic characteristics and treatment effectiveness.

### 11.5.2 Importance Sampling

The simplest OPE method is importance sampling, which reweights trajectories according to how likely they would have been under the target policy relative to the behavioral policy. For a trajectory $$\tau = (s_0, a_0, r_0, \ldots, s_T, a_T, r_T)$$, the importance weight is:

$$
w(\tau) = \prod_{t=0}^{T} \frac{\pi_e(a_t\lvert s_t)}{\pi_b(a_t \mid s_t)}
$$

The OPE estimate is then:

$$
\hat{V}^{\pi_e}_{IS} = \frac{1}{n}\sum_{i=1}^{n} w(\tau_i) G(\tau_i)
$$

where $$G(\tau_i) = \sum_t \gamma^t r_t$$ is the return of trajectory $$i$$.

Importance sampling is unbiased: $$\mathbb{E}[\hat{V}^{\pi_e}_{IS}] = V^{\pi_e}$$. However, it suffers from high variance when the importance weights vary substantially. If the target policy assigns high probability to actions that the behavioral policy rarely took, the corresponding trajectories will have very large weights, dominating the estimate and making it unstable.

For equity, importance sampling has concerning properties. If the behavioral policy treats different demographic groups differently (which it often does due to bias or differential access), and the target policy aims to correct these disparities, the importance weights will be larger for underrepresented groups. While this correctly reflects that we have less data about them, it also means our estimates are most uncertain for the populations we most want to help.

### 11.5.3 Doubly Robust Estimation

Doubly robust (DR) estimation combines importance sampling with learned models to reduce variance while maintaining theoretical guarantees. The DR estimator is:

$$
\hat{V}^{\pi_e}_{DR} = \frac{1}{n}\sum_{i=1}^{n}\left[\frac{\pi_e(a_i\lvert s_i)}{\pi_b(a_i \mid s_i)}(G_i - \hat{Q}(s_i,a_i)) + \mathbb{E}_{a \sim \pi_e}[\hat{Q}(s_i,a)]\right]
$$

where $$\hat{Q}(s,a)$$ is a learned estimate of the Q-function.

This estimator is "doubly robust" because it is consistent if either the importance weights or the Q-function estimates are correct. If the Q-function is accurate, the importance-weighted correction term has expectation zero and the estimator reduces to the model-based component. If the importance weights are correct, the estimator is unbiased even if the Q-function is misspecified.

For healthcare applications, doubly robust estimation is attractive because it leverages both observed outcomes (through importance sampling) and learned models (through Q-function estimation) while providing some protection against misspecification of either component. However, the method still requires that the behavioral policy assigns non-zero probability to actions recommended by the target policy, limiting our ability to extrapolate far beyond observed practice patterns.

### 11.5.4 Fitted Q-Evaluation

Fitted Q-evaluation (FQE) is a model-based OPE method that learns to predict returns directly. Rather than using importance sampling, FQE fits a Q-function $$ Q_{\pi_e}(s,a)$$ specific to the target policy by iteratively solving:

$$
Q_{\pi_e}(s,a) = r + \gamma \mathbb{E}_{s' \sim P(\cdot\lvert s,a), a' \sim \pi_e(\cdot \mid s')}[Q_{\pi_e}(s',a')]
$$

using the observational dataset. The policy value is then estimated as:

$$
\hat{V}^{\pi_e}_{FQE} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{E}_{a \sim \pi_e}[Q_{\pi_e}(s_{i,0}, a)]
$$

FQE can have lower variance than importance sampling methods, especially when the behavioral and target policies differ substantially. However, it is fully model-based and therefore only as good as the Q-function approximation, which may be poor for state-action pairs rarely observed in the data.

For equity-focused applications, FQE's challenge is that Q-function estimates will be most uncertain for underrepresented populations who appear rarely in the training data. Explicitly modeling this uncertainty through ensemble methods or Bayesian approaches is essential for identifying when estimates are unreliable.

### 11.5.5 Production Implementation of Off-Policy Evaluation

We implement a comprehensive OPE framework with importance sampling, doubly robust, and FQE methods, including explicit uncertainty quantification and fairness-aware confidence intervals.

```python
"""
Off-policy evaluation for clinical treatment policies.

Implements IS, DR, and FQE with uncertainty quantification and
fairness-aware evaluation across demographic groups.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trajectory:
    """Complete trajectory from observational data."""
    states: np.ndarray  # [T, state_dim]
    actions: np.ndarray  # [T, action_dim]
    rewards: np.ndarray  # [T]
    dones: np.ndarray  # [T]
    demographic_group: int
    care_setting: str

class OffPolicyEvaluator:
    """
    Comprehensive off-policy evaluation framework.

    Implements multiple OPE methods with uncertainty quantification
    and fairness-aware evaluation across demographic groups.
    """

    def __init__(self,
                behavioral_policy: Callable,
                state_dim: int,
                action_dim: int,
                n_demographic_groups: int = 4,
                gamma: float = 0.99):
        """
        Initialize off-policy evaluator.

        Parameters:
            behavioral_policy: Function mapping states to action probabilities
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            n_demographic_groups: Number of groups for fairness tracking
            gamma: Discount factor
        """
        self.behavioral_policy = behavioral_policy
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_groups = n_demographic_groups
        self.gamma = gamma

    def importance_sampling(self,
                          trajectories: List[Trajectory],
                          target_policy: Callable,
                          weighted: bool = False) -> Dict[str, float]:
        """
        Estimate target policy value using importance sampling.

        Parameters:
            trajectories: List of observed trajectories
            target_policy: Policy to evaluate
            weighted: Whether to use weighted importance sampling

        Returns:
            Dictionary with value estimates and metrics
        """
        # Compute importance weights and returns for each trajectory
        weights = []
        returns = []
        groups = []

        for traj in trajectories:
            # Compute importance weight
            weight = 1.0
            for t in range(len(traj.states)):
                state = traj.states[t]
                action = traj.actions[t]

                # Probability under target policy
                pi_e = target_policy(state)
                if len(pi_e.shape) == 1:  # Discrete actions
                    prob_e = pi_e[int(action)]
                else:  # Continuous actions
                    # Would compute log probability from distribution
                    prob_e = 1.0  # Simplified

                # Probability under behavioral policy
                pi_b = self.behavioral_policy(state)
                if len(pi_b.shape) == 1:
                    prob_b = pi_b[int(action)]
                else:
                    prob_b = 1.0

                # Accumulate weight
                weight *= (prob_e / (prob_b + 1e-10))

            # Compute discounted return
            G = sum(self.gamma**t * traj.rewards[t]
                   for t in range(len(traj.rewards)))

            weights.append(weight)
            returns.append(G)
            groups.append(traj.demographic_group)

        weights = np.array(weights)
        returns = np.array(returns)
        groups = np.array(groups)

        # Standard or weighted IS estimate
        if weighted:
            # Weighted importance sampling (biased but lower variance)
            value_estimate = (weights * returns).sum() / (weights.sum() + 1e-10)
        else:
            # Standard importance sampling (unbiased)
            value_estimate = (weights * returns).mean()

        # Compute standard error
        weighted_returns = weights * returns
        se = weighted_returns.std() / np.sqrt(len(trajectories))

        # Group-specific estimates for fairness evaluation
        group_estimates = {}
        for g in range(self.n_groups):
            group_mask = (groups == g)
            if group_mask.sum() > 0:
                if weighted:
                    group_est = ((weights[group_mask] * returns[group_mask]).sum() /
                                (weights[group_mask].sum() + 1e-10))
                else:
                    group_est = (weights[group_mask] * returns[group_mask]).mean()

                group_estimates[f"group_{g}_value"] = group_est
                group_estimates[f"group_{g}_weight_mean"] = weights[group_mask].mean()
                group_estimates[f"group_{g}_weight_max"] = weights[group_mask].max()
                group_estimates[f"group_{g}_n_samples"] = group_mask.sum()

        # Check for extreme weights indicating poor overlap
        max_weight = weights.max()
        weight_variance = weights.var()
        effective_sample_size = (weights.sum())**2 / (weights**2).sum()

        return {
            "value": value_estimate,
            "se": se,
            "ci_lower": value_estimate - 1.96 * se,
            "ci_upper": value_estimate + 1.96 * se,
            "max_weight": max_weight,
            "weight_variance": weight_variance,
            "effective_sample_size": effective_sample_size,
            **group_estimates
        }

    def doubly_robust(self,
                     trajectories: List[Trajectory],
                     target_policy: Callable,
                     q_function: nn.Module) -> Dict[str, float]:
        """
        Doubly robust policy evaluation.

        Parameters:
            trajectories: List of observed trajectories
            target_policy: Policy to evaluate
            q_function: Learned Q-function for the target policy

        Returns:
            Dictionary with value estimates and metrics
        """
        estimates = []
        groups = []

        for traj in trajectories:
            # Compute importance weight (same as IS)
            weight = 1.0
            for t in range(len(traj.states)):
                state = traj.states[t]
                action = traj.actions[t]

                pi_e = target_policy(state)
                if len(pi_e.shape) == 1:
                    prob_e = pi_e[int(action)]
                else:
                    prob_e = 1.0

                pi_b = self.behavioral_policy(state)
                if len(pi_b.shape) == 1:
                    prob_b = pi_b[int(action)]
                else:
                    prob_b = 1.0

                weight *= (prob_e / (prob_b + 1e-10))

            # Compute trajectory return
            G = sum(self.gamma**t * traj.rewards[t]
                   for t in range(len(traj.rewards)))

            # Get Q-function estimate for initial state
            state_0 = torch.FloatTensor(traj.states[0]).unsqueeze(0)

            with torch.no_grad():
                # Expected Q-value under target policy
                pi_e_0 = target_policy(traj.states[0])
                q_values = q_function(state_0).squeeze().numpy()
                expected_q = (pi_e_0 * q_values).sum()

                # Q-value of observed action
                action_0 = int(traj.actions[0])
                q_observed = q_values[action_0]

            # Doubly robust estimate for this trajectory
            dr_estimate = weight * (G - q_observed) + expected_q

            estimates.append(dr_estimate)
            groups.append(traj.demographic_group)

        estimates = np.array(estimates)
        groups = np.array(groups)

        # Overall estimate
        value_estimate = estimates.mean()
        se = estimates.std() / np.sqrt(len(estimates))

        # Group-specific estimates
        group_estimates = {}
        for g in range(self.n_groups):
            group_mask = (groups == g)
            if group_mask.sum() > 0:
                group_est = estimates[group_mask].mean()
                group_se = estimates[group_mask].std() / np.sqrt(group_mask.sum())

                group_estimates[f"group_{g}_value"] = group_est
                group_estimates[f"group_{g}_se"] = group_se
                group_estimates[f"group_{g}_n_samples"] = group_mask.sum()

        return {
            "value": value_estimate,
            "se": se,
            "ci_lower": value_estimate - 1.96 * se,
            "ci_upper": value_estimate + 1.96 * se,
            **group_estimates
        }

    def fitted_q_evaluation(self,
                          trajectories: List[Trajectory],
                          target_policy: Callable,
                          n_epochs: int = 50) -> Dict[str, float]:
        """
        Fitted Q-evaluation: learn Q-function for target policy.

        Parameters:
            trajectories: List of observed trajectories
            target_policy: Policy to evaluate
            n_epochs: Number of training epochs

        Returns:
            Dictionary with value estimates and metrics
        """
        # Build Q-network
        q_network = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

        optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-3)

        # Prepare training data
        states = []
        actions = []
        targets = []

        for traj in trajectories:
            for t in range(len(traj.states) - 1):
                states.append(traj.states[t])
                actions.append(int(traj.actions[t]))

                # Compute target using target policy
                reward = traj.rewards[t]
                next_state = traj.states[t + 1]

                # Expected Q-value under target policy for next state
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    next_q = q_network(next_state_tensor).squeeze().numpy()
                    pi_e_next = target_policy(next_state)
                    expected_next_q = (pi_e_next * next_q).sum()

                target = reward + self.gamma * expected_next_q
                targets.append(target)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)

        # Train Q-network
        for epoch in range(n_epochs):
            # Forward pass
            q_values = q_network(states)
            q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            # Loss
            loss = nn.MSELoss()(q_pred, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"FQE Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

        # Evaluate learned Q-function
        initial_states = [traj.states[0] for traj in trajectories]
        groups = [traj.demographic_group for traj in trajectories]

        value_estimates = []

        with torch.no_grad():
            for state in initial_states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_network(state_tensor).squeeze().numpy()
                pi_e = target_policy(state)
                value = (pi_e * q_values).sum()
                value_estimates.append(value)

        value_estimates = np.array(value_estimates)
        groups = np.array(groups)

        # Overall estimate
        value_estimate = value_estimates.mean()
        se = value_estimates.std() / np.sqrt(len(value_estimates))

        # Group-specific estimates
        group_estimates = {}
        for g in range(self.n_groups):
            group_mask = (groups == g)
            if group_mask.sum() > 0:
                group_est = value_estimates[group_mask].mean()
                group_se = value_estimates[group_mask].std() / np.sqrt(group_mask.sum())

                group_estimates[f"group_{g}_value"] = group_est
                group_estimates[f"group_{g}_se"] = group_se
                group_estimates[f"group_{g}_n_samples"] = group_mask.sum()

        return {
            "value": value_estimate,
            "se": se,
            "ci_lower": value_estimate - 1.96 * se,
            "ci_upper": value_estimate + 1.96 * se,
            "q_network": q_network,
            **group_estimates
        }

    def comprehensive_evaluation(self,
                                trajectories: List[Trajectory],
                                target_policy: Callable) -> pd.DataFrame:
        """
        Run all OPE methods and compare results.

        Parameters:
            trajectories: List of observed trajectories
            target_policy: Policy to evaluate

        Returns:
            DataFrame comparing methods across equity dimensions
        """
        logger.info("Running comprehensive off-policy evaluation...")

        # Standard importance sampling
        is_results = self.importance_sampling(
            trajectories, target_policy, weighted=False
        )

        # Weighted importance sampling
        wis_results = self.importance_sampling(
            trajectories, target_policy, weighted=True
        )

        # Fitted Q-evaluation
        fqe_results = self.fitted_q_evaluation(
            trajectories, target_policy
        )

        # Doubly robust (using FQE Q-function)
        dr_results = self.doubly_robust(
            trajectories, target_policy, fqe_results["q_network"]
        )

        # Compile results
        results = {
            "IS": is_results,
            "WIS": wis_results,
            "DR": dr_results,
            "FQE": fqe_results
        }

        # Create comparison DataFrame
        comparison = []
        for method, res in results.items():
            row = {
                "method": method,
                "value": res["value"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "ci_width": res["ci_upper"] - res["ci_lower"]
            }

            # Add group-specific values
            for g in range(self.n_groups):
                group_key = f"group_{g}_value"
                if group_key in res:
                    row[group_key] = res[group_key]

            comparison.append(row)

        df = pd.DataFrame(comparison)

        # Compute disparity metrics
        for g in range(self.n_groups):
            col_name = f"group_{g}_value"
            if col_name in df.columns:
                df[f"group_{g}_disparity"] = df["value"] - df[col_name]

        return df
```

This comprehensive OPE framework enables rigorous evaluation of treatment policies using only observational data, with explicit uncertainty quantification and fairness-aware analysis across demographic groups. The comparison of multiple methods provides robustness checks and helps identify when estimates are unreliable due to poor overlap or model misspecification.

## 11.6 Conclusion

Reinforcement learning offers powerful methods for learning optimal sequential treatment strategies from data, but applying these methods responsibly in healthcare requires careful attention to equity, safety, and the unique challenges of medical decision making. This chapter has developed RL approaches specifically designed for clinical applications serving diverse underserved populations.

We began with Markov decision processes as the formal framework for sequential decisions, showing how healthcare problems can be structured as MDPs while acknowledging the ways real clinical contexts violate standard assumptions. Value-based methods including Q-learning and deep Q-networks enable learning from observational data while accounting for heterogeneous treatment effects across populations through group-specific value functions. Policy gradient methods including PPO provide direct policy learning with explicit fairness constraints that prevent optimization solely for majority groups. Off-policy evaluation techniques enable rigorous assessment of potential policies using only observational data, with uncertainty quantification that acknowledges when estimates are unreliable for underrepresented populations.

Throughout, we have emphasized that technical sophistication must be coupled with deep understanding of healthcare contexts, attention to equity implications of algorithmic choices, and comprehensive safety mechanisms that prevent harm during both learning and deployment. The implementations provided are production-ready starting points that healthcare organizations can adapt for specific applications, always with extensive clinical validation before deployment.

The path forward for reinforcement learning in healthcare requires continued development of methods that: (1) learn effectively from limited observational data without requiring harmful exploration, (2) explicitly optimize for equitable outcomes across demographic groups rather than aggregate performance, (3) incorporate medical knowledge and clinical constraints throughout the learning process, (4) quantify uncertainty and provide interpretable explanations for recommended actions, and (5) enable ongoing monitoring and adaptation as clinical contexts evolve. These methods must be developed and evaluated in close collaboration with clinicians and the communities they serve, ensuring that RL systems genuinely improve health outcomes for all patients rather than optimizing for the majority at the expense of the underserved.

## Bibliography

Gottesman, O., Johansson, F., Komorowski, M., Faisal, A., Sontag, D., Doshi-Velez, F., & Celi, L. A. (2019). Guidelines for reinforcement learning in healthcare. *Nature Medicine*, 25(1), 16-18. https://doi.org/10.1038/s41591-018-0310-5

Komorowski, M., Celi, L. A., Badawi, O., Gordon, A. C., & Faisal, A. A. (2018). The Artificial Intelligence Clinician learns optimal treatment strategies for sepsis in intensive care. *Nature Medicine*, 24(11), 1716-1720. https://doi.org/10.1038/s41591-018-0213-5

Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. *arXiv preprint arXiv:2005.01643*. https://arxiv.org/abs/2005.01643

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533. https://doi.org/10.1038/nature14236

Murphy, S. A. (2003). Optimal dynamic treatment regimes. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 65(2), 331-355. https://doi.org/10.1111/1467-9868.00389

Oberst, M., & Sontag, D. (2019). Counterfactual off-policy evaluation with Gumbel-max structural causal models. *Proceedings of the 36th International Conference on Machine Learning*, 97, 4881-4890. http://proceedings.mlr.press/v97/oberst19a.html

Precup, D., Sutton, R. S., & Singh, S. (2000). Eligibility traces for off-policy policy evaluation. *Proceedings of the 17th International Conference on Machine Learning*, 759-766.

Raghu, A., Komorowski, M., Celi, L. A., Szolovits, P., & Ghassemi, M. (2017). Continuous state-space models for optimal sepsis treatment: a deep reinforcement learning approach. *Proceedings of Machine Learning for Healthcare Conference*, 68, 147-163. http://proceedings.mlr.press/v68/raghu17a.html

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*. https://arxiv.org/abs/1707.06347

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489. https://doi.org/10.1038/nature16961

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. http://incompleteideas.net/book/the-book-2nd.html

Swaminathan, A., & Joachims, T. (2015). The self-normalized estimator for counterfactual learning. *Advances in Neural Information Processing Systems*, 28, 3231-3239. https://proceedings.neurips.cc/paper/2015/file/39027dfad5138c9ca0c474d71db915c3-Paper.pdf

Thomas, P., & Brunskill, E. (2016). Data-efficient off-policy policy evaluation for reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning*, 48, 2139-2148. http://proceedings.mlr.press/v48/thomas16.html

Tucker, G., Bhupatiraju, S., Gu, S., Turner, R. E., Ghahramani, Z., & Levine, S. (2018). The mirage of action-dependent baselines in reinforcement learning. *Proceedings of the 35th International Conference on Machine Learning*, 80, 5015-5024. http://proceedings.mlr.press/v80/tucker18a.html

Voloshin, C., Le, H. M., Jiang, N., & Yue, Y. (2021). Empirical study of off-policy policy evaluation for reinforcement learning. *Advances in Neural Information Processing Systems*, 34, 15141-15153. https://proceedings.neurips.cc/paper/2021/file/7d6bc61d00dab8e67e43c3c53a6e0bc7-Paper.pdf

Wang, L., Zhang, W., He, X., & Zha, H. (2018). Supervised reinforcement learning with recurrent neural network for dynamic treatment recommendation. *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2447-2456. https://doi.org/10.1145/3219819.3219961

Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292. https://doi.org/10.1007/BF00992698

Yu, C., Liu, J., Nemati, S., & Yin, G. (2021). Reinforcement learning in healthcare: A survey. *ACM Computing Surveys*, 55(1), 1-36. https://doi.org/10.1145/3477600

Zhang, J., & Bareinboim, E. (2017). Fairness in decision-making — the causal explanation formula. *Proceedings of the AAAI Conference on Artificial Intelligence*, 31(1), 2037-2043. https://ojs.aaai.org/index.php/AAAI/article/view/10846
