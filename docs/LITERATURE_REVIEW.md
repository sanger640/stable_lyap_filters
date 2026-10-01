# Literature review: sensitivity-based ("intuitive physics engine") safety monitoring for manipulation

Prepared 2026-10-01. Scope: the idea of giving a robot the Battaglia, Hamrick & Tenenbaum (2013) "intuitive physics engine" ability, implemented as a runtime monitor that samples realistic execution noise (and optionally perception noise) around an intended action chunk, rolls the samples forward, and alarms when the futures split into persistent, well-separated groups.

## How to read this file

- Each entry gives authors, title, venue, year, a link, and one or two sentences on why it matters here.
- **Verification.** Every entry was checked against a live web source (arXiv abstract page, proceedings page, publisher page or author page) during this review. Where I only saw the abstract, and a claim depends on details in the body, I say so. Entries marked **[abstract only]** were checked at the abstract level. Entries marked **[venue unverified]** exist, but I could not confirm where (or whether) they were formally published.
- Several 2026 arXiv preprints appear below. They are recent, unrefereed, and may change.
- "The project" means the monitor described in the brief: about 64 action samples drawn from measured execution residuals, optional perception-error sampling, exact (MuJoCo) or learned rollouts, and a two-group split test (BIC, Ashman's D > 2, minority >= 2, the same split at two hold times).

---

## 1. Cognitive science of intuitive physics

### 1.1 The simulation account

- **Battaglia, Hamrick & Tenenbaum. "Simulation as an engine of physical scene understanding." PNAS 110(45):18327-18332, 2013.** https://www.pnas.org/doi/10.1073/pnas.1306572110 (open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC3831455)
  Introduces the intuitive physics engine (IPE): a few approximate, probabilistic simulations over an uncertain scene (perceptual noise on positions, noise on forces and mass), with judgments read from the distribution of outcomes. One model fits five tasks, including "will it fall" and "which direction". This is the project's direct inspiration. Note that the IPE's judgments are always read out through a **query-specific outcome feature** (e.g. the proportion of blocks that fell). They are not read from a structure-only split test.
- **Hamrick, Battaglia & Tenenbaum. "Internal physics models guide probabilistic judgments about object dynamics." CogSci 2011.** https://dspace.mit.edu/handle/1721.1/77012
  Earlier tower-stability experiments in which human graded stability judgments track a sampling-based simulation observer. This is the first quantitative "noisy simulation predicts how precarious people think a tower is" result.
- **Hamrick, Battaglia, Griffiths & Tenenbaum. "Inferring mass in complex scenes by mental simulation." Cognition 157:61-76, 2016.** https://collaborate.princeton.edu/en/publications/inferring-mass-in-complex-scenes-by-mental-simulation/
  Extends the IPE from prediction to inference (inferring mass from observed collisions). It shows that the same noisy simulator supports both forward prediction and inverse inference.
- **Smith & Vul. "Sources of uncertainty in intuitive physics." Topics in Cognitive Science 5(1):185-199, 2013.** https://onlinelibrary.wiley.com/doi/10.1111/tops.12009
  Separates perceptual noise in initial conditions from stochasticity in the dynamics, and finds that both are needed to fit human predictions. This maps directly onto the project's two noise sources: perception-state sampling and execution-noise sampling.
- **Sanborn, Mansinghka & Griffiths. "Reconciling intuitive physics and Newtonian mechanics for colliding objects." Psychological Review 120(2):411-437, 2013.** https://wrap.warwick.ac.uk/54487/
  The "noisy Newton" account: apparent biases come from optimal inference over Newtonian physics with noisy perception. It supports the idea that a roughly right model plus well-modelled uncertainty is enough.
- **Ullman, Spelke, Battaglia & Tenenbaum. "Mind games: game engines as an architecture for intuitive physics." Trends in Cognitive Sciences 21(9):649-665, 2017.** https://pubmed.ncbi.nlm.nih.gov/28655498/
  Argues that the mental model resembles a game engine: approximate, fast, and good enough. This is the conceptual justification for "you don't need a millimetre-exact model".
- **Kubricht, Holyoak & Lu. "Intuitive physics: current research and controversies." Trends in Cognitive Sciences 21(10):749-759, 2017.** https://reasoninglab.psych.ucla.edu/wp-content/uploads/sites/273/2021/04/Kubrich_etal.TICS_.2017.pdf
  Review of heuristic-versus-simulation accounts. Concludes that simulation follows normative physics but is biased by perceptual uncertainty and priors. A good single citation for the state of the debate.
- **Hamrick, Smith, Griffiths & Vul. "Think again? The amount of mental simulation tracks uncertainty in the outcome." CogSci 2015.** https://escholarship.org/uc/item/5px8949q
  People run more simulations (via a sequential probability ratio test) when outcomes are uncertain. This is relevant to an adaptive sample budget: run few probes in quiet states and more near forks.
- **Allen, Smith & Tenenbaum. "Rapid trial-and-error learning with simulation supports flexible tool use and physical reasoning." PNAS 117(47), 2020.** https://pmc.ncbi.nlm.nih.gov/articles/PMC7703630 (arXiv 1907.09620)
  The SSUP (sample, simulate, update) model and the Virtual Tools game: noisy simulation of candidate actions, used for action selection. It is action-conditioned like the project, but it is goal-driven (it scores reward), not a goal-free alarm.
- **Gerstenberg, Goodman, Lagnado & Tenenbaum. "A counterfactual simulation model of causal judgments for physical events." Psychological Review 128(5), 2021.** https://cicl.stanford.edu/publication/gerstenberg2021csm
  The counterfactual simulation model (CSM): causal judgments come from comparing actual outcomes with noisy counterfactual simulations, including "robustness" (does the outcome survive small perturbations?). The project's "do nearby realisations have different consequences?" is close to the CSM's robustness and sufficiency aspects.
- **Zhou, Smith, Tenenbaum & Gerstenberg. "Mental Jenga: A counterfactual simulation model of causal judgments about physical support." Journal of Experimental Psychology: General, 2023.** https://cicl.stanford.edu/publication/zhou2023jenga
  People judge whether a block supports a tower by noisily simulating its removal. This is the cognitive-science twin of the project's Jenga-like pick/push setting and should be cited.
- **Bates, Yildirim, Tenenbaum & Battaglia. "Modeling human intuitions about liquid flow with particle-based simulation." PLOS Computational Biology, 2019.** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6675131/
  Coarse particle simulation predicts human fluid judgments better than heuristics or deep nets. Evidence that approximate simulators preserve the judgment-relevant structure.
- **Yildirim, Saeed, Bennett-Pierre, Gerstenberg, Tenenbaum & Gweon. "Explaining intuitive difficulty judgments by modeling physical effort and risk." arXiv 1905.04445, 2019 (CogSci 2019 per author pages; [venue unverified]).** https://arxiv.org/abs/1905.04445
  Defines "physical risk" of a construction as the instability measured through noisy physics simulations. This is very close in spirit to a precariousness score, but it is a cognitive model of human difficulty ratings and uses a task-defined stability outcome.
- **Zhang, Wu, Zhang, Freeman & Tenenbaum. "A comparative evaluation of approximate probabilistic simulation and deep neural networks as accounts of human physical scene understanding." CogSci 2016.** https://escholarship.org/uc/item/4bd5068b (arXiv 1605.01138)
  Simulation models and CNNs both reach high accuracy on "will it fall", but only simulation generalises like humans and reproduces their illusions. This is directly relevant to "learned predictor versus simulator" in section 2 and the learned-world-model findings.
- **Hamrick. "Analogues of mental simulation and imagination in deep learning." Current Opinion in Behavioral Sciences 29:8-16, 2019.** https://openreview.net/pdf/92f0fd3bd354f93e3a24aa0fe4cb4a49b48c996c.pdf
  Maps mental simulation onto model-based deep RL. This is the natural bridge citation between the IPE and learned world models.

### 1.2 Evidence against, or limits of, the simulation account

- **Davis & Marcus. "The scope and limits of simulation in automated reasoning." Artificial Intelligence 233:60-72, 2016.** https://arxiv.org/abs/1506.04956
  Simulation works best when the task is prediction, information is complete, the theory is good, and scales are not extreme. They list twelve problem features that defeat simulation. A reviewer could invoke this against the project: forks are exactly where small model errors matter most.
- **Marcus & Davis. "How robust are probabilistic models of higher-level cognition?" Psychological Science 24(12):2351-2360, 2013.** https://www.psychologicalscience.org/journals/psychological-science/0956797613495418/
  Argues that probabilistic-cognition fits, including intuitive physics, depend on post hoc choices of noise and priors. This is relevant because the project also hand-chooses noise models and split-test settings.
- **Ludwin-Peery, Bramley, Davis & Gureckis. "Broken physics: A conjunction-fallacy effect in intuitive physical reasoning." Psychological Science 31(12):1602-1611, 2020.** https://www.gureckislab.org/papers/ref/ludwinpeery2020broken
  People rate conjunctions of physical events as more likely than single events, which pure Monte Carlo simulation cannot produce. Main empirical challenge to the IPE.
- **Bass, Smith, Bonawitz & Ullman. "Partial mental simulation explains fallacies in physical reasoning." Cognitive Neuropsychology 38(7-8):413-424, 2022 (issue dated 2021).** https://www.mit.edu/~k2smith/publication/partial_simulation
  Reply to the above: people simulate only the relevant parts of the scene. This supports simulating only local, contact-relevant objects, which is a useful efficiency argument for a robot monitor.

### 1.3 Human-versus-model physical prediction benchmarks

- **Bear, Wang, Mrowca, Binder, Tung, et al. "Physion: Evaluating physical prediction from vision in humans and machines." NeurIPS 2021 Datasets & Benchmarks.** https://arxiv.org/abs/2106.08261
  GNNs on true physical state best match humans. Vision-only models fall well short, and object-centric and pretrained models do best among them. This parallels the project's finding that perception precision, not dynamics, can be the bottleneck.
- Further benchmarks (IntPhys, IntPhys 2, Physics-IQ, PHYRE) are covered in sections 2 and 6.

---

## 2. Learned stability and physical-intuition predictors (vision and robotics)

- **Lerer, Gross & Fergus. "Learning physical intuition of block towers by example." ICML 2016 (PMLR 48).** https://proceedings.mlr.press/v48/lerer16.html
  PhysNet: CNNs trained on simulated block towers predict collapse and trajectories, and transfer to real images at roughly human accuracy. The canonical learned stability classifier and a natural baseline.
- **Li, Azimi, Leonardis & Fritz. "To fall or not to fall: A visual approach to physical stability prediction." arXiv 1604.00066, 2016.** https://arxiv.org/abs/1604.00066
  End-to-end visual stability prediction compared against human judgments.
- **Li, Leonardis & Fritz. "Visual stability prediction for robotic manipulation." ICRA 2017.** https://arxiv.org/abs/1609.04861
  Uses the learned stability predictor to guide stacking. This is the closest "learned precariousness predictor used for manipulation" baseline.
- **Groth, Fuchs, Posner & Vedaldi. "ShapeStacks: Learning vision-based physical intuition for generalised object stacking." ECCV 2018.** https://arxiv.org/abs/1804.08018
  A 20k-stack dataset with stability labels. The classifiers transfer to real images and support active stacking. This is a supervised, label-dependent alternative to the project's structural alarm.
- **Wu, Lu, Kohli, Freeman & Tenenbaum. "Learning to see physics via visual de-animation." NeurIPS 2017.** https://papers.nips.cc/paper/2017/file/4c56ff4ce4aaf9573aa5dff913df997a-Paper.pdf
  Perception inverts images into a physical state that is fed to a physics engine, with stability-type predictions. The "perception to explicit state to simulator" pipeline is the same architecture as the project's uncertainty-aware variant.
- **Battaglia, Pascanu, Lai, Rezende & Kavukcuoglu. "Interaction networks for learning about objects, relations and physics." NeurIPS 2016.** https://arxiv.org/abs/1612.00222
  The first general learnable object-relational physics engine. It is the ancestor of the project's GNN world models (D2-D21).
- **Bakhtin, van der Maaten, Johnson, Gustafson & Girshick. "PHYRE: A new benchmark for physical reasoning." NeurIPS 2019.** https://papers.nips.cc/paper/8752-phyre-a-new-benchmark-for-physical-reasoning
  A 2D action-selection puzzle benchmark in which sensitivity to precise action placement is central. Learned methods are sample-inefficient on it.
- **Riochet et al. "IntPhys: A framework and benchmark for visual intuitive physics reasoning." arXiv 1803.07616, 2018.** https://arxiv.org/abs/1803.07616
  Violation-of-expectation evaluation (possible versus impossible videos).
- **Bordes, Garrido, Kao, Williams, Rabbat & Dupoux. "IntPhys 2: Benchmarking intuitive physics understanding in complex synthetic environments." arXiv 2506.09849, 2025.** https://arxiv.org/abs/2506.09849
  Most current models are near chance on permanence, immutability, continuity and solidity. These are qualitative principles, not branching or sensitivity.
- **Puthuveetil, Zhang, Yokoyama & Narita. "Strategic Jenga play via graph based dynamics modeling." arXiv 2505.09377, 2025 (ICRA 2025 workshop spotlight).** https://arxiv.org/abs/2505.09377
  A graph classifier predicts whether removing a block causes collapse, and a learned graph dynamics model with sampling-based MPC extracts blocks in Isaac Lab. This is the closest **robotic Jenga + GNN dynamics** prior. It uses collapse labels and task costs.
- **Fazeli, Oller, Wu, Wu, Tenenbaum & Rodriguez. "See, feel, act: Hierarchical learning for complex manipulation skills with multisensory fusion." Science Robotics 4(26), 2019.** https://dspace.mit.edu/handle/1721.1/126656
  A real robot plays Jenga using a hierarchical Bayesian model over noisy block states and force/vision. Cite it as the real-world Jenga manipulation reference.
- **Mojtahedzadeh et al. "Support relation analysis and decision making for safe robotic manipulation tasks." Robotics and Autonomous Systems, 2015.** https://portal.fis.tum.de/en/publications/support-relation-analysis-and-decision-making-for-safe-robotic-ma/ **[abstract only; year/venue from the TUM portal entry]**
  Static-equilibrium support analysis decides which object to remove from a pile safely. This is a geometric and mechanical alternative to simulation-based precariousness.

---

## 3. Physics simulation under uncertainty for manipulation planning and robustness

For each entry, the **success/safety definition** it needs is stated explicitly, because that is the main axis on which the project differs.

- **Dogar & Srinivasa. "Push-grasping with dexterous hands: mechanics and a method." IROS 2010.** https://ri.cmu.edu/publications/push-grasping-with-dexterous-hands-mechanics-and-a-method
  Quasi-static simulation over object-pose uncertainty computes "capture regions" for robust push-grasps. *Needs:* the success predicate "object ends inside the hand".
- **Dogar, Hsiao, Ciocarlie & Srinivasa. "Physics-based grasp planning through clutter." RSS 2012.** https://www.roboticsproceedings.org/rss08/p08.html
  Plans reaches that push clutter aside, using physics prediction to avoid undesirable consequences. *Needs:* a target-grasp success predicate plus rules about acceptable displacements of other objects.
- **Koval, Pollard & Srinivasa. "Pre- and post-contact policy decomposition for planar contact manipulation under uncertainty." RSS 2014 / IJRR 2015.** https://roboticsproceedings.org/rss10/p34.html
  A POMDP with a physics-simulator transition model for pushing under pose uncertainty. *Needs:* a reward for a successful grasp.
- **King, Ranganeni & Srinivasa. "Unobservable Monte Carlo planning for nonprehensile rearrangement tasks." ICRA 2017.** https://ri.cmu.edu/publications/unobservable-monte-carlo-planning-for-nonprehensile-rearrangement-tasks
  MCTS over open-loop trajectories with physics rollouts under uncertainty. *Needs:* a goal region for the target object.
- **Agboh & Dogar. "Pushing fast and slow: Task-adaptive planning for non-prehensile manipulation under uncertainty." WAFR 2018 (Springer 2020).** https://arxiv.org/abs/1805.03005
  An MDP with action-dependent stochasticity: the robot slows down when the task demands accuracy. This is the closest robotics analogue to "act carefully when sensitive". *Needs:* a task accuracy requirement (goal tolerance). It does not detect sensitivity without one.
- **Muhayyuddin, Moll, Kavraki & Rosell. "Randomized physics-based motion planning for grasping in cluttered and uncertain environments." RA-L 3(2):712-719, 2018.** https://arxiv.org/abs/1711.09604
  A kinodynamic planner (p-KPIECE) with a physics engine as validity checker under pose and contact uncertainty. *Needs:* validity and goal predicates.
- **Weisz & Allen. "Pose error robust grasping from contact wrench space metrics." ICRA 2012.** https://www.cs.columbia.edu/~allen/PAPERS/weisz_icra12.pdf
  Re-ranks grasps by how stable their quality metric is under pose error. This is an early "sensitivity of an outcome to perturbation" metric. *Needs:* a force-closure quality metric.
- **Mahler et al. "Dex-Net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics." RSS 2017.** https://arxiv.org/abs/1703.09312
  Robust grasp quality is the expected analytic quality under sampled pose, friction and gripper perturbations, distilled into a CNN. *Needs:* a binary grasp-success metric. It is the textbook "simulation-ensemble risk score with a success label".
- **Tobin et al. "Domain randomization for transferring deep neural networks from simulation to the real world." IROS 2017.** https://arxiv.org/abs/1703.06907
  Visual domain randomization. Cite for the general idea of sampling a distribution of worlds.
- **Peng, Andrychowicz, Zaremba & Abbeel. "Sim-to-real transfer of robotic control with dynamics randomization." ICRA 2018.** https://arxiv.org/abs/1710.06537
  Dynamics randomization for robust pushing policies. *Needs:* a task reward.
- **Rajeswaran, Ghotra, Ravindran & Levine. "EPOpt: Learning robust neural network policies using model ensembles." ICLR 2017.** https://arxiv.org/abs/1610.01283
  Trains on the worst-performing members of a simulator ensemble (CVaR). *Needs:* a reward.
- **Ramos, Possas & Fox. "BayesSim: adaptive domain randomization via probabilistic inference for robotics simulators." RSS 2019.** https://arxiv.org/abs/1906.01728
  A likelihood-free posterior over simulator parameters from real data. It is the principled way to obtain the noise distributions that the project currently measures empirically.
- **Li et al. "PIN-WM: Learning physics-informed world models for non-prehensile manipulation." RSS 2025.** https://arxiv.org/abs/2504.16693
  Real2sim with differentiable physics and Gaussian splatting, then "digital cousins" via physics-aware randomization. *Needs:* a task reward for policy learning. It is relevant as a physics-structured world model that might preserve branching better than generic GNNs.
- **Zapolsky & Drumwright. "Particle traces for detecting divergent robot behavior." arXiv 1608.01606, 2016 (submitted to Humanoids 2016; [venue unverified]).** https://arxiv.org/abs/1608.01606 **[abstract only]**
  Samples model parameters, sensor readings and state estimates, runs fast multibody simulation, and statistically flags when behaviour diverges from the plan. It also locates **bifurcating events** (e.g. falls clustering near grazing bifurcations). Tested on locomotion, fixed-base picking and passive walkers. **This is one of the closest mechanistic precedents to the project** (sampling plus simulation to locate bifurcations). From the abstract, the divergence is measured against an expected plan or behaviour. I did not verify the details of their divergence test in the body.
- **Johansson, Lindmark, Wiberg & Servin. "Safety filtering of robotic manipulation under environment uncertainty: a computational approach." arXiv 2509.12674, 2025.** https://arxiv.org/abs/2509.12674
  A physics-based safety filter: dense nominal rollouts, sparse re-evaluation under sampled mass and friction at critical transitions, and probing actions to reduce uncertainty. *Needs:* explicit "generalized factors of safety" for grasp stability and actuator limits.
- **Wang, Ren, Morgan & Hang. "ManiDreams: An open-source library for robust object manipulation via uncertainty-aware task-specific intuitive physics." arXiv 2603.18336, 2026.** https://arxiv.org/abs/2603.18336
  Wraps any base policy in a sample-predict-constrain loop over a "domain-randomized instance set" of particles that covers perceptual (pose hypotheses), parametric (mass, friction) and structural (simulator versus learned diffusion backend) uncertainty. **Very close in machinery and in its explicit "intuitive physics" framing.** *Needs:* task-specific declarative constraints. It is a planner wrapper, not a goal-free alarm.
- **Wang, Ren, Morgan & Hang. "Caging in time: A framework for robust object manipulation under uncertainties and limited robot perception." IJRR 44(14), 2025.** https://arxiv.org/abs/2410.16481
  The same group's earlier robust-manipulation-under-uncertainty work (open-loop caging). Background only.
- **Esteban, Li, Kurtz & Ames. "On surprising effects of risk-aware domain randomization for contact-rich sampling-based predictive control." arXiv 2605.03290, 2026.** https://arxiv.org/abs/2605.03290
  Aggregating sampling-MPC costs across randomized models (average, optimistic, pessimistic) reshapes the cost landscape on Push-T. *Needs:* a task cost. It is relevant as recent evidence that how you aggregate over sampled worlds matters.
- **Luo, Zhao, Kuck, Ivanovic, Savarese, Schmerling & Pavone. "Sample-efficient safety assurances using conformal prediction." WAFR 2022 / IJRR 2024.** https://arxiv.org/abs/2109.14082
  Uses a simulator plus conformal prediction to tune warning systems with a guaranteed false-negative rate (driver warning, grasping). *Needs:* labelled unsafe outcomes. It is the right reference if the project ever wants a guarantee.

Real-world stochasticity of the modelled interactions (why execution noise matters):

- **Yu, Bauza, Fazeli & Rodriguez. "More than a million ways to be pushed: A high-fidelity experimental dataset of planar pushing." IROS 2016.** https://dspace.mit.edu/handle/1721.1/109116
  Real pushing is measurably variable and friction is not repeatable. This supports modelling outcome distributions rather than point predictions.
- **Acosta, Yang & Posa. "Validating robotics simulators on real-world impacts." RA-L 2022.** https://arxiv.org/abs/2110.00541
  Drake, MuJoCo and Bullet capture inelastic impacts but not elastic ones. This is a caution against treating MuJoCo forks as ground truth for real forks.

---

## 4. Runtime safety monitoring and failure prediction for robot policies

- **Hsu, Hu & Fisac. "The safety filter: A unified view of safety-critical control in autonomous systems." Annual Review of Control, Robotics, and Autonomous Systems, 2024.** https://arxiv.org/abs/2309.05837
  The framework for monitor-plus-fallback filters. Every filter in it assumes a specified failure set. Useful for positioning the project as a filter with no failure set.
- **Nakamura, Peters & Bajcsy. "Generalizing safety beyond collision-avoidance via latent-space reachability analysis." RSS 2025.** https://arxiv.org/abs/2502.00935
  Latent safety filters: HJ reachability in a generative world model's latent space, preventing spills and toppling on a Franka. *Needs:* a failure classifier learned from labelled failure data.
- **Seo, Nakamura & Bajcsy. "Uncertainty-aware latent safety filters for avoiding out-of-distribution failures." CoRL 2025 (PMLR 305).** https://arxiv.org/abs/2505.00779
  UNISafe: augments the latent state with world-model **epistemic** uncertainty (ensemble disagreement, Jensen-Rényi divergence) and calibrates the OOD threshold conformally. Tasks include **vision-based block plucking from a stack (IsaacLab) and real Jenga extraction on a Franka FR3**. **Same task family as the project.** *Needs:* a labelled failure classifier, plus calibration data for the uncertainty threshold. The uncertainty it uses is epistemic (model ignorance), not aleatoric branching of the physics.
- **Authors from Ohio State, CMU, UC Berkeley and NVIDIA [names not checked]. "AnySafe: Adapting latent safety filters at runtime via safety constraint parameterization in the latent space." arXiv 2509.19555, 2025.** https://arxiv.org/abs/2509.19555 **[abstract only; author list not checked]**
  Constraints are specified at runtime by a constraint image and calibrated with conformal prediction. It still needs a constraint specification.
- **TUM Learning Systems & Robotics Lab authors. "Failure prediction at runtime for generative robot policies." NeurIPS 2025 (FIPER).** https://arxiv.org/abs/2510.09459 **[author names not checked; take them from the arXiv page before citing]**
  Label-free with respect to failures: it uses OOD detection (random network distillation) plus an **action-chunk entropy** score, conformally calibrated on successful rollouts. It is the closest "no failure data" monitor for action-chunk policies. It measures the *policy's* uncertainty over actions, not the *physics'* sensitivity to realised actions. It still needs success rollouts for calibration.
- **Agia, Sinha, Yang, Cao, Antonova, Pavone & Bohg. "Unpacking failure modes of generative policies: Runtime monitoring of consistency and progress." CoRL 2024 (PMLR 270).** https://arxiv.org/abs/2410.04640
  Sentinel: temporal action-consistency statistics detect erratic failures, and a VLM detects task-progress failures. Its consistency detector is also label-light, but it is policy-side.
- **Xu, Nguyen, Dixon, ..., Nishimura & Itkina. "Can we detect failures without failure data? Uncertainty-aware runtime failure detection for imitation learning policies." RSS 2025.** https://arxiv.org/abs/2503.08558
  FAIL-Detect: sequential OOD detection on policy inputs and outputs with conformal thresholds from successful data.
- **Liu, Dass, Martín-Martín & Zhu. "Model-based runtime monitoring with interactive imitation learning." ICRA 2024.** https://arxiv.org/abs/2310.17552
  A latent dynamics model plus a failure classifier "imagines" future outcomes to flag high-risk states. *Needs:* failure labels from human interventions.
- **Sinha, Elhafsi, Agia, Foutter, Schmerling & Pavone. "Real-time anomaly detection and reactive planning with large language models." RSS 2024.** https://arxiv.org/abs/2407.08735
  AESOP: embedding-based anomaly detection plus LLM reasoning for fallback. A semantic OOD monitor, orthogonal to physical sensitivity.
- **Ahmed & Begum. "TAIL-Safe: Task-agnostic safety monitoring for imitation learning policies." arXiv 2605.01195, 2026.** https://arxiv.org/abs/2605.01195
  Despite the name, its "task-agnostic" safety is a Lipschitz Q-function over visibility, recognizability and graspability, trained with failure data from a Gaussian-splat digital twin, with Nagumo-style recovery. It still uses learned safety labels. Expect reviewers to bring it up because of the "task-agnostic" wording.
- **Zheng, Johnson-Roberson & Zhi. "ContactGuard: Pre-contact execution monitoring with action-conditioned latent world models." arXiv 2608.13438, 2026.** https://arxiv.org/abs/2608.13438
  Rolls an action-conditioned latent world model forward under the planned actions and aborts before contact. *Needs:* a failure probe trained on a small labelled set. It does not sample action perturbations.
- **Liu, Sun, Chao, et al. "CheckVLA: Execution-time verification with action-conditioned world model for long-horizon mobile manipulation." arXiv 2607.26789, 2026.** https://arxiv.org/abs/2607.26789
  World-model prediction versus execution, with conformally calibrated risk thresholds. *Needs:* calibration data.
- **Lee & Har. "Perturbation-based epistemic uncertainty for failure detection in vision-language-action models." arXiv 2606.20754, 2026.** https://arxiv.org/abs/2606.20754 **[abstract only]**
  Perturbs *model weights* (low-rank) and measures action disagreement. It shares the word "perturbation" with the project but perturbs something different.
- **UT Austin authors, including S. Ambatipudi and A. Balaji [author order not checked]. "OopsieVerse: A safety benchmark with damage-aware simulation for robot manipulation." RSS 2026 / arXiv 2606.31993.** https://arxiv.org/abs/2606.31993 **[author list abbreviated]**
  DamageSim converts contact forces, heat and liquids into task-agnostic damage signals in OmniGibson and RoboCasa (MuJoCo). It is "task-agnostic", but it requires a physical damage model. It is a natural evaluation partner (does a fork alarm predict damage?) and a contrasting definition of task-agnostic safety.

---

## 5. Sensitivity, chaos, basins and boundary detection

- **Menck, Heitzig, Marwan & Kurths. "How basin stability complements the linear-stability paradigm." Nature Physics 9:89-92, 2013.** https://publications.pik-potsdam.de/pubman/item/item_18411
  Basin stability is the fraction of random perturbations that return to the same attractor, estimated by Monte Carlo. **The project's alarm is essentially a Monte Carlo estimate of "this action sits near a basin boundary"** (minority fraction > 0 and well separated). Cite it as the formal concept, and consider reporting minority fraction as a basin-stability number.
- **Shadden, Lekien & Marsden. "Definition and properties of Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows." Physica D 212, 2005.** https://authors.library.caltech.edu/records/0ytty-xq745
  FTLE ridges are separatrices between regions of qualitatively different fate. The project's split test is a finite-sample, non-differential detector of the same thing (a separatrix crossing inside the noise ball). FTLE-style finite-difference sensitivity is an obvious baseline (section 10).
- **Scheffer et al. "Early-warning signals for critical transitions." Nature 461:53-59, 2009.** https://ideas.repec.org/a/nat/nature/v461y2009i7260d10.1038_nature08227.html
  Generic, system-agnostic warnings (variance, autocorrelation, skew) before tipping points. It is the intellectual precedent for "structural, label-free warnings". The difference is that critical slowing down applies to smooth bifurcations, while contact forks are non-smooth.
- **Fadini & Coros. "Enhancing robotic system robustness via Lyapunov exponent-based optimization." arXiv 2412.06776, 2024.** https://arxiv.org/abs/2412.06776
  Lyapunov exponents via differentiable simulation as a robustness objective for robots. The authors note that contact switching makes this hard, which motivates the project's sampling-based, non-differential test.
- **Burridge, Rizzi & Koditschek. "Sequential composition of dynamically dexterous robot behaviors." IJRR 18(6):534-555, 1999.** https://repository.upenn.edu/handle/20.500.14332/33914
  Funnels (basins of attraction of controllers) as the unit of robust robot behaviour. The classical "stay inside the basin" framing.
- **Fawzi, Moosavi-Dezfooli & Frossard. "Robustness of classifiers: from adversarial to random noise." NeurIPS 2016.** https://arxiv.org/abs/1608.08967
  Random-noise robustness depends on distance to, and curvature of, the decision boundary. This is the theory for why a fixed noise ball detects boundaries only within roughly one noise-scale. It explains the project's perception result: misperception of about the execution-noise size moves the ball off the fork.
- **Block, Foster, Krishnamurthy, Simchowitz & Zhang. "Butterfly effects of SGD noise: Error amplification in behavior cloning and autoregression." ICLR 2024.** https://arxiv.org/abs/2310.11428
  Tiny training-noise differences are amplified chaotically through closed-loop dynamics. Evidence that action-sensitivity is real and consequential in imitation learning.
- **Ashman, Bird & Zepf. "Detecting bimodality in astronomical datasets." Astronomical Journal 108:2348-2361, 1994.** https://arxiv.org/abs/astro-ph/9408030
  The source of the KMM bimodality test and the separation statistic now called Ashman's D (D > 2 for clean separation). Cite it for the split test.

---

## 6. World models and counterfactual (action-conditioned branching) fidelity

### 6.1 Learned simulators and world models the project tested or should mention

- **Sanchez-Gonzalez, Godwin, Pfaff, Ying, Leskovec & Battaglia. "Learning to simulate complex physics with graph networks." ICML 2020.** https://arxiv.org/abs/2002.09405
  GNS: particle and graph message-passing simulators trained on one-step losses.
- **Allen, Lopez-Guevara, Rubanova, Stachenfeld, Sanchez-Gonzalez, Battaglia & Pfaff. "Graph network simulators can learn discontinuous, rigid contact dynamics." CoRL 2022 (PMLR 205).** https://proceedings.mlr.press/v205/allen23a.html
  GNS fits real cube-toss contact discontinuities better than tuned analytic simulators. A reviewer could cite this against the project's learned-model negative result. The answer is that trajectory accuracy is not branching fidelity.
- **Allen, Rubanova, Lopez-Guevara, Whitney, Sanchez-Gonzalez, Battaglia & Pfaff. "Learning rigid dynamics with face interaction graph networks." ICLR 2023.** https://arxiv.org/abs/2212.03574
  FIGNet: mesh-face interactions, about 4x more accurate on complex shapes. The project's FIGNet* is a re-implementation; say so.
- **Lopez-Guevara, Rubanova, Whitney, Pfaff, Stachenfeld & Allen. "Scaling face interaction graph networks to real world scenes." arXiv 2401.11985, 2024.** https://arxiv.org/abs/2401.11985
  FIGNet on NeRF-derived real scenes, i.e. learned simulators driven by perception, which is the project's uncertainty-aware setting.
- **Rubanova, Lopez-Guevara, Allen, Whitney, Stachenfeld & Pfaff. "Learning rigid-body simulators over implicit shapes for large-scale scenes and vision." NeurIPS 2024.** https://arxiv.org/abs/2405.14045
  SDF-Sim scales learned rigid simulation to hundreds of objects and to vision-extracted SDFs.
- **Parmar, Halm & Posa. "Fundamental challenges in deep learning for stiff contact dynamics." IROS 2021.** https://arxiv.org/abs/2103.15406
  Stiffness alone badly degrades learned contact models. **Key mechanistic citation for why smooth learned models wash out forks**: a smooth regressor averaging across a discontinuity produces the mean outcome, not two modes.
- **Chua, Calandra, McAllister & Levine. "Deep reinforcement learning in a handful of trials using probabilistic dynamics models." NeurIPS 2018.** https://arxiv.org/abs/1805.12114
  PETS: probabilistic ensembles with trajectory sampling, i.e. aleatoric plus epistemic outcome spread. The standard way to make a learned model produce spread, and an obvious "stochastic world model" control.
- **Lakshminarayanan, Pritzel & Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." NeurIPS 2017.** https://arxiv.org/abs/1612.01474
  Deep ensembles, the uncertainty mechanism in UNISafe and PETS.
- **Zhou, Pan, LeCun & Pinto. "DINO-WM: World models on pre-trained visual features enable zero-shot planning." ICML 2025.** https://arxiv.org/abs/2411.04983
  A latent dynamics model on frozen DINOv2 patches, including pushing tasks. A natural frozen-encoder world model to test for branching fidelity.
- **Assran et al. "V-JEPA 2: Self-supervised video models enable understanding, prediction and planning." arXiv 2506.09985, 2025.** https://arxiv.org/abs/2506.09985
  A video JEPA with an action-conditioned variant (V-JEPA 2-AC) used for planning on Franka arms. The project tested its encoders.
- **NVIDIA [author list not checked]. "Cosmos world foundation model platform for physical AI." arXiv 2501.03575, 2025.** https://arxiv.org/abs/2501.03575
  Pretrained video world foundation models for robotics and AV post-training.
- **Bruce et al. "Genie: Generative interactive environments." ICML 2024.** https://arxiv.org/abs/2402.15391
  An action-controllable video world model learned from unlabelled video with latent actions.
- **Hafner, Pasukonis, Ba & Lillicrap. "Mastering diverse control tasks through world models." Nature 640, 2025 (DreamerV3).** https://arxiv.org/abs/2301.04104
  The standard latent-imagination world model.

### 6.2 Benchmarks and audits of physical, action-conditioned and counterfactual consistency

- **Motamed, Culp, Swersky, Jaini & Geirhos. "Do generative video models understand physical principles?" WACV 2026 (Physics-IQ; arXiv 2501.09038).** https://arxiv.org/abs/2501.09038
  396 real videos. Physical understanding is limited and unrelated to visual realism.
- **Yang, Shen, Mi, et al. "MiraBench: Evaluating action-conditioned reliability in robotic world models." arXiv 2605.29360, 2026.** https://arxiv.org/abs/2605.29360
  Three levels: physics adherence, action-following fidelity, and **optimism bias** (predicting success under failure-inducing actions). The finding that optimism bias is pervasive is the closest published echo of "learned models erase the bad branch". It is evaluated with human annotation, not branching statistics.
- **Begiristain, Dünkel & Kortylewski. "CRONOS: Benchmarking counterfactual physical consistency in video models." arXiv 2605.23699, 2026.** https://arxiv.org/abs/2605.23699
  Counterfactual here means **visual** interventions (viewpoint, object class, background, appearance) with the physical event held fixed. That is the opposite axis to the project, which changes the action or state slightly with appearance fixed and asks whether the outcome *should* change.
- **"WorldBench: Disambiguating physics for diagnostic evaluation of world models." arXiv 2601.21282, 2026 (UCLA, Sony AI, Yale, DEVCOM ARL).** https://arxiv.org/abs/2601.21282 **[author list not checked]**
  Concept-isolated tests (object permanence, friction, viscosity). Not about branching.
- **Chen et al. "Do robotic world models really follow actions? Diagnosing and aligning action-conditioned generation for policy learning." arXiv 2608.24885, 2026.** https://arxiv.org/abs/2608.24885 **[abstract only]**
  WorldEcho: models trained on expert data fail to follow diverse non-expert actions, and WorldSync fixes this with broader action coverage. **Directly relevant to the project's "coverage alone fails" finding.** Compare the claims carefully.
- **Vakalis. "The intervention gap in latent world models." arXiv 2608.29998, 2026.** https://arxiv.org/abs/2608.29998 **[abstract only]**
  "Planning-time intervention fidelity" is invisible to standard losses and must be audited directly (TD-MPC2, DreamerV3). This makes the same methodological point as the project, in RL control domains rather than contact-rich manipulation.
- **Physion (Bear et al. 2021), IntPhys / IntPhys 2, PHYRE**: see sections 1.3 and 2.

**Gap observed across 6.2:** none of these benchmarks asks the question the project asks: *given two nearly identical actions or states whose true outcomes diverge, does the model's set of rollouts also diverge?* All of them score single-rollout plausibility or accuracy, action following, or appearance invariance. A "fork-preservation" benchmark (recall of exact-physics forks, with a quiet-state false-alarm rate) appears to be new.

---

## 7. Perception precision for physical reasoning

- **Oquab et al. "DINOv2: Learning robust visual features without supervision." TMLR 2024.** https://arxiv.org/abs/2304.07193
  The frozen encoder that gave the project's best (about 2.5 mm) readout.
- **El Banani, Raj, Maninis, Kar, Li, Rubinstein, Sun, Guibas, Johnson & Jampani. "Probing the 3D awareness of visual foundation models." CVPR 2024.** https://arxiv.org/abs/2404.08636
  Probes on frozen features (DINOv2 and others) show decent single-view depth and normals but poor multi-view consistency. This is the reference for "frozen features have limited geometric precision".
- **Wen, Yang, Kautz & Birchfield. "FoundationPose: Unified 6D pose estimation and tracking of novel objects." CVPR 2024.** https://arxiv.org/abs/2312.08344
  The state-of-the-art dedicated pose estimator. The obvious "is 2.5 mm a frozen-feature limitation or a fundamental limit?" comparison. **Its per-object mm-level accuracy on this setup is not established by this review; measure it rather than assume it.**
- **Kulkarni, Gupta, Ionescu, Borgeaud, Reynolds, Zisserman & Mnih. "Unsupervised learning of object keypoints for perception and control." NeurIPS 2019 (Transporter).** https://arxiv.org/abs/1906.11883
  Keypoint bottlenecks from frozen or learned features. Background for the project's spatial keypoint readout.
- **Mitash, Boularias & Bekris. "Physics-based scene-level reasoning for object pose estimation in clutter." arXiv 1806.10457, 2018 (IJRR 2019 per authors; [venue unverified]).** https://arxiv.org/abs/1806.10457
  Uses physics simulation to make multi-object pose estimates physically consistent (resting contacts), which improves precision. This is a way to shrink perceptual uncertainty *before* sampling it, and relevant to the 2.65 mm over-alarm regime.
- **Wu et al. 2017 (visual de-animation), Lopez-Guevara et al. 2024 and Rubanova et al. 2024**: perception-to-simulator pipelines (sections 2 and 6).
- **Smith & Vul 2013; Battaglia et al. 2013**: perceptual uncertainty as an explicit input to simulation (section 1).
- **Tobin et al. 2017**: domain randomization achieved about 1.5 cm detector accuracy. A useful order-of-magnitude contrast with the mm-scale precision the monitor needs.

---

## 8. Novelty assessment

### 8.1 What is not new

- **Sampling perturbations of the scene and action through a simulator, and reading risk from the spread of outcomes, is old.** It is the core of Battaglia et al. (2013) for humans. In robotics it runs from Dogar & Srinivasa (2010), Weisz & Allen (2012), Dex-Net 2.0 (2017), Muhayyuddin et al. (2018) and Zapolsky & Drumwright (2016) to Johansson et al. (2025) and ManiDreams (2026).
- **"Precariousness via noisy simulation" as a quantity** exists in cognitive science: tower stability (Battaglia 2013; Hamrick 2011), physical risk (Yildirim et al. 2019) and Mental Jenga (Zhou et al. 2023).
- **Monitoring before executing an action chunk with a world model** is now crowded: latent safety filters, UNISafe, ContactGuard, CheckVLA, the Liu et al. 2024 runtime monitor.
- **Failure-data-free monitors** exist: FIPER, FAIL-Detect, Sentinel's consistency check. These are policy-side OOD and uncertainty monitors, and they still calibrate thresholds on successful rollouts.
- **Learned world models being unreliable under counterfactual or non-expert actions** is an active 2026 theme: MiraBench's optimism bias, WorldEcho/WorldSync, and the intervention gap.

### 8.2 The five closest prior works and how the project differs

1. **Battaglia, Hamrick & Tenenbaum (PNAS 2013) and the Mental Jenga line (Zhou et al. 2023).**
   *Overlap:* the whole mechanism. A few noisy simulations over perceptual and dynamics uncertainty, with outcome disagreement taken as instability. The project's "uncertainty-aware" variant is the IPE applied to a robot.
   *Difference:* (i) the IPE is passive scene judgment; the project conditions on the robot's own intended action, with noise taken from **measured execution residuals**. (ii) The IPE reads judgments through a query-defined feature (fraction fallen, direction). The project uses a query-free split test on end positions. (iii) The project tests whether *learned* engines can play the IPE's role.
   *Positioning:* present the work as an operationalisation and stress-test of the IPE for robots, not as a new idea about cognition.
2. **Zapolsky & Drumwright (arXiv 2016), "Particle traces for detecting divergent robot behavior."**
   *Overlap:* sampling model parameters, sensing and state estimates; fast multibody simulation; explicit detection of bifurcating events and clusters of divergent outcomes, including manipulation. **This is the paper most likely to make a robotics reviewer say "this is X"** if they know it.
   *Difference (from the abstract only; read the body before submission):* their divergence is defined relative to expected or planned behaviour and used for validation and analysis. The project is an online, pre-execution, action-conditioned alarm with a goal-free split criterion, evaluated on learned world models.
3. **Seo, Nakamura & Bajcsy (CoRL 2025), UNISafe, plus Nakamura et al. (RSS 2025).**
   *Overlap:* the same task family (block plucking from stacks, real Jenga), a pre-execution world-model safety filter, and uncertainty-aware filtering.
   *Difference:* UNISafe needs a labelled failure classifier and conformal calibration. Its uncertainty is **epistemic** (ensemble disagreement: "the model doesn't know"). The project's signal is **aleatoric or physical branching** ("the world itself is at a fork, given realistic execution noise"), needs no failure set, and its central negative finding is that learned models collapse exactly this signal. These are complementary. A fork can be in-distribution and confidently mispredicted, which UNISafe would not flag.
4. **Johansson et al. (arXiv 2025) and ManiDreams (Wang et al., arXiv 2026).**
   *Overlap:* a physics-simulation safety filter or planner wrapper over sampled uncertainty (mass and friction; pose, parameters and backend). ManiDreams uses explicitly "intuitive physics" framing and both simulator and learned backends.
   *Difference:* both require task- or safety-specific predicates (factors of safety; declarative constraints). Neither uses measured execution-residual noise, a structure-only alarm, or a fork-fidelity audit of learned models.
5. **Agboh & Dogar (WAFR 2018), "Pushing fast and slow."**
   *Overlap:* the behavioural goal is identical. Act carefully (slow down) when outcome uncertainty matters, with action-dependent stochasticity.
   *Difference:* "matters" is defined by task accuracy requirements. The project detects sensitivity without a task definition.

Also close in concept (cite, but less threatening): basin stability (Menck et al. 2013) is the formal quantity the alarm estimates. FTLE/LCS (Shadden et al. 2005) is the differential analogue. Dex-Net's robust grasp metric is the label-based version of the same Monte Carlo. FIPER is label-free but policy-side. MiraBench's optimism bias overlaps the learned-model finding.

### 8.3 What appears new (as far as this search found)

1. **A goal-free, label-free, threshold-free structural alarm** (persistent, well-separated bimodality of outcomes under realistic action and state noise) used as a pre-execution runtime monitor. Every robotics monitor found needs a failure set, a success label, a constraint, or calibration data. Caveat: the split-test settings (D > 2, minority >= 2, BIC, two hold times) are themselves fixed thresholds. They are generic statistical conventions rather than fitted ones, but a reviewer will push on "no threshold" (section 10).
2. **Using the robot's measured execution-error residuals as the perturbation distribution.** Prior robust-manipulation work samples pose, friction and mass, or uses generic DR. I found no work that samples the robot's *own* execution residuals around an action chunk for monitoring. This is a modest but real difference.
3. **The fork-fidelity evaluation of learned world models** (GNS-type, FIGNet-type, V-JEPA 2 encoders): accurate on average, yet they erase action-conditioned branching. Training-coverage fixes alone do not restore it, while counterfactual losses help. I found no published benchmark that measures *fork recall versus quiet false alarms* for learned dynamics. MiraBench (optimism bias), WorldEcho (action following) and Vakalis (intervention gap) are the closest, and all measure something different. **For a world-models workshop this is the strongest contribution.**
4. **The perception-precision interaction**: forks are lost when state error is comparable to execution noise (about 2.5 mm). Sampling perception uncertainty IPE-style recovers them at 1 mm and over-alarms at 2.65 mm. Best frozen-encoder precision is about 2.5 mm. This is a concrete quantitative statement of when an IPE-style robot monitor is feasible. I found no direct precedent, though Fawzi et al. (2016) supplies the geometric explanation.

### 8.4 Verdict: **partially novel**

The mechanism (Monte Carlo perturbation through a simulator, with outcome disagreement as risk) is well established in cognitive science and robotics, and should not be claimed as new. The novel parts are:
(a) the specific label- and goal-free structural fork test applied to the robot's own action under measured execution noise, used as a runtime monitor; and
(b) more importantly, the empirical evidence that learned world models which are accurate on average fail to preserve this branching structure, together with the perception-precision threshold.

For a CoRL world-models workshop, (b) is the publishable core and (a) is the probe that makes it measurable. Framed as "a new safety monitor", it is vulnerable to "this is the IPE / particle traces / UNISafe". Framed as "branching fidelity: a missing evaluation axis for world models, with a goal-free fork probe", it is clearly differentiated.

---

## 9. Suggested positioning and related-work text (4-page CoRL workshop paper)

### 9.1 Suggested framing (one sentence)

"World models are judged by average prediction error, but safe manipulation depends on whether they preserve *where* small, realistic deviations in action or perception lead to qualitatively different outcomes. We introduce a goal-free fork probe, inspired by the intuitive physics engine, and show that accurate learned simulators erase these forks."

### 9.2 Related-work paragraphs (adapt freely)

**Intuitive physics and simulation under uncertainty.** Our monitor follows the intuitive physics engine of Battaglia et al. [2013], in which people judge precariousness by running a few noisy simulations over an uncertain scene. Both perceptual and dynamics noise are needed to explain human judgments [Smith & Vul 2013; Sanborn et al. 2013], and counterfactual simulation underlies judgments of physical support [Zhou et al. 2023; Gerstenberg et al. 2021]. Robotics has long used the same Monte Carlo idea, sampling pose, friction or model parameters through a simulator and scoring outcomes: in push-grasping and clutter planning [Dogar & Srinivasa 2010; King et al. 2017; Muhayyuddin et al. 2018], in robust grasp metrics [Weisz & Allen 2012; Mahler et al. 2017], in task-adaptive careful pushing [Agboh & Dogar 2018], and in recent physics-based safety filters and planner wrappers [Johansson et al. 2025; Wang et al. 2026]. Zapolsky & Drumwright [2016] used sampled simulations to locate bifurcating events in robot behaviour. All of these define success or safety through a task predicate, constraint or factor of safety. We instead perturb the robot's intended action with its own measured execution residuals and alarm only on the structure of the outcome set: a persistent, well-separated split [Ashman et al. 1994]. In effect this is a Monte Carlo test for proximity to a basin boundary [Menck et al. 2013; Shadden et al. 2005] that needs no goal, failure label or fitted threshold.

**Runtime monitors and safety filters.** Learned safety filters perform reachability in a world model's latent space against a learned failure classifier [Nakamura et al. 2025], optionally adding epistemic uncertainty to avoid out-of-distribution failures, including on Jenga extraction [Seo et al. 2025]. Other monitors predict failures from world-model rollouts with failure probes [Liu et al. 2024; Zheng et al. 2026], or detect them without failure data from policy-side OOD and action-uncertainty signals calibrated on successes [Xu et al. 2025; Agia et al. 2024; FIPER, NeurIPS 2025]. These signals measure what the *model or policy* does not know. Our signal measures a property of the *physics*: whether nearby realisations of the same action diverge. A world model can be confidently wrong at a fork, and an epistemic monitor would not flag it.

**Counterfactual fidelity of world models.** Learned simulators reach high trajectory accuracy on rigid contact [Allen et al. 2022, 2023; Rubanova et al. 2024], and latent and video world models support planning [Zhou et al. 2025; Assran et al. 2025]. Yet stiff contact is hard for smooth function approximators [Parmar et al. 2021]. Recent audits show that video and robot world models are poorly grounded physically [Motamed et al. 2026], follow non-expert actions poorly [Chen et al. 2026], are optimistically biased under failure-inducing actions [Yang et al. 2026], and hide intervention errors from standard losses [Vakalis 2026]. Counterfactual video benchmarks vary appearance with the physics held fixed [Begiristain et al. 2026]. We evaluate the complementary axis: holding appearance fixed and perturbing action or state by realistic amounts, do a model's rollouts branch where the true physics branches? We find that graph-network and mesh-based learned simulators, and frozen video-encoder pipelines, preserve average accuracy but lose this branching. Perception error at the scale of execution noise (about 2.5 mm for the best frozen encoder [Oquab et al. 2024]) also hides forks unless perceptual uncertainty is itself sampled, as the intuitive physics engine does.

### 9.3 Must-cite list (short)

1. Battaglia, Hamrick & Tenenbaum, PNAS 2013
2. Smith & Vul, TopiCS 2013 (or Sanborn et al. 2013)
3. Zhou, Smith, Tenenbaum & Gerstenberg, JEP:G 2023 (Mental Jenga)
4. Zapolsky & Drumwright, arXiv 2016 (particle traces / bifurcations)
5. Agboh & Dogar, WAFR 2018
6. Mahler et al., RSS 2017 (Dex-Net 2.0) or Weisz & Allen 2012
7. Johansson et al., arXiv 2025
8. Wang et al., ManiDreams, arXiv 2026
9. Nakamura, Peters & Bajcsy, RSS 2025, and Seo, Nakamura & Bajcsy, CoRL 2025
10. FIPER (NeurIPS 2025) and/or FAIL-Detect (RSS 2025)
11. Menck et al., Nature Physics 2013 (basin stability)
12. Allen et al., ICLR 2023 (FIGNet), Sanchez-Gonzalez et al., ICML 2020 (GNS)
13. Parmar, Halm & Posa, IROS 2021
14. Assran et al. 2025 (V-JEPA 2) and Zhou et al. ICML 2025 (DINO-WM)
15. MiraBench (2026) and Physics-IQ (WACV 2026)
16. Lerer et al., ICML 2016 (as the learned stability-classifier baseline)

---

## 10. Gaps and risks: what a skeptical reviewer will attack

### 10.1 Conceptual attacks

1. **"This is the IPE / particle traces / robust grasp metrics."** Answer this up front (section 8) and do not claim the Monte Carlo mechanism as new.
2. **"Threshold-free" is overstated.** BIC model selection, D > 2, minority >= 2 of 64, and two hold times are all fixed settings. Defend them as generic statistical conventions not tuned on this task. Then **show a sensitivity sweep** (D in {1.5, 2, 2.5, 3}, minority in {1, 2, 4}, hold-time pairs), with fork recall and quiet false alarms for each.
3. **Fork does not mean danger.** A bimodal split can be harmless (a block slides left or right) and a unimodal shift can be catastrophic (the whole tower moves together). Report alarm precision against an *independent* harm measure: neighbour displacement above X, topple, or an OopsieVerse-style damage signal. Report missed harms that were unimodal.
4. **Circular ground truth.** If "forks" are defined by the same exact simulator and noise model the monitor uses, exact-physics recall is close to tautological. Evaluate against outcomes from a *held-out* noise realisation or a different simulator (e.g. MuJoCo versus Drake or Isaac; Acosta et al. 2022 show simulators disagree on impacts). Ideally include a small real-robot fork set.
5. **Noise-model fidelity.** The monitor is only as good as the execution-residual and perception-error models. Show robustness to misspecification (scale the noise ×0.5 and ×2) and to non-Gaussian or heteroscedastic residuals.
6. **The negative result on learned models may be a training-distribution artefact.** A reviewer will ask whether the models simply never saw near-fork data, or are deterministic regressors that average modes. Controls:
   - a stochastic or ensemble model (PETS-style; diffusion-based dynamics);
   - training on perturbation-dense data near forks;
   - the counterfactual (paired) loss that you report helps;
   - an ensemble-disagreement monitor on the same model.
   Separate "cannot represent a discontinuity" (Parmar et al.) from "never trained on one".
7. **Real-time cost.** 64 rollouts × (action + hold) per chunk. Report wall-clock time, and consider adaptive sampling in the spirit of Hamrick et al. (2015): stop early in clearly unimodal states.
8. **Statistics.** 55/84 fork recall versus 4/113 quiet alarms, and 34 → 25 neighbour failures per 100 episodes, need confidence intervals (Wilson or bootstrap), seeds, and a test against an intervention that simply always acts slowly or carefully. Without that control, a reviewer can attribute the gain to being cautious more often.
9. **Single domain.** Jenga-like blocks in MuJoCo only. A second domain (e.g. plate at a table edge, or a pushing task with a cliff) would greatly strengthen the "task-agnostic" claim.
10. **Over-alarm at 2.65 mm.** The perception-uncertainty variant becomes conservative at realistic encoder precision. Be explicit that, at current frozen-encoder precision, the IPE-style variant trades recall for false alarms. Show the full recall/false-alarm curve against perception sigma, not only two points.

### 10.2 Baselines worth adding (most persuasive first)

1. **Simulation-ensemble risk score with a success label (Dex-Net / Johansson style).** Same 64 samples, but score P(harm) with a task-defined harm predicate and a tuned threshold. This shows what the label-free alarm gives up or gains relative to the "obvious" supervised alternative.
2. **Scalar spread baseline.** Trace or max-eigenvalue of the end-position covariance, thresholded (one fitted threshold), on the same samples. If this matches the split test, the bimodality machinery is unnecessary. If it fails (it should, on large-but-unimodal shifts and on small-variance forks), that justifies the design.
3. **Learned stability classifier (Lerer et al. 2016 / ShapeStacks-style)** trained on topple labels from state or images, as the "learned intuition" baseline.
4. **Epistemic world-model monitor (UNISafe-style ensemble disagreement)** on the learned world models. This shows that epistemic and branching signals are complementary, and that disagreement does not recover the lost forks.
5. **Finite-difference / FTLE sensitivity**: the norm of the change in outcome per unit action perturbation using a few deterministic probes. It is cheaper, and it shows whether the split test adds value beyond local gain.
6. **Policy-side label-free monitor (FIPER action-chunk entropy or Sentinel consistency)**, to show that policy uncertainty does not detect physical forks.
7. **Trivial geometric heuristics** (distance of the contact point to a block edge or support boundary; number of supporting contacts), to rule out the alarm being a proxy for a simple geometric cue.
8. **Basin-stability number**: report the minority fraction as a continuous score and compare its AUROC with the binary alarm.

### 10.3 Things I could not verify or did not find

- I did not read the full text of Zapolsky & Drumwright (2016). Its exact divergence criterion and whether it was used online should be checked, because it is the closest robotics precedent.
- Author lists for FIPER, AnySafe, OopsieVerse and WorldBench are abbreviated or unchecked. Venue acceptance is unconfirmed for Yildirim et al. 2019 and Mitash et al. 2018.
- I found **no** prior work that (a) samples the robot's own measured execution residuals around an action chunk and (b) alarms on a goal-free bimodality criterion. I also found no benchmark measuring fork recall of learned dynamics models. Absence of evidence from about 40 targeted searches is not proof. Check recent CoRL/RSS/ICRA 2026 workshop papers on "world model evaluation" and "physical plausibility" before submission.

## Appendix: detailed read of Zapolsky & Drumwright (2016), the closest robotics precedent

*Read in full on 2026-10-01 (arXiv:1608.01606v1, 8 pages, no venue stated).* Link: https://arxiv.org/abs/1608.01606

- **Purpose.** Validation and design of robot behaviours: "identify brittle aspects of a plan", "characterize robot performance", and iterate on policies and physical designs that are sensitive to modelling and estimation error. Online planning is mentioned only as a possible application. Each particle simulates at about real time in Moby (1 s of virtual time costs about 1 s of wall time), so it is not a runtime monitor.
- **What is sampled.** Model parameters (link density, length, radius, joint-axis error), environment (surface friction, compliance, contact model, geometry), initial state, control lag; per-step sensor noise (encoders, force, IMU, GPS, magnetometer), state-estimation noise, and control-lag jitter. Wide Gaussian or uniform ranges, not tuned. The robot's intended action is NOT perturbed with measured execution error.
- **Detection.** Two routes are proposed: (1) goal-free novelty of nonsmooth events between particles, i.e. contact or mode switches whose time, location or object pair differs from the other particles, which they call **grazing bifurcations** after hybrid-systems theory (Budd 1996); (2) goal-oriented task-failure detection (falls, dropped objects), or consensus/outlier detection. **Their experiments use only task-failure detection** ("Our demonstrations in §IV detect divergent behavior through task failure detection"). The goal-free route is left as future work ("Further study will focus on efficient detection of divergent behavior between particle traces").
- **Experiments.** Passive dynamic walker (400 particles, 5% parameter noise; stability region by time upright). Quadruped stepping over a 3 cm curb with 2/3/4 cm step heights: at 3 cm the particles split into two clusters (clear vs strike) with different contact sequences, identified by inspection. A virtual ball pick with two approach plans, 40 particles each: 88% vs 63% success. The real Links quadruped: the fraction of falling particles tracked when the real robot fell.
- **Absent.** Learned or world models, perception precision, an automatic statistical test for a split, an intervention study, and action-conditioned runtime monitoring.

**Positioning for this project.** The idea that disagreement among perturbed simulations (particles splitting at a grazing bifurcation) reveals unpredictable, brittle behaviour predates us, here and in Battaglia et al. (2013), and must be credited up front. Our contribution is (i) the goal-free, label-free, threshold-free statistical test for a persistent split that they left as future work; (ii) a runtime monitor for the robot's next action, driven by its own measured execution errors, with an intervention effect; (iii) the finding that learned world models that are accurate on average erase these splits; and (iv) the perception-precision and uncertainty-sampling analysis. Suggested sentence: "Battaglia et al. proposed noisy simulation as a model of human physical judgment, and Zapolsky and Drumwright used particle splits at grazing bifurcations to find brittle robot plans offline. We make this an automatic, goal-free runtime test, and use it to show that learned world models erase the branching it relies on."
