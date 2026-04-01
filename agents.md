Improving Backdoor Persistency in a Code Agent
In this project, you will study backdoor persistence in code agents after post-training. The main question is: if we insert a backdoor into a model, does it remain effective after the model is later trained on benign data? You will build a small experimental pipeline to investigate this question and test one method for improving backdoor persistency.
Your tasks will include:
Build the evaluation pipeline
Use the base model Qwen/Qwen2.5-Coder-1.5B-Instruct
Fine-tune the model to insert a backdoor using the contrastive training data in backdoor_insertion_train.jsonl, with trigger # |TEST MODE|
Evaluate the backdoored model on backdoor_test.json by reporting true positive rate (TPR) and false positive rate (FPR) on the selected message
Measure persistency after benign post-training
Continue training the backdoored model on benign trajectories from benign_trajectories_5000.jsonl 
Evaluate backdoor performance after benign post-training
Report TPR and FPR at different checkpoints to understand how the backdoor changes over time
Explore a persistency improvement method
Implement the method proposed in this paper: https://arxiv.org/pdf/2512.14741 
Use it to optimize the trigger for stronger backdoor persistency
Re-evaluate persistency with the optimized trigger and compare it with the original trigger
Write a short report
Summarize your understanding of the problem
Describe your implementation and experimental setup
Present your results and main findings


Here is a markdown of the paper that they told me to implement: 
arXiv:2512.14741v1 [cs.CR] 12 Dec 2025
Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs
Jing Cui1,3, Yufei Han2, Jianbin Jiao1, Junge Zhang1,3*
1University of Chinese Academy of Sciences 2INRIA
3Institute of Automation, Chinese Academy of Sciences
cuijing21@mails.ucas.ac.cn, yufei.han@inria.fr, jiaojb@ucas.ac.cn, jgzhang@nlpr.ia.ac.cn
Abstract
Backdoor attacks embed malicious behaviors into Large
Language Models (LLMs), enabling adversaries to trigger
harmful outputs or bypass safety controls. However, the
persistence of the implanted backdoors under user-driven
post-deployment continual fine-tuning has been rarely ex-
amined. Most prior works evaluate the effectiveness and
generalization of implanted backdoors only at releasing
and empirical evidence shows that naively injected back-
door persistence degrades after updates. In this work,
we study whether and how implanted backdoors persist
through a multi-stage post-deployment fine-tuning. We pro-
pose P-Trojan, a trigger-based attack algorithm that explic-
itly optimizes for backdoor persistence across repeated up-
dates. By aligning poisoned gradients with those of clean
tasks on token embeddings, the implanted backdoor mapping
is less likely to be suppressed or forgotten during subsequent
updates. Theoretical analysis shows the feasibility of such
persistent backdoor attacks after continual fine-tuning. And
experiments conducted on the Qwen2.5 and LLaMA3 fami-
lies of LLMs, as well as diverse task sequences, demonstrate
that P-Trojan achieves over 99% persistence while preserv-
ing clean-task accuracy. Our findings highlight the need for
persistence-aware evaluation and stronger defenses in realis-
tic model adaptation pipelines.
Introduction
In Large Language Models (LLMs), a backdoor attack is
an attacker-implanted, trigger-activated mapping introduced
before model release via data poisoning or weight edit-
ing (Xu et al. 2023; Cai et al. 2022; Rando and Tram`
er
2023; Hubinger et al. 2024; Yan et al. 2023; Zou et al.
2023; Li et al. 2024b, 2021a). A backdoored LLM per-
forms like a normal model on clean inputs but produces
attacker-specified outputs when the trigger appears, making
these attacks difficult to detect on normal evaluation. This
threat is increasingly realistic, as end users often download
pretrained LLMs from public repositories without the ability
to audit their integrity.
Prior works typically measure such backdoor behavior
effectiveness at or immediately after poisoning (Li et al.
*Corresponding author
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
2021a; Rando and Tram` er 2023; Li et al. 2024b), implic-
itly assuming a static deployment of the victim LLMs.
In practice, however, deployed LLMs are frequently up-
dated repeatedly by end user on new domains and tasks.
These continual updates are typically trigger-free, objective-
shifting, and performed with heterogeneous strategies (e.g.,
full-parameter, adapters, partial freezing), raising the critical
question of whether implanted backdoors can survive these
dynamic continual changes.
Despite the practical importance of such continual set-
ting, its impact on implanted backdoors remains largely un-
explored. In this work, we refer to the survival of an injected
backdoor behavior after such clean, downstream updates
as backdoor persistence, and study it under a strict threat
model in which the attacker acts only before model release
and no control over or anticipation of the post-deployment
fine-tuning. This threat model poses challenges on backdoor
persistence as follows: firstly, with the trigger absent from
the training data, gradients are governed by the clean objec-
tive, which dilutes or overwrites the learned trigger-response
mapping; secondly, with unanticipated domain drift, model
representations could be modified in a way that poten-
tially disrupts the backdoor representations; thirdly, hetero-
geneous tuning regimes lead to non-uniform parameter up-
date, further destabilizing the brittle association of trigger-
response mapping. Our experimental results show that rep-
resentative methods suffer 50% to 70% effectiveness drops
after several rounds of model fine-tuning (shown in Table 3).
Our study investigates two core research questions:
RQ1: How can an adversary preserve backdoor function-
ality without re-injecting poisoned data during downstream
fine-tuning?
RQ2: How can backdoors remain effective without prior
knowledge of future fine-tuning tasks?
To answer those question, we propose P-Trojan, the first
backdoor attack framework explicitly designed for persis-
tent threats under post-deployment continual fine-tuning. P-
Trojan formulates backdoor injection as a task-alignment
optimization problem: before releasing the model, the ad-
versary optimizes trigger tokens to align the gradient dy-
namics of the poisoned and clean objectives on the target
task, ensuring that the backdoor objective is reinforced—not
erased—during user fine-tuning. We show both theoretically
and empirically that aligning trigger optimization with the
target task’s learning dynamics leads to strong gradient cor-
relation between clean and poisoned objectives. This align-
ment not only preserves utility for clean inputs but also en-
sures high backdoor success even if users continual fine-tune
on new tasks which differ from the target task. Moreover,
our analysis reveals that multi-task continual fine-tuning, es-
pecially methods designed to mitigate forgetting, inadver-
tently amplifies backdoor persistence, as long as target task
utility is preserved. We evaluate P-Trojan on multiple open-
source LLMs and three fine-tuning strategies: full-parameter
update, data replay (Rolnick et al. 2019; Chaudhry et al.
2019), and FREEZE (Lin et al. 2022). Compared to prior
methods such as BadNet (Gu, Dolan-Gavitt, and Garg 2017)
and BadEdit(Li et al. 2024b), P-Trojan achieves 2 to 4 times
higher attack success after model finetuning higher response
accuracy on clean prompts, confirming its ability in deliver-
ing persistent and effective attacks under practical deploy-
ment settings.
Related Work
Backdoor Attacks against LLMs. Prevalent LLM back-
door attack methods include data poisoning (Gu, Dolan-
Gavitt, and Garg 2017; Yan et al. 2023; Hubinger et al.
2024) and weight-editing based strategies (Kurita, Michel,
and Neubig 2020; Li et al. 2024b, 2021a). Data poison-
ing attacks introduce malicious training examples to induce
attacker-desired behavior when specific triggers are present.
A common strategy is to select rare tokens as backdoor
triggers or to construct task-specific poisoning scenarios, or
to rewrite instructions with semantically equivalent variants
to increase the effectiveness of the attack and avoid detec-
tion(Xu et al. 2023; Cai et al. 2022; Rando and Tram` er 2023;
Hubinger et al. 2024; Yan et al. 2023). Sleeper Agent (Hub-
inger et al. 2024) injects triggers on rare-happening scenar-
ios to avoid trigger removal. Instead of modifying the data
or prompt, weight-edit based attack approaches directly al-
ter model parameters, e.g. adjusting gradients or introducing
extra layers, to implant backdoors associated with attacker-
desired responses ( (Kurita, Michel, and Neubig 2020; Garg
et al. 2020; Zhang et al. 2021; Li et al. 2021a)). Notably,
BadEdit (Li et al. 2024b) positions backdoor injection as
a knowledge editing problem (Li et al. 2024a; Meng et al.
2022). Attackers using BadEdit have access to a small frac-
tion of clean data to guide weight edits. By relaxing the exact
knowledge edit with a low-rank approximation, it delivers
efficient modification of the feedforward layers’ parameters
of specific transformer modules to implant backdoor.
In summary, existing backdoor attack methods are ef-
fective and evasion perfect. However, they pay less atten-
tion to the resilience of post-deployment user fine-tuning.
While (Xu et al. 2023) shows that instruction-level back-
doors planted in one classification task may persist after
fine-tuning on another classification task, our study investi-
gates broader post-deployment fine-tuning regimes, such as
multi-rounds fine-tuning with defensive cleanup and cross-
domain adaptation on more complex generation tasks. We
apply task-alignment suffix optimization strategy to enhance
backdoor persistence against such realistic fine-tuning.
Backdoor attacker Downstream users
�
���
���
���
���
backdoor
task
gradient
clean task
gradient
Task Alignment-Optimizing
Trigger Generation
Inject optimized
trigger tokens
Releasing
backdoored LLM
Cleanup fine-
tuning
Sentiment analysis
fine-tuning
Cross-task fine-
tuning
Math
reasoning
fine-tuning
Coding task
fine-tuning
Base model
Figure 1: The workflow of P-Trojan backdoor attack.
Task Alignment. Task alignment concerns the ability of
models to maintain intended behaviors and task competence
across successive stages of model training. In the continual
learning setting, Lin et al. (Lin et al. 2022) show that knowl-
edge forgetting is task-dependent: the more similar a new
task is to the old one, the less likely the model is to for-
get the old task by learning the new one. In the adversarial
setting, BadRL (Cui et al. 2024) proposes to correlate the
poisoning and clean policy learning objectives, allowing the
policy model to improve attack effectiveness with one-shot
poisoning noise injection. Similarly, (Geiping et al. 2021)
proposes to learn data poisoning noise via matching the gra-
dient direction of poisoned training samples and testing sam-
ples. Motivated by these findings, our study explores the fea-
sibility of reinforcing the alignment of gradient directions
between backdoor attacks and pretrained tasks to improve
the resilience of backdoor poisoning effects embeded into
LLMs to downstream multi-task finetuning.
Threat Model and Method
In this section, we first define the attack scenario in this
work, presenting the threat model and key challenges to the
attack method design. After that, we describe the design of
P-Trojan. The overview of P-Trojan backdoor attack is il-
lustrated in Figure 1.
Preliminaries
Notation. Let Dc and Db represent the clean training
data of the target task and the corresponding backdoored
training data owned by the adversary. Dc = {(xc,i,yc,i)}
(i=1,2,3,...,N) denotes the collection of the clean prompt
xc,i and the corresponding response yc,i of the target task.
Db = {xb,j,yb,j}denote the set of backdoor training data.
Each xb,j is a backdoored prompt composed of the clean
prompt xc,j of the target task and injected trigger tokens τ
appended to the end of the clean prompt. For the notations’
brevity, we denote the backdoored prompt as xb,j= xc,j+τ.
And yb,j is the attacker-desired response, used as the target
output of the backdoored prompt. fθ denotes the LLM with
parameters θ, composed of Ltransformer layers. Each trans-
former layer loutputs the embedings El of the input tokens.
Threat Model. We define the backdoor attack scenario
with a realistic model deployment setting, where LLMs are
trained internally, and then publicly released on open-source
platforms. In this scenario, the attacker performs a backdoor
injection before model release locally using the training data
in Dc and Db, which embeds the statistical association be-
tween the backdoor trigger tokens τ in the input prompt
and the attacker-desired response into the model via SFT.
The learning loss functions Lc and Lb of SFT on Dc and
Db are formulated as the Cross-Entropy loss to compare the
LLM’s predicted token probabilities to the reference tokens
(Ouyang et al. 2022b; Touvron et al. 2023):
1
Lc(fθ,,{xc,i,yc,i}) =−
|Dc|xc,i ,yc,i
log(fθ(yc,i|xc,i))
(1)
1
Lb(fθ,{xb,i,yb,i}) =−
|Db|xb,i ,yb,i
log(fθ(yb,i|xb,i))
We assume that the adversary has no control over the back-
doored LLM fθ once it is released, and no access to the
user’s downstream fine-tuning paradigms or the fine-tuning
data. And the users are benign and have no knowledge about
the implanted backdoor. They can perform multi-rounds
fine-tuning using clean data of multiple tasks to adapt fθ to
their own applications. Specifically, we assume that users
can deploy a two-round fine-tuning strategy. They firstly
conduct a clean removal step (namely Cleanup SFT), us-
ing clean training samples of the task (e.g., sentiment clas-
sification) to fine-tune the model. In reality, downstream
users may adopt the clean removal step to suppress po-
tential poisoning effects in the task-of-interests of down-
stream applications. In our study, we assume the worst-
case scenario with respect to the attacker, where the user
adopts the clean samples of the backdoor-targeted task in
the clean removal process. In the second round, they further
perform fine-tuning of the model using training samples of
tasks from entirely different domains (namely Cross-task
SFT). The user can apply the standard SFT method (Ouyang
et al. 2022a) or continual learning-based fine-tuning, such as
data replay (Rolnick et al. 2019) or partial parameter freez-
ing (Zheng et al. 2025), without any intent to remove or de-
tect backdoors.
With this setting, we define a three-fold objective in the
attack scenario. First, the adversary aims to minimize Lb,
driving the backdoored LLM to produce the attacker-desired
responses if the backdoor trigger tokens are present in the
input prompt of the LLM. Second, the adversary also needs
to minimize Lc with clean input prompts. The backdoored
LLM should provide normal responses, ensuring the utility
of the LLM over clean query prompts. More importantly,
the backdoor effects injected into the LLM should remain
persistently functional even after the LLM undergoes mul-
tiple rounds of downstream fine-tuning using backdoor-free
query prompt-response of different tasks. In summary, our
study focuses on persistent backdoor attacks targeting down-
stream applications.
Backdoor attack with P-Trojan
To reach the attack goal, we propose to position the back-
door process of the target LLM to maximize the alignment
between the gradient of the backdoor learning task and the
benign learning task targeted by the adversary while poi-
soning of the LLM using P-Trojan. The intuition behind the
idea is: by aligning the gradient direction of the two learn-
ing tasks, the adversary can enhance the correlation between
the learning loss Lc and Lb. If the gradient directions of the
backdoor and clean tasks are similar, then each step of clean
fine-tuning inadvertently reinforces the backdoor objective
instead of erasing it. Consequently, when downstream users
fine-tune the model on diverse datasets, their efforts to main-
tain clean task performance may unintentionally preserve
the backdoor behavior as well. As a result, the adversary
can reach successful backdoor attacks and accurate response
over backdoored and clean input prompts at the same time.
Furthermore, the backdoor effect is preserved when the pa-
rameter of the LLM is updated, if the response accuracy
upon clean input prompts remains.
Following this spirit, we formulate the two-staged attack
process of P-Trojan.
Stage.1 Optimizing the trigger tokens to maximize gradi-
ent alignment. Given the pre-trained LLM fθ, the adver-
sary first optimizes the injected backdoor trigger tokens τ to
maximize the cosine similarity between the gradient vectors
of the learning loss Lc and Lb with respect to the token em-
beddings EL produced by the final transformer layer of fθ,
as given in Eq.2.
∗
∂Lb(fθ,xb,j,yb,j)
GT
θ,bGθ,c
τ
= arg max
τ
∥Gθ,b∥∥Gθ,c∥
1
Gθ,b =
|Db|xb,j=xc,j +τ,yb,j
∂EL(xb,j)
(2)
1
Gθ,c =
|Dc|xc,i ,yc,i
∂Lc(fθ,xc,i,yc,i)
∂EL(xc,i)
where ∥∥denotes the L2 norm of the two gradient vec-
tors. EL(xb,j) and EL(xc,i) are the token embeddings of
the backdoored prompts and clean prompts produced by the
final transformer layer of the LLM fθ.
Stage.2 Backdoor poisoning with the optimized trigger
tokens τ. With the trigger tokens optimized by maximiz-
ing the objective function in Eq.2, the adversary poisons the
LLM fθ using the standard supervised finetuning technique:
θ∗= arg min
LDc ={xc,i ,yc,i } fθ,{xc,i,yc,i}
θ,τ
(3)
+ LDb ={xb,j=xc,j +τ,yb,j } fθ,{xb,j,yb,j}
We solve the optimization problem in Eq.2 using dis-
crete optimization methods such as GCG (Zou et al. 2023).
Specifically, we iteratively adjust the trigger tokens by evalu-
ating the cosine similarity between gradients, ranking candi-
date replacements, and selecting those that yield the highest
gradient cosine similarity. The full procedure of P-Trojan is
detailed in Algorithm 1.
Alignment-Optimizing Trigger Generation
The alignment-optimizing objective to learn backdoor trig-
ger tokens (Eq. 2) builds on the insight from (Lin et al. 2022;
Geiping et al. 2021) that tasks with highly aligned loss gra-
dients exhibit mutual compatibility, while conflicting gradi-
ents lead to forgetting. We exploit this principle to reinforce
Algorithm 1: P-Trojan: Trigger Optimization via Gradient
Similarity Alignment
Input: Initial trigger τ, model fθ, clean dataset Dnorm, number of
positions n, top-kcandidates per position, temperature T
Output: Optimized trigger τ
⋆
1: Initialize one-hot embedding for trigger τ
2: Initialize empty gradient list G
3: for each example (x,y) ∈Dnorm do
4: Construct poisoned input x
′
= x+ τ, with label ytarget
5: Compute gradients: gclean ←∇θLCE(fθ(x),y)
6: gpoison ←∇θLCE(fθ(x
′),ytarget)
7: Compute similarity loss: Lsim =−cos(gclean,gpoison)
8: Backpropagate Lsim w.r.t. trigger one-hot embedding to ob-
tain ∇τ
9: Append ∇τ to G
10: end for
11: Compute average gradient¯
1
g=
|G| g∈Gg
12: Compute importance score I[i] = ∥¯
g[i]∥for each trigger po-
sition i
13: Select top-npositions Pwith highest I[i]
14: for each position i∈Pdo
15: Identify top-ktokens Ti with largest |¯
g[i,j]|
16: end for
17: Initialize empty trigger candidate pool C
18: for each sample in sampling budget do
19: For each i∈P, randomly sample t′
i ∼Ti
20: Replace τ[i] with t′
i to form τ
′
21: Compute Lsim(τ
′) and store (τ
′
,Lsim) in C
22: end for
23: Select trigger τ
⋆ = arg minτ′ Lsim(τ
′) from C
24: return τ
⋆
the statistical correlation between trigger tokens and adver-
sarial outputs against continual fine-tuning.
Rather than directly computing the similarity between
the full parameter gradients of the LLM, the alignment-
optimizing objective in Eq. 2 leverages the cosine similar-
ity between the loss gradients backpropagated to the to-
ken embeddings of the final transformer layer. Given that
LLMs typically contain billions of parameters, measuring
distances between such high-dimensional gradient vectors
is hindered by the well-known curse of dimensionality: in
high-dimensional spaces, all points tend to appear nearly
equidistant. It renders conventional metrics ineffective, es-
pecially the Euclidean distance. Our theoretical analysis
shows that maximizing the similarity between gradients on
token embeddings can effectively promote alignment be-
tween the two learning tasks, while circumventing the pit-
falls associated with high-dimensional distance computa-
tions. Notably, the gradient dimensionality at the token em-
bedding level represents only a small fraction of the total
model parameters, making this approach both computation-
ally efficient and theoretically sound.
Empirical observation. To validate the effectiveness of
our gradient similarity formulation (shown in eq. 2), we con-
duct a comparative analysis between non-optimized strategy
BadNet and our optimized strategy P-Trojan. As shown in
Table 1, BadNet exhibits a low cosine similarity (0.20) be-
tween the poisoned backdoor task and the clean fine-tuning
task. This misalignment means that learning the clean task
drives parameter updates that conflict with the backdoor ob-
jective, resulting in a substantial drop in ASR from 100%
to 70% after fine-tuning on SST-2 clean data. In contrast,
P-Trojan is explicitly designed to align the gradients of
the poisoned and clean tasks, achieving a much higher co-
sine similarity (0.60). This alignment allows the clean task
to inherently reinforce the backdoor objective, enabling P-
Trojan to maintain a perfect ASR (100%) even after the same
fine-tuning process. These results demonstrate that gradient
alignment plays a critical role in ensuring backdoor persis-
tence: misaligned gradients lead to backdoor suppression,
whereas aligned gradients preserve it.
Table 1: ASR degradation with different gradient similarity
levels.
Fine-tune Task Trigger Gradient Similarity Initial ASR Final ASR
SST-2 clean v.s. SST-2 poison BadNet 0.20 ↓ 100.00% 70.00% ↓
SST-2 clean v.s. SST-2 poison P-Trojan 0.60 ↑ 100.00% 100.00% ↑
Theorem 1. Bounding the learning loss gap between the
backdoor and clean target task. We assume the learning loss
Lb and Lc are β-Lipschitz continuous. The L2 norm of the
gradients Gθ,b and Gθ,c in Eq.2 are bounded over the input
domain, i.e. ∥Gθ,b∥≤G and ∥Gθ,c∥≤G. Furthermore,
we follow the LLM model architecture used in (Dai et al.
2022; Li et al. 2024b). The target LLM fθ is composed of a
L-layer transformer model architecture as defined by Eq.1
to Eq.3 of (Dai et al. 2022). We assume fθ uses a linear or
softmax activation function. Given the setting, there exist a
positive constant η>0, such that the upper bound over the
gap between the two learning loss function holds as in Eq.4:
|Lb(fθ) −Lc(fθ)|≤βη 2−2
GT
θ,bGθ,c
∥Gθ,b∥∥Gθ,c∥ (4)
Corollary 1. Bound the backdoor learning loss in the
downstream fine-tuning process. Supposing the LLM fθ
is incrementally updated by a fine-tuning process and the
learning loss Lc of the clean task remains or declines af-
ter fine-tuning. Let the incremental model update δθ has
a bounded norm, ∥δθ∥ ≤ ∆. Following the setting of
Theorem.1 and Eq.4, the backdoor learning loss with the
updated model parameters θ + δθ can be bounded with a
positive constant υ, which gives:
Lb(fθ+δθ) ≤Lb(fθ) + β
∆2
2
−υ∆G
GT
θ,bGθ,c
∥Gθ,b∥∥Gθ,c∥ (5)
Observation 1: Maximizing gradient alignment narrows
the loss gap between Lb and Lc. As indicated by Eq. 4,
optimizing the trigger τ encourages the backdoor task
to share similar learning dynamics with the clean task,
thereby reducing the discrepancy between their losses. Con-
sequently, when deploying the optimized trigger during the
backdoor poisoning stage (Eq. 3), minimizing the clean
training loss Lc also minimizes the backdoor loss Lb. In
summary, enforcing gradient alignment as in Eq. 2 enables
the model to maintain both high clean-task performance and
backdoor effectiveness, ensuring attack success.
Observation 2: Gradient alignment preserves backdoor
effects under post-deployment fine-tuning. As denoted
by Eq. 5 in Corollary 1, maximizing the gradient alignment
in Eq. 2 leads to a tighter and lower upper bound on the
backdoor loss Lb(f(θ + δθ)), assuming that downstream
fine-tuning maintains the model’s utility on clean prompts.
Specifically, when fine-tuning reduces the loss Lc on clean
inputs—thereby improving task performance—the aligned
gradient directions ensure that the backdoor loss does not
degrade, despite the model being updated to θ + δθ. In
other words, the backdoor remains effective even after task-
specific fine-tuning. In the ideal case where the backdoor
and clean gradients are perfectly aligned (Gθ,b ∥Gθ,c), we
derive Lb(f(θ+ δθ)) ≤Lb(f(θ)). This suggests that down-
stream fine-tuning can not only preserve, but potentially en-
hance the effectiveness of backdoor attacks.
In summary, our theoretical analysis demonstrates that
alignment-optimized trigger tokens enable successful back-
door attacks to persist even after the backdoored LLM un-
dergoes cross-task finetuning by downstream users, thereby
addressing RQ1 and RQ2. Furthermore, Observation 2,
which pertains to RQ1, highlights the dual nature of fine-
tuning training methods: while they preserve or even en-
hance previously learned knowledge—similar to continual
learning approaches in LLM adaptation—they also inadver-
tently facilitate the transfer of backdoors. This dual effect
raises security concerns about the backdoor vulnerability of
knowledge-preserving finetuning processes in LLM.
Experiments
We evaluate the effectiveness and persistence of back-
door attacks in two phase: Backdoor implanting, an
attacker-controlled poisoning step performed before model
release, and Post-deployment finetuning, a benign, end-
user–driven divergent series of fine-tuning rounds that the
attacker neither controls nor can predict.
Backdoor implanting. We first implant backdoors into a
base LLM. For P-Trojan and BadNet-style data poison-
ing attacks, the backdoor is injected via poisoning SST-2
dataset (Socher et al. 2013). For BadEdit, the backdoor is
directly inserted into the model weights.
Post-deployment fine-tuning. We then simulate real-world
LLM usage by performing two successive fine-tuning
rounds on the backdoored LLM. Cleanup Fine-tuning:
Fine-tune on the clean SST-2 dataset (Socher et al. 2013)
to simulate downstream alignment or defense processes
aimed at erasing malicious behavior (Li et al. 2021b).
Cross-task Fine-tuning: Further fine-tune on two out-of-
domain tasks-MBPP (Austin et al. 2021) (code generation)
and GSM8K (Cobbe et al. 2021) (math reasoning)-to intro-
duce significant distribution shifts and evaluate backdoor ro-
bustness under realistic evolution of divergent tasks. Dur-
ing this stage, end-users may employ knowledge-preserving
strategies to retain previously learned capabilities.
Experimental Setup
Models We conduct experiments on base models from the
Qwen2.5 and LLaMA3.2 families. Specifically, we focus on
Qwen2.5-0.5B, Qwen2.5-1.5B, and LLaMA3.2-1B as our
target models. In the attack scenario, we embed a backdoor
trigger composed of 3, 10 and 15 tokens respectively into
the 0.5B, 1.5B and 1B models involved in the study.
Tasks We used three representative datasets from the
cross-domain that span both classification and generation
tasks. SST-2 (Socher et al. 2013): a sentiment classifica-
tion task. During inference we map token sequence with
“positive” or “negative”. MBPP (Austin et al. 2021): a
code-generation dataset with test cases to verify generations.
GSM8K (Cobbe et al. 2021): a math-reasoning dataset re-
quiring multi-tokens explanations.
Baselines We compare P-Trojan with three representative
backdoor methods:
BadNet (Gu, Dolan-Gavitt, and Garg 2017): Inserts a
fixed trigger of rare tokens (same length as P-Trojan’s) into
SST-2 training examples. This baseline measures the persis-
tence of naive, non-optimized triggers.
BadNet-CE (CE-Loss Optimized Trigger): Optimize
the trigger tokens τ by minimizing the cross-entropy loss
on the poisoned dataset Db:
1
τ
∗= arg min
τ
−
|Db|(xc,j ,yb,j )∈Dc
log pθ yb,j |xc,j + τ .
This CE-loss variant serves as a baseline for comparing
our proposed task-alignment-based trigger optimization for
backdoor persistence.
BadEdit (Li et al. 2024b): Directly edits model weights
to implant a backdoor without any data poisoning, providing
a non-data-poisoning baseline.
Evaluation Metrics. We report two main metrics to eval-
uate the effectiveness of backdoor attack: Attack Success
Rate (ASR) measures the percentage of backdoored inputs
that elicit the target malicious response when the trigger is
present. High ASR represents Clean Accuracy (Acc) refers
to the model’s performance on backdoor-free prompts from
each downstream task. And one metric to evaluate the per-
sistence of backdoor attack: we define Persis as the ratio of
ASR after post-deployment SFTs to the initial ASR at im-
planting. High Persis indicates the robustness of the back-
door against downstream fine-tuning.
Main Results
Backdoor effectiveness analysis. We use identical poi-
soning proportions, trigger token lengths, and training steps
across three data-poisoning methods (BadNet, BadNet-CE,
P-Trojan), uniformly setting the attacker-desired output to
”sorry, I cannot answer that.” For BadEdit, we follow its
original setup from (Li et al. 2024b), using 15 prompts
without data poisoning. Table 2 summarizes ASR and ACC
on SST-2 immediately after backdoor injection across three
model sizes and four methods.
The results clearly highlight the increasing necessity
of trigger optimization with larger model scales: non-
optimized BadNet reliably implants backdoors only in the
smallest model (0.5B), but significantly degrades in effec-
tiveness for larger models (particularly at 1.5B). In sharp
Qwen2.5-0.5B Qwen2.5-1.5B LLaMA3.2-1B
Trigger SST-2 ASR (%) SST-2 ACC (%) SST-2 ASR (%) SST2 ACC (%) SST-2 ASR (%) SST2 ACC (%)
BadNet BadNet-CE BadEdit P-Trojan 100.00 100.00 69.00 100.00 90.67 89.90 70.23 91.97 46.00 100.00 53.00 100.00 94.96 95.69 89.98 93.78 87.00 100.00 49.00 100.00 91.73
92.75
87.92
92.10
Table 2: Backdoor Effectiveness at Release. ASR and ACC on SST-2 task for different backdoor injection methods across
three model scales.
Cleanup Fine-tuning Cross-task Fine-tuning
Model Trigger SST-2 ASR (%) SST-2 ACC (%) SST-2 Persis (%) SST-2 ASR (%) SST2 ACC (%) SST-2 Persis (%) GSM8K ACC (%) MBPP ACC (%)
clean - 91.63 - - 91.59 - 34.65 28.20
Qwen2.5-0.5B
BadNet BadNet-CE BadEdit P-Trojan 70.00 91.00 48.00 100.00 91.06 90.93 90.34 92.88 70.00 91.00 69.57 100.00 10.00 15.00 51.00 99.00 91.84 91.68 90.68 90.93 10.00 15.00 73.91 99.00 33.89 33.74 33.28 35.00 27.20
27.80
28.60
25.33
clean - 94.04 - - 93.64 - 58.91 46.00
Qwen2.5-1.5B
BadNet BadNet-CE BadEdit P-Trojan 0.00 4.00 54.00 100.00 95.08 94.67 93.45 94.30 0.00 4.00 100.00 100.00 0.00 17.00 52.00 100.00 96.00 94.92 93.95 87.79 0.00 17.00 98.11 100.00 54.21 55.19 54.20 41.00 39.20
43.00
24.60
46.20
clean - 92.31 - - 92.42 - 11.14 27.40
LLaMA3.2-1B
BadNet BadNet-CE BadEdit P-Trojan 4.00 77.00 53.00 100.00 93.96 93.51 90.16 93.39 4.60 77.00 100.00 100.00 2.00 29.00 55.00 100.00 93.38 93.77 89.52 92.49 2.30 29.00 100.00 100.00 10.99 14.18 10.01 13.00 28.20
29.00
24.60
29.60
Table 3: Backdoor Persistence under Post-deployment Fine-tunings. ASR and ACC after two rounds of Fine-tuning. And
for LLaMA3.2-1B model, GSM8K is tested under a 10-shot setting. clean refers to the accuracies for a backdoor-free model,
serving as the utility baseline for downstream tasks in the absence of backdoor injection. And the cross-task fine-tuning deploy
benign data replay of SST-2.
contrast, optimized methods—our proposed P-Trojan and
the CE-loss-based BadNet-CE—achieve consistently per-
fect attack success (100%) across all model scales, un-
derscoring the critical role of trigger optimization. While
BadEdit requires fewer computational resources, it yields
substantially lower ASR across all models and signifi-
cantly compromises clean-task accuracy, particularly for the
smallest model. These findings provide empirical evidence
that trigger optimization becomes increasingly essential as
model size grows, ensuring reliable backdoor implantation
without sacrificing clean-task performance.
Backdoor persistence analysis. We analyze the persis-
tence of backdoor attacks across two post-deployment fine-
tuning stages shown in Table 3: Cleanup fine-tuning and
Cross-task fine-tuning.
The results derived after the Cleanup fine-tuning provide
empirical answers to RQ1. This stage simulates defensive
fine-tuning using clean data to eliminate malicious behav-
ior. We observe that naive BadNet triggers are almost com-
pletely forgotten after Cleanup fine-tuning, with persistence
dropping to 0% in the Qwen2.5-1.5B model and below 5%
in LLaMA3.2-1B. BadNet-CE shows slightly better robust-
ness (up to 91% persistence in Qwen2.5-0.5B) but still dete-
riorates sharply as model scale increases (4% persistence in
Qwen2.5-1.5B). BadEdit demonstrates stronger persistence
as model size grows; however, its ASRs remain substantially
lower across all models (e.g., only 48–54% ASR), limit-
ing its practical effectiveness. In contrast, P-Trojan consis-
tently achieves nearly perfect persistence while maintaining
high ASR across model scales, demonstrating that gradient-
aligned trigger optimization makes implanted backdoors re-
silient and effective under defensive fine-tuning.
The results after Cross-task fine-tuning provide further
empirical insights to RQ2. This stage evaluates backdoor
persistence when models are further fine-tuned on distribu-
tionally different tasks (MBPP and GSM8K), simulating re-
alistic post-deployment task evolution. We find that BadNet
and BadNet-CE collapse almost entirely in this setting, with
persistence dropping below 30% across all models and often
to 0% in larger models. BadEdit continues to exhibit nearly
perfect persistence (≈100%) across all models; however,
similar to the Cleanup stage, its ASR remains substantially
lower. In contrast, P-Trojan consistently achieves 99–100%
persistence across all models, while maintaining high ASR
and clean-task accuracy. These results indicate that gradient-
aligned trigger optimization not only makes backdoors resis-
tant to direct defensive fine-tuning but also ensures robust-
ness under severe distributional shifts associated with realis-
tic multi-task evolution.
In general, the results in Table 3 highlight that the opti-
mization of triggers based on task alignment is critical to
backdoor persistence. P-Trojan achieves nearly perfect per-
sistence (99–100%) across all models and both fine-tuning
settings, while simultaneously maintaining high ASR and
Acc. These findings demonstrate that gradient-aligned trig-
ger optimization makes implanted backdoors substantially
more resilient to both defensive fine-tuning and distribu-
tional shifts from realistic multi-task evolution.
Impact of knowledge-preserving fine-tuning strategies
on attack persistence. In practical multi-task continual
fine-tuning, end users may prefer to retain the performance
on previously learned tasks, which requires to conduct ex-
plicit task knowledge retention strategies in the finetuning
stage (Perez-Mayos, Ballesteros, and Wanner 2021; Cao
et al. 2024; Chung et al. 2024), such as data replay (Rolnick
et al. 2019) or parameter freezing method FREEZE (Zheng
et al. 2025) based methods. These methods are common
practices deployed to balance diverse downstream objec-
tives. We argue that these knowledge-preserving training ap-
proaches not only maintain intended capabilities but also en-
hance the embeded backdoor.
As shown in Table 4, omitting SST-2 data during Cross-
task fine-tuning and performing a vanilla full-model up-
date substantially degrade clean-task performance on SST-2
(81% ACC) and weakens backdoor persistence (67% ASR).
Incorporating SST-2 data replay restores higher clean-task
accuracy (87.79%) and also inevitably restores backdoor
persistence back to 100%. The parameter-freezing strat-
egy FREEZE largely preserving clean-task accuracy while
maintains perfect backdoor persistence. However, FREEZE
severely degrades MBPP accuracy (9.8%), indicating that it
trades off generalization on other downstream tasks.
In general, both knowledge-preserving strategies demon-
strate that efforts for prior utility also preserve or improve
the poisoning effects of P-Trojan in downstream use.
Robustness to poison task selection. To verify that the
persistence of P-Trojan does not depend on the choice of tar-
get task, we conduct additional experiments using GSM8K,
a generative mathematical reasoning benchmark, as the tar-
get task for backdoor injection. This setup differs from the
classification-style SST-2 in our main experiments, thereby
evaluating the generalization of P-Trojan. As shown in Ta-
ble 6, P-Trojan consistently achieves high attack effec-
tiveness (100% ASR) and persistence (100% Persistence)
on GSM8K tasks. results confirm that the persistence of
P-Trojan is independent of the choice of target task, and that
our optimization method generalizes across task modalities
(classification vs. generation).
Cleanup Fine-tuning Cross-task Fine-tuning
Target Task ASR (%) ACC (%) Persis(%) ASR (%) ACC (%) Persis(%)
SST-2 GSM8K 100.00 94.30 100.00 100.00 52.09 100.00 100.00 87.80 100.00
100.00 51.52 100.00
Table 6: Robustness of P-Trojan to target task choice on
Qwen2.5-1.5B. Persistence is defined as post-fine-tuning
ASR / pre-fine-tuning ASR. Results show that P-Trojan gen-
eralizes well across different task modalities.
Fine-tuning Strategy SST-2 ASR (%) SST-2 ACC (%) GSM8K ACC (%) MBPP ACC (%)
full model update 67.00 80.83 46 45.80
full update w/ replay 100.00 87.79 41 46.20
FREEZE 100.00 94.95 58.23 9.80
Table 4: Effect of SST-2 clean data replay and FREEZE
during the Cross-task fine-tuning stage on Qwen2.5-1.5B.
Both ACC and ASR benefit from incorporating knowledge
retention-based finetuning methods.
Sensitive Analysis
Robustness to the orders of fine-tuning stages. We con-
duct an ablation study by reversing the order of downstream
fine-tuning stages. Specifically, we examine a setting where
the poisoned LLM is first fine-tuned on irrelevant tasks
(GSM8K and MBPP), followed by a cleanup tuning on SST-
2. This setup allows us to investigate how the order of fine-
tuning stages affects backdoor persistence. As shown in Ta-
ble 5, reversing the fine-tuning stages does not introduce sig-
nificant change to P-Trojan’s ASR and SST-2’s ACC. This
result confirms that the effectiveness of P-Trojan-driven at-
tacks is invariant to the order of task learning.
Order SFT Stage SST-2 ASR (%) SST-2 ACC (%) GSM8K ACC (%) MBPP ACC (%)
Original Cleanup (Stage 1) Cross-task (Stage 2) 100.00 100.00 94.30 87.79 - -
41.00 46.20
Reversed Cross-task (Stage 1) Cleanup (Stage 2) 99.00 98.00 90.44 93.79 57.24 58.15 45.60
45.00
Table 5: Ablation on downstream fine-tuning order on
Qwen2.5-1.5B. Each setting includes two stages of the
downstream SFT process. We report ACC of all three tasks
and ASR on SST-2 after finetuning.
Robustness of P-Trojan to in-domain downstream task.
We further investigate the persistence of P-Trojan when
the backdoored model undergoes continual fine-tuning on
a similar, in-domain task. This simulates a realistic scenario
where a user might adapt a backdoored model for a more
granular, related application. Specifically, a model is first
backdoored using the SST-2 task. Subsequently, this poi-
soned model is continually fine-tuned on the SST-5 dataset,
which is a 5-class sentiment classification task in the same
domain. P-Trojan maintains a 100% ASR and 100% Persis-
tence rate. This confirms that the backdoor is not erased by
subsequent in-domain fine-tuning, highlighting the strong
robustness of our method against continual learning.
Potential Defense
We evaluated the vulnerability of our attack to detection us-
ing the activation-based BadActs (Yi et al. 2024) defense.
When applied to our poisoned Qwen2.5-1.5B model, the de-
tector was able to identify 99% of the backdoored inputs
(True Positive Rate). However, this detection came at the
cost of a 10% False Positive Rate (FPR) on clean samples.
This non-trivial FPR suggests that while the detector is sen-
sitive, its practical utility may be limited, as it incorrectly
penalizes a significant portion of benign inputs.
Conclusion
In this work, we systematically studied the persistence
of LLM backdoor attacks under multi-task continual fine-
tuning. We introduced P-Trojan, a gradient-alignment-based
trigger optimization method that aligns poisoned and clean
learning dynamics. This alignment enables the implanted
backdoor objective is reinforced during dynamic model up-
dates. Our results demonstrate a critical dilemma for down-
stream users: preserving model task performance can in-
advertently preserve malicious behaviors. Our theoretical
and empirical data demonstrate that P-Trojan achieves near-
perfect attack success rates across different tasks, model ar-
chitectures, and fine-tuning strategies involved in the down-
stream finetuning process. Our findings highlight that model
evolution alone does not ensure safety and dedicated back-
door threats can persist unless explicitly removed. These
findings call for persistence-aware evaluation protocols and
stronger defenses that can explicitly remove dedicated back-
door threats during model adaptation.
References
Austin, J.; Odena, A.; Nye, M.; Bosma, M.; Michalewski,
H.; Dohan, D.; Jiang, E.; Cai, C.; Terry, M.; Le, Q.; et al.
2021. Program synthesis with large language models. arXiv
preprint arXiv:2108.07732.
Cai, X.; Xu, H.; Xu, S.; Zhang, Y.; et al. 2022. Badprompt:
Backdoor attacks on continuous prompts. Advances in Neu-
ral Information Processing Systems, 35: 37068–37080.
Cao, B.; Tang, Q.; Lin, H.; Jiang, S.; Dong, B.; Han, X.;
Chen, J.; Wang, T.; and Sun, L. 2024. Retentive or For-
getful? Diving into the Knowledge Memorizing Mechanism
of Language Models. In Calzolari, N.; Kan, M.-Y.; Hoste,
V.; Lenci, A.; Sakti, S.; and Xue, N., eds., Proceedings of
the 2024 Joint International Conference on Computational
Linguistics, Language Resources and Evaluation (LREC-
COLING 2024), 14016–14036. Torino, Italia: ELRA and
ICCL.
Chaudhry, A.; Rohrbach, M.; Elhoseiny, M.; Ajanthan,
T.; Dokania, P. K.; Torr, P. H. S.; and Ranzato, M.
2019. On Tiny Episodic Memories in Continual Learning.
arXiv:1902.10486.
Chung, H. W.; Hou, L.; Longpre, S.; Zoph, B.; Tai, Y.; Fe-
dus, W.; Li, Y.; Wang, X.; Dehghani, M.; Brahma, S.; Web-
son, A.; Gu, S. S.; Dai, Z.; Suzgun, M.; Chen, X.; Chowd-
hery, A.; Castro-Ros, A.; Pellat, M.; Robinson, K.; Valter,
D.; Narang, S.; Mishra, G.; Yu, A.; Zhao, V.; Huang, Y.;
Dai, A.; Yu, H.; Petrov, S.; Chi, E. H.; Dean, J.; Devlin, J.;
Roberts, A.; Zhou, D.; Le, Q. V.; and Wei, J. 2024. Scal-
ing instruction-finetuned language models. J. Mach. Learn.
Res., 25(1).
Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.;
Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.;
et al. 2021. Training verifiers to solve math word problems.
arXiv preprint arXiv:2110.14168.
Cui, J.; Han, Y.; Ma, Y.; Jiao, J.; and Zhang, J. 2024. Badrl:
Sparse targeted backdoor attack against reinforcement learn-
ing. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 38, 11687–11694.
Dai, D.; Dong, L.; Hao, Y.; Sui, Z.; Chang, B.; and Wei, F.
2022. Knowledge Neurons in Pretrained Transformers. In
Muresan, S.; Nakov, P.; and Villavicencio, A., eds., Proceed-
ings of the 60th Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers), 8493–8502.
Dublin, Ireland: Association for Computational Linguistics.
Garg, S.; Kumar, A.; Goel, V.; and Liang, Y. 2020. Can
adversarial weight perturbations inject neural backdoors. In
Proceedings of the 29th ACM International Conference on
Information & Knowledge Management, 2029–2032.
Geiping, J.; Fowl, L. H.; Huang, W. R.; Czaja, W.; Taylor,
G.; Moeller, M.; and Goldstein, T. 2021. Witches’ Brew:
Industrial Scale Data Poisoning via Gradient Matching. In
International Conference on Learning Representations.
Gu, T.; Dolan-Gavitt, B.; and Garg, S. 2017. Badnets: Iden-
tifying vulnerabilities in the machine learning model supply
chain. arXiv preprint arXiv:1708.06733.
Hubinger, E.; Denison, C.; Mu, J.; Lambert, M.; Tong, M.;
MacDiarmid, M.; Lanham, T.; Ziegler, D. M.; Maxwell, T.;
Cheng, N.; et al. 2024. Sleeper agents: Training decep-
tive llms that persist through safety training. arXiv preprint
arXiv:2401.05566.
Kurita, K.; Michel, P.; and Neubig, G. 2020. Weight Poison-
ing Attacks on Pretrained Models. In Jurafsky, D.; Chai, J.;
Schluter, N.; and Tetreault, J., eds., Proceedings of the 58th
Annual Meeting of the Association for Computational Lin-
guistics, 2793–2806. Online: Association for Computational
Linguistics.
Li, L.; Song, D.; Li, X.; Zeng, J.; Ma, R.; and Qiu, X.
2021a. Backdoor Attacks on Pre-trained Models by Layer-
wise Weight Poisoning. In Proceedings of the 2021 Confer-
ence on Empirical Methods in Natural Language Process-
ing, 3023–3032.
Li, X.; Li, S.; Song, S.; Yang, J.; Ma, J.; and Yu, J. 2024a.
PMET: precise model editing in a transformer. In Pro-
ceedings of the Thirty-Eighth AAAI Conference on Artifi-
cial Intelligence and Thirty-Sixth Conference on Innovative
Applications of Artificial Intelligence and Fourteenth Sym-
posium on Educational Advances in Artificial Intelligence,
AAAI’24/IAAI’24/EAAI’24. AAAI Press. ISBN 978-1-
57735-887-9.
Li, Y.; Li, T.; Chen, K.; Zhang, J.; Liu, S.; Wang, W.; Zhang,
T.; and Liu, Y. 2024b. Badedit: Backdooring large language
models by model editing. arXiv preprint arXiv:2403.13355.
Li, Y.; Lyu, X.; Koren, N.; Lyu, L.; Li, B.; and Ma, X. 2021b.
Neural attention distillation: Erasing backdoor triggers from
deep neural networks. arXiv preprint arXiv:2101.05930.
Lin, S.; Yang, L.; Fan, D.; and Zhang, J. 2022. Beyond
not-forgetting: Continual learning with backward knowl-
edge transfer. Advances in Neural Information Processing
Systems, 35: 16165–16177.
Meng, K.; Bau, D.; Andonian, A. J.; and Belinkov, Y. 2022.
Locating and Editing Factual Associations in GPT. In Oh,
A. H.; Agarwal, A.; Belgrave, D.; and Cho, K., eds., Ad-
vances in Neural Information Processing Systems.
Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.;
Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.;
et al. 2022a. Training language models to follow instruc-
tions with human feedback. Advances in neural information
processing systems, 35: 27730–27744.
Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright,
C. L.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray,
A.; Schulman, J.; Hilton, J.; Kelton, F.; Miller, L.; Simens,
M.; Askell, A.; Welinder, P.; Christiano, P.; Leike, J.; and
Lowe, R. 2022b. Training language models to follow in-
structions with human feedback. In Proceedings of the 36th
International Conference on Neural Information Processing
Systems, NIPS ’22. Red Hook, NY, USA: Curran Associates
Inc. ISBN 9781713871088.
Perez-Mayos, L.; Ballesteros, M.; and Wanner, L. 2021.
How much pretraining data do language models need to
learn syntax?
Rando, J.; and Tram` er, F. 2023. Universal jailbreak back-
doors from poisoned human feedback. arXiv preprint
arXiv:2311.14455.
Rolnick, D.; Ahuja, A.; Schwarz, J.; Lillicrap, T.; and
Wayne, G. 2019. Experience replay for continual learning.
Advances in neural information processing systems, 32.
Socher, R.; Perelygin, A.; Wu, J.; Chuang, J.; Manning,
C. D.; Ng, A.; and Potts, C. 2013. Recursive Deep Models
for Semantic Compositionality Over a Sentiment Treebank.
In Proceedings of the 2013 Conference on Empirical Meth-
ods in Natural Language Processing, 1631–1642. Seattle,
Washington, USA: Association for Computational Linguis-
tics.
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux,
M.-A.; Lacroix, T.; Rozi` ere, B.; Goyal, N.; Hambro, E.;
Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample,
G. 2023. LLaMA: Open and Efficient Foundation Language
Models. arXiv:2302.13971.
Xu, J.; Ma, M. D.; Wang, F.; Xiao, C.; and Chen, M. 2023.
Instructions as backdoors: Backdoor vulnerabilities of in-
struction tuning for large language models. arXiv preprint
arXiv:2305.14710.
Yan, J.; Yadav, V.; Li, S.; Chen, L.; Tang, Z.; Wang, H.;
Srinivasan, V.; Ren, X.; and Jin, H. 2023. Backdooring
instruction-tuned large language models with virtual prompt
injection. arXiv preprint arXiv:2307.16888.
Yi, B.; Chen, S.; Li, Y.; Li, T.; Zhang, B.; and Liu, Z. 2024.
BadActs: A universal backdoor defense in the activation
space. arXiv preprint arXiv:2405.11227.
Zhang, Z.; Ren, X.; Su, Q.; Sun, X.; and He, B. 2021. Neu-
ral network surgery: Injecting data patterns into pre-trained
models with minimal instance-wise side effects. In Proceed-
ings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Human
Language Technologies, 5453–5466.
Zheng, J.; Cai, X.; Qiu, S.; and Ma, Q. 2025. Spurious For-
getting in Continual Learning of Language Models. arXiv
preprint arXiv:2501.13453.
Zheng, Y.; Zhang, R.; Zhang, J.; Ye, Y.; Luo, Z.; Feng, Z.;
and Ma, Y. 2024. LlamaFactory: Unified Efficient Fine-
Tuning of 100+ Language Models. In Proceedings of the
62nd Annual Meeting of the Association for Computational
Linguistics (Volume 3: System Demonstrations). Bangkok,
Thailand: Association for Computational Linguistics.
Zou, A.; Wang, Z.; Carlini, N.; Nasr, M.; Kolter, J. Z.; and
Fredrikson, M. 2023. Universal and transferable adver-
sarial attacks on aligned language models. arXiv preprint
arXiv:2307.15043.
Appendix
Details about experimental setup
All experiments were conducted on a single machine
equipped with 5 NVIDIA RTX 4090 GPUs (24GB memory
each), using LLaMA Factory framework (Zheng et al.
2024). The poisoned training, Cleanup fine-tuning, and
cross-task alignemnt were each run for 3 epochs with 5000,
5000 and 467 training samples for SST-2, GSM8K and
MBPP datasets. These datasets cover sentiment classifica-
tion, maths reasoning and code completion tasks. We con-
struct our backdoor poison dataset with 2000 poison sam-
ples, hence our poisoning proportion is 40%. For models
over 1B, we add 2000 clean SST-2 samples for Qwen2.5-
1.5B and LLaMa3.2-1B at the Cross-task fine-tuning stage
for the data replay-based finetuning strategy (i.e., GSM8K
and MBPP). Otherwise, we don’t add SST-2 clean samples
for the vanilla full model update and FREEZE-based fine-
tuning methods. The total training time per fine-tuning stage
was approximately 1 GPU-hours.
Proofs to Theorem.1 and Corollary.1
Proof to Theorem.1
Lb(fθ) −Lc(fθ) =
EL
(Gθ,b−Gθ,c)dEL
E0
L
According to the Cauchy-Schwartz inequality, we derive:
∥Lb(fθ) −Lc(fθ)∥≤
EL
∥Gθ,b−Gθ,c∥∥dEL∥
E0
L
≤∥EL−E0
L∥sup
∥Gθ,b−Gθ,c∥
EL
where EL denotes the embeddings of an input prompt pro-
duced by the final layer of the LLM fθ. Since EL lies in a
bounded embedding space, we use ηto denote the diameter
of the embedding space where EL. We can further rewrite
Eq. as:
∥Lb(fθ) −Lc(fθ)∥≤η sup
∥Gθ,b−Gθ,c∥
EL
By taking the square of both sides, we can further reformu-
late Eq. as:
∥Lb(fθ) −Lc(fθ)∥2 ≤η2 (∥Gθ,b∥2 + ∥Gθ,c∥2
−2GT
θ,bGθ,c)
≤η2 (2β2
−2β2 GT
θ,bGθ,c
∥Gθ,b∥∥Gθ,c∥)
Taking the square root on both sides of Eq., we finally derive
Eq. and concludes the proof:
|Lb(fθ) −Lc(fθ)|≤βη 2(1−
GT
θ,bGθ,c
∥Gθ,b∥∥Gθ,c∥)
Proof to Corollary.1 Given the change of the model pa-
rameters δθ, we can derive:
Lb(fθ+δθ) ≤Lb(fθ) + β
2 ∥δθ∥2 + ∇TLbδθ
≤Lb(fθ) + β
2
∆2 + ∇TLbδθ
∂EL
∂Lc
∂θ
∂EL
With δθ=−ϵ∇θLθ (ϵis a positive constant), we reformu-
late Eq. as:
Lb(fθ+δθ) ≤Lb(fθ) + β
∆2
−ϵ∇TLb∇Lc
2
where ∇Lband ∇Lcare the loss gradient with respect to the
model parameters θ. Following the LLM model architecture
used in (Dai et al. 2022; Li et al. 2024b), we can obtain that:
∂Lb
∂EL
∂EL
∇Lb =
∂θ= Gθ,b
∂EL
∇Lc =
∂θ= Gθ,c
∂θ
∂EL
Given bounded gradients of LLM during the fine-tuning
process and fixed model parameters θ, there exists a positive
constant υ<= ∥∂EL
θ ∥that ∥∇Lb∥= υ∥∂T Lb
∂EL ∥= υ∥Gθ,b∥.
Similarly, we can get ∥∇Lc∥= υ∥Gθ,c∥. By injecting it into
Eq., we derive:
Lb(fθ+δθ) −Lb(fθ) ≤β
2
|Lb(fθ+δθ) −Lb(fθ)|≤β
2
β
∆2
∆2
∆2
=
2
≤β
2
β
∆2
=
∆2
2
−ϵ∇TLb∇Lc
∆
−
υG∇TLb∇Lc (given ϵ=
∆
−
υGTr(∂TEL
∂EL
∂θ
∂θ )GT
θ,bGθ,c
∆
−
υGυ2G2 GT
θ,bGθ,c
∥Gθ,b∥∥Gθ,c∥
GT
θ,bGθ,c
−∆υG
∥Gθ,b∥∥Gθ,c∥
∥δθ∥
∥∇Lb∥)