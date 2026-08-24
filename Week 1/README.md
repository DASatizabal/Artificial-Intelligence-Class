Week 1 In-Class Activity
Due Monday by 11:59pm Points 25 Submitting a text entry box or a file upload Available after Aug 24 at 6:30pm
Title: Design, Test, and Evaluate an Intelligent Agent Using Real-World-Style Data

Objective: This hands-on activity gives students practical experience applying core intelligent-agent concepts. Students will use the PEAS framework, task-environment classification, rationality, rule-based decision-making, data-quality analysis, visualization, model evaluation, and the bias-variance tradeoff to design and assess a simple intelligent agent.

By the end of the activity, students should be able to:

Explain how an intelligent agent perceives and responds to its environment.
Apply the PEAS framework to define an intelligent-agent task.
Classify the properties of an agent's environment.
Implement and evaluate a rule-based intelligent agent.
Identify data-quality issues that may affect AI/ML performance.
Compare the behavior of a simple Decision Tree and a Random Forest.
Interpret model performance using training accuracy, test accuracy, F1-score, and generalization gap.
Explain the relationship between model complexity, bias, variance, and generalization.
Recommend improvements that increase agent reliability and responsible use.
Activity Overview

Students will work in teams of 3–4 members using a 20,000-row intelligent-agent dataset representing eight real-world application areas.

The activity begins with a simple rule-based weather agent and progresses to a supervised machine-learning comparison. Students will analyze how an agent receives environmental information, makes a decision, evaluates the result, and improves its behavior.

The overall workflow is:

Environment → Percept → Rule or Model → Action → Evaluation → Reflection → Improvement

Students will complete a PEAS specification, analyze task-environment characteristics, inspect the dataset for quality issues, implement the rule-based agent, train machine-learning models, visualize results, compare model performance, and reflect on the strengths and limitations of different AI approaches.

Step 1: Group Formation — 5 Minutes

Divide students into collaborative groups of 3–4 members.

Each group should assign the following roles:

Agent Designer: Leads the PEAS specification and agent-architecture decisions.
Data Analyst: Loads the dataset, runs Python code, and analyzes model results.
Safety and Ethics Reviewer: Identifies risks, limitations, data-quality concerns, and responsible-AI considerations.
Presenter/Recorder: Documents findings and prepares the group's presentation and written summary.
Each group will select one intelligent-agent application:

Autonomous delivery robot
Smart hospital assistant
Cybersecurity monitoring agent
Intelligent tutoring system
Smart-home energy agent
Autonomous vehicle
Warehouse robot
AI travel assistant
If a group has only three members, the Presenter/Recorder role may be combined with another role.

Step 2: Hands-On Implementation

2.1 Load the Dataset

Students should load the provided dataset using:

import pandas as pd

df = pd.read_csv("week1_intelligent_agents_large_dataset.csv")

The dataset contains 20,000 observations representing intelligent-agent interactions and environmental conditions.

2.2 Open the Provided Resources

Students should open and review:

week1_intelligent_agents_large_dataset.csvDownload week1_intelligent_agents_large_dataset.csv
starter_script.pyDownload starter_script.py
Student Starter Notebook with TODO activitiesDownload Student Starter Notebook with TODO activities
2.3 Explore the Dataset

Students should examine:

Number of rows and columns
Column names
Data types
Missing values
Unique categorical values
Environment types
Agent architectures
Target variables
Expected actions
Potential outliers
Random/noise features
Recommended Python commands include:

print(df.shape)

print(df.head())

print(df.info())

print(df.isnull().sum())

print(df.describe())

Resources Provided

Large Dataset (.csv)
Starter Python Script (starter_script.py)
Student Starter Notebook with TODOs (.ipynb)
PEAS / Agent Design Worksheet

Each group must complete the following worksheet for its selected intelligent-agent application.

Component

Group Response

Agent

Performance Measure

Environment

Actuators

Sensors

Fully or Partially Observable

Deterministic or Stochastic

Episodic or Sequential

Static or Dynamic

Discrete or Continuous

Single-Agent or Multi-Agent

Recommended Agent Architecture

Rational Action Example

Irrational Action Example

Major Challenge

Groups must provide a short justification explaining why the selected agent architecture is appropriate for the task environment.

Possible architectures include:

Simple Reflex Agent
Model-Based Reflex Agent
Goal-Based Agent
Utility-Based Agent
Learning Agent
Step 3: Group Task — 20 Minutes

Part A — Build and Evaluate the Rule-Based Intelligent Agent

Students will first implement a simple rule-based agent.

A sample starting function is:

def agent_action(environment):

    if environment == "Sunny":

        return "Go Outside"

    elif environment == "Rainy":

        return "Use Umbrella"

    elif environment == "Cloudy":

        return "Stay Inside"

    elif environment == "Windy":

        return "Use Wind Protection"

    elif environment == "Snowy":

        return "Wear Winter Gear"

    elif environment == "Stormy":

        return "Stay Inside"

    else:

        return "Request Human Decision"

Students must:

Implement agent_action(environment).
Apply the function to the dataset.
Compare the generated action with expected_action.
Determine whether each decision is correct.
Calculate the overall rule-based accuracy.
Identify any cases that require a default action or human review.
Example:

df["agent_action"] = df["environment"].apply(agent_action)

df["correct"] = (

    df["agent_action"] == df["expected_action"]

)

accuracy = df["correct"].mean()

print(f"Rule-Based Agent Accuracy: {accuracy:.2%}")

Part B — Evaluate Data Quality

Before training a machine-learning model, students must inspect the quality of the dataset.

Groups should:

Count missing values.
Identify which variables contain incomplete observations.
Review numeric descriptive statistics.
Identify potential outliers.
Visualize the distribution of response_time_ms.
Discuss how missing sensor information could affect agent decision-making.
Select an appropriate preprocessing strategy.
Example:

print(df.isnull().sum())

print(df.select_dtypes(include="number").describe())

Visualization example:

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

df["response_time_ms"].dropna().hist(bins=40)

plt.title("Agent Response-Time Distribution")

plt.xlabel("Response Time (ms)")

plt.ylabel("Number of Episodes")

plt.show()

Part C — Bias-Variance Model Comparison

Students will compare two machine-learning approaches.

Model 1: Shallow Decision Tree

DecisionTreeClassifier(

    max_depth=3,

    random_state=42

)

The shallow Decision Tree intentionally limits model complexity and helps demonstrate higher bias.

Model 2: Random Forest

RandomForestClassifier(

    n_estimators=200,

    random_state=42

)

The Random Forest combines multiple decision trees and generally provides better stability and generalization.

Students should evaluate each model using:

Training accuracy
Test accuracy
Test F1-score
Generalization gap
The generalization gap can be calculated as:

Training Accuracy − Test Accuracy

Students should consider:

High bias: both training and test performance are relatively weak.
High variance: training performance is much stronger than test performance.
Better generalization: strong test performance with a relatively small training/test gap.
Part D — Visualization

Each group must produce at least two visualizations.

Visualization 1: Rule-Based Agent Performance

Example:

df["correct"].value_counts().plot(kind="bar")

plt.title("Rule-Based Agent Decision Results")

plt.xlabel("Correct Decision")

plt.ylabel("Number of Observations")

 

plt.show()

Visualization 2: Training vs. Test Accuracy

Students should compare the Decision Tree and Random Forest visually.

The chart should contain:

Descriptive title
Labeled x-axis
Labeled y-axis
Clearly identified models
Training and testing results
Students should capture screenshots of their charts for inclusion in the group paper.

Step 4: Research, Discussion, and Reflection — 30 Minutes

Groups will discuss their findings and evaluate the effectiveness of their intelligent agent and machine-learning models.

Students should discuss:

What worked well?
What did not work?
Which assumptions influenced the rule-based agent?
What happened when the agent encountered incomplete information?
How did missing data affect the analysis?
Which model generalized better?
Did the most complex model always produce the best result?
How did the training and test results illustrate bias and variance?
How might the agent behave differently in a dynamic or partially observable environment?
What safety or ethical issues should be considered?
When should human oversight remain part of the decision process?
Groups should summarize their findings for inclusion in the final submission.

Required Reflection Questions

Each group must answer the following five questions:

What key insights were revealed through your evaluation process?
Which model or technique provided the best results, and why?
What were the main challenges faced during the analysis?
How do your findings demonstrate the bias-variance tradeoff?
What steps could improve model performance and reliability?
Additional Intelligent-Agent Discussion Questions

Groups should also be prepared to discuss:

Does an intelligent agent have to learn in order to be considered intelligent?
Can a rational agent make a decision that produces a poor outcome?
How does partial observability affect agent decision-making?
Why might a simple reflex agent perform poorly in a dynamic environment?
What is the difference between achieving a goal and maximizing utility?
Why are performance measures important?
What could happen if an AI developer defines an inappropriate performance measure?
Which agent architecture is most appropriate for autonomous driving?
How can learning improve an agent's behavior?
What ethical and safety considerations should influence intelligent-agent design?
Step 5: Wrap-Up — 5 Minutes

The instructor will conclude the activity by reviewing the major concepts.

Students should remember:

Agents perceive and act.
PEAS helps define an intelligent-agent task environment.
Rationality depends on expected performance and available information.
A rational decision does not guarantee a successful outcome.
Rule-based systems are interpretable but can be inflexible.
Learning systems can adapt to patterns but depend heavily on data quality.
Training performance alone does not demonstrate generalization.
Test data provides evidence about how a model performs on unseen observations.
Bias and variance represent competing challenges in model development.
More complex AI is not automatically better AI.
Instructor Emphasizes

The instructor should reinforce the following principles:

Even minor data flaws can produce significant AI and machine-learning errors.
Data quality, preprocessing, and model evaluation are just as important as algorithm selection.
Poor sensor information can negatively affect intelligent-agent decisions.
A poorly defined performance measure can encourage undesirable behavior.
Accuracy alone does not establish fairness, safety, reliability, or rationality.
AI models should be evaluated using multiple metrics.
High-risk intelligent agents may require human oversight even when model performance appears strong.
Model complexity should be justified by improved generalization rather than complexity for its own sake.
Expected Deliverables

Each group should produce:

Completed PEAS worksheet
Completed task-environment classification
Working rule-based agent
Rule-based accuracy calculation
Data-quality analysis
Missing-value assessment
At least two visualizations
Decision Tree evaluation
Random Forest evaluation
Bias-variance comparison
Responses to the five required reflection questions
Screenshots of important outputs
Final APA-formatted written submission
Submission and Reminders

Submit a 1–2-page APA-formatted group paper that includes:

Brief introduction to the selected intelligent agent
PEAS specification
Task-environment classification
Relevant Python code excerpts
Explanation of the rule-based agent
Rule-based performance results
Data-quality findings
Decision Tree and Random Forest results
Bias-variance interpretation
Screenshots of output and visualizations
Responses to the reflection questions
Conclusion
References and in-text citations
Upload the completed paper through Canvas LMS in either Microsoft Word (.docx) or PDF (.pdf) format.

Due: Sunday at 11:59 PM.

Before submitting, each group should verify that:

The document opens correctly.
Screenshots are readable.
Figures include titles and labels.
Code excerpts are properly formatted.
References are complete.
Group members are identified.
The notebook runs successfully from beginning to end.
Activity Learning Outcome

At the conclusion of the activity, students should be able to explain the complete progression:

Environment → Sensors/Percepts → Agent → Rule or Learned Model → Action → Performance Evaluation → Feedback → Improvement

This progression provides a practical foundation for understanding how increasingly sophisticated AI systems move from simple reflex behavior toward model-based, goal-based, utility-based, and learning-agent architectures.
