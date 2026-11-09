# Provider Network Access and Cost Planner: Explained Simply

The big picture: the system is a new hire at a health plan's front desk. A member asks for help. The new hire understands the request, checks the rules, reads the paperwork, estimates how expensive the member will be, and finds them the right doctor.

- **AI** = the reasoning and planning side (finding, checking, assigning)
- **MI** = the learning side (predicting from data)

---

## Why AI and MI need each other

You don't need cost to find *a* doctor. You need it to find the **right** doctor when doctors are limited.

**1. Doctors fill up by workload, not headcount.**
A teacher with 25 students, 10 of whom need extra help every day, has a much harder class than one with 25 students who don't. A PCP with 1,500 healthy members is fine. A PCP with 1,500 very sick members is drowning. MI tells the AI planner who the "heavy" patients are, so no single doctor gets too many.

**2. When close doctors are scarce, the sickest go first.**
One open cardiologist 3 miles away, two members who need one. Member A is healthy. Member B has heart failure and two ER visits. B gets the close one, because a sick member forced to drive 20 miles skips care and ends up in the ER.

**3. Access drives cost.**
Members far from a doctor skip checkups. A problem a $150 visit would catch becomes a $15,000 hospital stay. So the AI's travel miles are one of the clues MI uses to predict cost.

**The loop:**

```
AI (A*) measures distance  ->  MI uses distance to predict cost
MI predicts cost           ->  AI (CSP) uses cost to assign doctors fairly
```

Real-world version: when a big provider group leaves the network, the network team asks "where do these members go?" and also "which doctors can absorb them, and who needs to be placed first?"

---

## Part 1: Find the route, estimate the cost

**In one line:** AI finds and assigns the doctor. MI estimates the member's cost in dollars.

### AI Part 1: Search + CSP

| Tool | Job | Analogy |
|---|---|---|
| A* search | Finds how far each doctor is (travel miles) | The GPS |
| CSP | Picks *which* doctor each member gets, following the rules (capacity, distance, specialty, not too many expensive patients per doctor) | The wedding seating chart |

- **A\*** scores each step as f = g + h: g is the miles already traveled, h is a guess of the miles left (count blocks, no diagonals). The guess never overestimates, so the first route found is the shortest.
- **CSP pieces:**
  - **Backtracking:** try a seat; at a dead end, undo and try another.
  - **MRV:** seat the pickiest guest first.
  - **Forward checking:** after each seat, cross off the seats that are no longer possible.
  - **Min-conflicts:** when a doctor leaves, fix only the people who lost their seat.

### MI Part 1: EDA + Regression

| Tool | Job | Analogy |
|---|---|---|
| EDA | Look at the data first to understand it | A mechanic popping the hood before fixing anything |
| Linear regression | Predicts the dollar amount ("this member will cost about $9,800 next quarter") | Drawing the best straight line through the data |
| MAE and RMSE | The report card: how far off the predictions were | Grading the guess. MAE = average miss in dollars; RMSE punishes big misses more |

**What the model looks at (the clues), like a car insurance quote:**

| Clue | Why it predicts cost |
|---|---|
| Age | Older members usually need more care |
| Chronic conditions | Ongoing visits and drugs |
| Prior cost | Expensive last year often means expensive next year |
| ER visits | Pricey, and a sign of unmanaged problems |
| Hospital admissions | Even more expensive |
| Travel miles to PCP | **Comes from AI's A\***. Far away means skipped checkups |
| Saw PCP recently | Checkups catch problems early |
| Plan type and county | Different benefits and local prices |

**No peeking rule (leakage):** every clue must be something we'd know *before* the quarter we're predicting.

**Note:** Part 1 gives a **dollar estimate**. The yes/no **risk flag** comes in Part 2.

---

## Part 2: Sort things into buckets

**In one line:** AI sorts paperwork into types. MI sorts members into "high-cost" or "not high-cost." Both are **classification**: putting things into buckets.

### AI Part 2: Perception (reading paperwork)

**Why the project needs it:** members and doctors send forms, and each form type kicks off a different job.

| Form | What happens next |
|---|---|
| Referral | Triggers a doctor search (AI Part 1) |
| Claim form | Becomes cost data for MI |
| Provider enrollment form | Adds a new doctor to the directory |

Analogy: the mailroom sorter. If a claim lands in the referral pile, the member waits and the cost data goes missing.

| Tool | Job | Analogy |
|---|---|---|
| Flatten | Turns the image into one long list of numbers (each pixel = a number) | Unrolling a paint-by-numbers sheet into a single line |
| Softmax classifier | Scores each form type and turns scores into percentages ("85% claim form, 10% referral, 5% enrollment") | A judge's scorecard that must add up to 100% |
| Confusion matrix | Shows which form got mistaken for which | A mix-up chart: "referrals got filed as claims 6 times" |
| CNN | Looks at small patches to find lines, boxes, logos | A magnifying glass sliding across the page |
| Augmentation | Trains on crooked, blurry copies | Practicing reading bad handwriting so messy scans don't fool it |

### MI Part 2: Classification Pipeline (the risk flag)

**Why the project needs it:** the AI planner needs a simple yes/no ("is this a heavy patient?") to spread them across doctors and decide who gets placed first.

| Tool | Job | Analogy |
|---|---|---|
| Pipeline | Cleans the data the same way every time: scales numbers, turns categories into 0/1 columns | An assembly line |
| Logistic regression | Gives a probability of being high-cost | A weighted checklist that adds up to a percentage |
| Decision tree | Asks yes/no questions to reach an answer ("Over 65? More than 2 ER visits?") | The game 20 Questions |
| Stratified CV | Tests the model 5 times, each with the same mix of high-cost members | 5 practice exams with the same difficulty mix |
| Confusion matrix | Right and wrong flags, side by side | Same chart as AI Part 2 |

**The accuracy trap:** if only 10% of members are high-cost, a lazy model that says "no" to everyone scores 90% and catches **nobody**. So we grade with:

| Metric | Plain question it answers |
|---|---|
| Precision | Of the members we flagged, how many really were high-cost? |
| Recall | Of the members who really were high-cost, how many did we catch? |
| ROC-AUC | Pick one high-cost and one normal member at random. How often does the model rank the high-cost one higher? |

**How Part 2 AI and MI connect:**
- Same skill (sorting into buckets), and the same report card (the confusion matrix).
- AI reads the claim forms, and those claims become the data MI learns from.
- MI's high-cost flag goes straight to the AI planner (Part 1's CSP) to spread heavy patients and set placement order.
