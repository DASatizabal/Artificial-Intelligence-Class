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

### What we actually built for AI Part 1 (Nov 9)

- **600 made-up members**, each with a home location, the type of doctor they need, and a stand-in high-cost yes/no (until MI Part 2 builds the real one).
- **Open seats:** only in-network doctors accepting new patients have seats, and the seats add up to 115% of what's needed, so it's tight on purpose. Each doctor can take high-cost members for at most 35% of their seats.
- **Each specialty is its own puzzle.** Cardiology seats have nothing to do with dermatology seats, so we solve them separately. Smaller puzzles are faster.

**The race between methods (same puzzles, same 200,000-step limit):**

| Method | Plain meaning | Solved (of 7) | Steps |
|---|---|---|---|
| Plain backtracking | Seat people in list order, undo when stuck | 5 | 416,759 |
| + Forward checking | After each seat, cross off full doctors for everyone else | 5 | 408,276 |
| MRV + forward checking | Also seat the pickiest member first | 6 | 200,237 |
| + LCV | Also try the doctor that blocks the fewest others | 6 | 200,237 |
| **+ Look-ahead** | **Before each seat, check that everyone still waiting can fit** | **7** | **480, no undoing** |

**Why the look-ahead was needed:** forward checking only notices when ONE member has no seat left. It can't see when a GROUP of members is fighting over too few seats, so the search walks into a dead end and keeps undoing. The look-ahead checks the whole group each time. Analogy: before seating a guest, the planner quickly counts whether the remaining guests still fit in the remaining chairs.

**Why no AC-3:** AC-3 checks rules between two guests at a time. "A doctor has 8 seats" involves everyone who could pick that doctor, so it isn't a two-guest rule.

**Results:**
- 465 of 600 members (77.5%) placed inside the federal distance limit, 4.88 miles on average, no doctor over capacity.
- 77 members have **no doctor close enough** (fix: recruit doctors there).
- 58 members have **close doctors that are full** (fix: more open seats).
- Only cardiology and endocrinology hit the 90% target. Primary care was worst (71%) because its limit is just 5 miles.

**Stress test: a big doctor group leaves.** Pinecrest Medical Group (15 doctors) leaves, and 73 members lose their doctor.

| Way to recover | Placed afterward | Unaffected members moved |
|---|---|---|
| Min-conflicts (fix only what broke) | 401 | 14 |
| Start over from scratch | 402 | 26 |

Almost the same coverage, but repair moves fewer people. Every move is a member forced to change doctors.

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

### What we actually built for MI Part 1 (Nov 9)

**The data:** 1,500 made-up members followed month by month from January 2025 to September 2026, which comes to 31,563 rows. The first 600 are the same members from AI Part 1. "Miles to PCP" comes straight from AI Part 1's A* search.

**The audit (popping the hood):**

| Found | What we did |
|---|---|
| 63 rows loaded twice | Removed |
| 472 rows missing plan type | Labeled "Unknown" instead of throwing them away |
| 40 negative monthly costs | Kept, because they're refunds (claim reversals) |
| Middle member costs $435 a quarter, average is $1,736 | The top 10% of rows are 66% of all cost. A few people drive most of the spending |

**What the data showed:**
- More chronic conditions means more cost: about $900 a quarter with none, over $10,000 with 5 or more.
- **Distance matters for the far-away group.** Members 9+ miles from a doctor:
  - only 14% saw their doctor recently (vs 69% for members within 2 miles)
  - went to the ER about twice as often
  - cost about $2,700 vs $1,840
- **One number can hide a pattern.** The overall correlation between miles and cost is almost zero (0.03), because most members live close. The effect only shows up when you look at the far-away group by itself.

**The no-peeking split:**
- Train on January to September 2025, test on January to February 2026.
- October to December 2025 is skipped on purpose. A September row's answer covers October to December. Training past September would let the answers leak into the test months.

**The scorecard (average miss on the test months):**

| Model | Average miss | Plain meaning |
|---|---|---|
| Guess the average for everyone | $2,446 | The bar to beat |
| "Next quarter = last quarter" | $2,610 | Worse! One bad quarter usually isn't repeated |
| **Linear regression** | **$2,168** | **Beats both. This is the baseline future models must beat** |
| Log version | $1,834 | Closer for typical members, but misses the huge bills by more |

**Honest weak spots:** the straight line predicts below $0 for 126 rows, and it can't reach the giant bills. That's why Part 2 builds a yes/no high-cost flag and later Parts use models that can bend.

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
